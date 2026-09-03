import functools
import logging
import pickle
import threading
import time
import uuid
from enum import Enum

import numpy as np
import requests

base_url = "http://api.robochallenge.cn"

MAX_RETRY = 3
RETRY_DELAY = 1
MAX_ACTION_POST_ATTEMPTS = 5


class ReturnCode(int, Enum):
    SUCCESS = 0
    FAILURE = 1
    TIMEOUT = 2
    EXCEPTION = 3


def setup_logger(name=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def timeout(seconds):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [
                Exception(
                    f"Function '{func.__name__}' timed out after {seconds} seconds."
                )
            ]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                print(f"Function '{func.__name__}' timed out after {seconds} seconds.")
                return ReturnCode.TIMEOUT
            if isinstance(result[0], Exception):
                print("res", result[0])
                return ReturnCode.EXCEPTION

            return result[0]

        return wrapper

    return decorator


def retry_request(retries=3, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e
                    if attempt < retries - 1:
                        time.sleep(delay)
            raise last_exception

        return wrapper

    return decorator


logger = setup_logger()


class InterfaceClient:
    def __init__(self, user_id):
        self.user_id = user_id
        self.session = requests.Session()
        self.job_id = None
        self.robot_id = None
        self.robot_url = None
        self.clock_offset = None

    def _get(self, url, **kwargs):
        @retry_request(retries=MAX_RETRY, delay=RETRY_DELAY)
        def inner():
            return self.session.get(url, **kwargs)

        return inner()

    def _post(self, url, **kwargs):
        @retry_request(retries=MAX_RETRY, delay=RETRY_DELAY)
        def inner():
            return self.session.post(url, **kwargs)

        return inner()

    @staticmethod
    def _print_response_error(prefix: str, error: requests.exceptions.RequestException):
        response = getattr(error, "response", None)
        if response is None:
            print(prefix)
            return
        print(f"{prefix} status={response.status_code} body={response.text}")

    def update_job_info(self, job_id, robot_id):
        self.job_id = job_id
        self.robot_id = robot_id
        self.robot_url = base_url + f"/robots/{robot_id}/direct"
        self.clock_offset = self.cal_clockoffset()
        print(f"clock jitter:{self.clock_offset}s")

    def cal_clockoffset(self):
        offsets = []
        while True:
            try:
                for _ in range(10):
                    t1 = time.time()
                    response = self._get(
                        f"{self.robot_url}/clock-sync",
                        headers={"x-user-id": self.user_id},
                    )
                    response.raise_for_status()
                    t2 = float(response.json()["timestamp"])
                    t3 = time.time()
                    offset = ((t2 - t1) + (t2 - t3)) / 2
                    offsets.append(offset)
                    time.sleep(0.5)
                break
            except requests.exceptions.RequestException as e:
                print(f"Error getting clock: {e}")
                time.sleep(0.5)
                continue
        return float(np.array(offsets).mean())

    def get_state(self, image_size, image_type, action_type, resize_name=None):
        try:
            url = f"{self.robot_url}/state.pkl"
            params = {
                "width": image_size[0],
                "height": image_size[1],
                "image_type": image_type,
                "action_type": action_type,
            }
            if resize_name:
                params["resize_name"] = resize_name

            response = self._get(
                url, params=params, headers={"x-user-id": self.user_id}
            )
            response.raise_for_status()
            data = pickle.loads(response.content)
            if isinstance(data, dict) and data.get("status") == "size_none":
                print("Warning: Robot state not ready (size is None)!")
                print("test state:", data)
            return data
        except requests.exceptions.RequestException as e:
            self._print_response_error(f"Error getting state: {e}", e)
            return None

    def post_actions(self, actions, duration, action_type):
        for attempt in range(1, MAX_ACTION_POST_ATTEMPTS + 1):
            try:
                req_hash = f"gpu-server-{uuid.uuid4()}"
                url = f"{self.robot_url}/action?hash={req_hash}"
                send_data = {"actions": actions, "duration": duration}
                response = self._post(
                    url,
                    params={"action_type": action_type},
                    json=send_data,
                    headers={"x-user-id": self.user_id},
                )

                response.raise_for_status()
                body = response.json()
                if body.get("result") == "success":
                    return

                print(
                    "Robot failed to process actions "
                    f"(attempt {attempt}/{MAX_ACTION_POST_ATTEMPTS}): "
                    f"{body.get('message')}"
                )
                print(f"Robot action response body: {response.text}")
            except requests.exceptions.RequestException as e:
                self._print_response_error(
                    f"Error posting actions "
                    f"(attempt {attempt}/{MAX_ACTION_POST_ATTEMPTS}): {e}",
                    e,
                )

        print(f"Failed to post actions after {MAX_ACTION_POST_ATTEMPTS} attempts.")

    def start_robot(self, job_id):
        url = f"{base_url}/jobs/update"
        response = self._post(
            url,
            json={"job_id": job_id, "action": "start"},
            headers={"x-user-id": self.user_id},
        )
        return response

    def _get_job_status(self, job_id):
        response = self._get(
            f"{base_url}/jobs/{job_id}", headers={"x-user-id": self.user_id}
        )
        return response.json()

    @timeout(600)
    def wait_for_robot_running(self, job_id, poll_interval=2):
        while True:
            res = self._get_job_status(job_id)
            print(res)
            if res and "status" in res:
                if res["status"] == "running":
                    return ReturnCode.SUCCESS
                elif res["status"] == "prepare":
                    pass
                else:
                    return ReturnCode.FAILURE
            time.sleep(poll_interval)

    def get_job_status(self, job_id):
        response = self._get_job_status(job_id)
        print(job_id, response)
        return response["device"], response["status"]

    def get_all_jobs(self, job_collection_id):
        response = self._get(
            f"{base_url}/job_collections/{job_collection_id}",
            headers={"x-user-id": self.user_id},
        )
        return response.json()

    def get_all_runs(self, submission_id):
        """Get all runs for a submission."""
        response = self._get(
            f"{base_url}/v2/job_collections/submission/{submission_id}/runs",
            headers={"x-user-id": self.user_id},
        )
        return response.json()
