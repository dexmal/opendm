import pickle
import time
import uuid

import numpy as np
import requests
from utils.enums import ReturnCode
from utils.log import setup_logger
from utils.util import retry_request, timeout

logger = setup_logger()

base_url = "http://api.robochallenge.cn"
mock_url = "http://127.0.0.1:9098"

MAX_RETRY = 3
RETRY_DELAY = 1


class InterfaceClient:
    def __init__(self, user_id, mock=False):
        self.user_id = user_id
        self.session = requests.Session()
        self.job_id = None
        self.robot_id = None
        self.robot_url = None
        self.clock_offset = None
        self.mock = mock

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

    def _put(self, url, **kwargs):
        @retry_request(retries=MAX_RETRY, delay=RETRY_DELAY)
        def inner():
            return self.session.put(url, **kwargs)

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
        if self.mock:
            self.robot_url = mock_url + "/"
        self.clock_offset = self.cal_clockoffset()
        print(f"clock jitter:{self.clock_offset}s")

    def reset_job_info(self):
        self.job_id = None
        self.robot_id = None
        self.robot_url = None
        self.clock_offset = None
        self.mock = False

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

    def start_motion(self):
        url = f"{self.robot_url}/start_motion"
        response = self._get(url)
        return response

    def end_motion(self):
        url = f"{self.robot_url}/stop_motion"
        response = self._get(url)
        return response

    def post_actions(self, actions, duration, action_type):
        i = 0
        while i < 5:
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
                if response.json().get("result") == "success":
                    break
                else:
                    print(
                        f"Robot failed to process actions: {response.json().get('message')}"
                    )
                    print(f"Robot action response body: {response.text}")
            except requests.exceptions.RequestException as e:
                i += 1
                self._print_response_error(f"Error posting actions: {e}", e)

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

    def wait_for_robot_ready(self, job_id, poll_interval=2):
        while True:
            res = self._get_job_status(job_id)
            if "device" in res and "robot_id" in res:
                robot_id = res["device"]["robot_id"]
                return robot_id, job_id
            time.sleep(poll_interval)

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
