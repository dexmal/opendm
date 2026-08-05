import logging
import time

from utils.enums import ReturnCode


def process_job(
    client,
    gpu_client,
    job_id,
    robot_id,
    image_size,
    image_type,
    action_type,
    duration,
    prompt=None,
    task_name=None,
    max_wait=600,
    resize_name=None,
):
    """Handles the processing of a single job."""
    try:
        device, status = client.get_job_status(job_id)
        logging.info(f"Processing job_id: {job_id}, status: {status}")
        if status == "ready":
            client.update_job_info(job_id, robot_id)
            r = client.start_robot(job_id)
            logging.info(f"Started robot: {r.content}")
            if r.status_code == 200:
                wait_result = client.wait_for_robot_running(job_id)
                if wait_result != ReturnCode.SUCCESS:
                    logging.warning(
                        f"Job {job_id} failed to reach running state: {wait_result}"
                    )
                    return

                if hasattr(gpu_client, "reset_policy"):
                    gpu_client.reset_policy()

                start_time = time.time()
                while True:
                    device, status = client.get_job_status(job_id)
                    if status != "running":
                        logging.info(f"Job {job_id} left running state: {status}")
                        break
                    state = client.get_state(
                        image_size,
                        image_type,
                        action_type,
                        resize_name=resize_name,
                    )
                    if not state:
                        time.sleep(0.5)
                        continue
                    if state["state"] != "normal" or state["pending_actions"] != 0:
                        time.sleep(0.5)
                        continue
                    logging.info(
                        "get_robot_state time: %.2f", time.time() - state["timestamp"]
                    )
                    result = gpu_client.infer(state, prompt=prompt, task_name=task_name)
                    logging.info(f"Inference result: {result}")
                    client.post_actions(result, duration, action_type)
                    if time.time() - start_time > max_wait:
                        logging.warning(f"Job {job_id} exceeded max wait time.")
                        break
    except Exception as e:
        logging.error(f"Error processing job {job_id}: {e}")


def job_loop(
    client,
    gpu_client,
    submission_id,
    image_size,
    image_type,
    action_type,
    duration,
    resize_name=None,
    target_robot_type=None,
    target_task_name=None,
    target_run_id=None,
):
    """Poll active runs under a submission and process ready jobs."""
    ACTIVE_STATES = ["pending", "assigned", "prepare", "ready", "running"]
    MAX_EMPTY_POLLS = 10
    empty_poll_count = 0
    current_run_id = None
    current_prompt = None
    current_task_name = None

    def normalize_robot_tag(value):
        normalized = str(value or "").strip().lower()
        aliases = {
            "arx5": "arx5",
            "ur5": "ur5",
            "aloha": "aloha",
            "w1": "w1",
            "dm05_dosw1": "w1",
        }
        return aliases.get(normalized, normalized)

    target_robot_type = normalize_robot_tag(target_robot_type)
    target_run_id = str(target_run_id or "").strip()

    while True:
        try:
            job_collections = client.get_all_runs(submission_id)
            if not isinstance(job_collections, list):
                logging.error(
                    f"Unexpected runs response for submission {submission_id}: {job_collections}"
                )
                time.sleep(2)
                continue
        except Exception as e:
            logging.error(
                f"Error fetching job collections for submission {submission_id}: {e}"
            )
            time.sleep(2)
            continue

        target_job_collection = None
        filtered_job_collections = [
            job_collection
            for job_collection in job_collections
            if (
                not target_robot_type
                or normalize_robot_tag(job_collection.get("robotTag"))
                == target_robot_type
            )
            and (
                not target_task_name
                or job_collection.get("task_name") == target_task_name
            )
            and (not target_run_id or job_collection.get("run_id") == target_run_id)
        ]

        if current_run_id is not None:
            target_job_collection = next(
                (
                    job_collection
                    for job_collection in filtered_job_collections
                    if job_collection.get("run_id") == current_run_id
                    and job_collection.get("status") in ACTIVE_STATES
                ),
                None,
            )

        if target_job_collection is None:
            for preferred_status in [
                "prepare",
                "ready",
                "running",
                "assigned",
                "pending",
            ]:
                target_job_collection = next(
                    (
                        job_collection
                        for job_collection in filtered_job_collections
                        if job_collection.get("status") == preferred_status
                    ),
                    None,
                )
                if target_job_collection is not None:
                    break

        if target_job_collection is None:
            if current_run_id is not None:
                logging.info(
                    f"Run {current_run_id} is no longer active, waiting for the next run..."
                )
                current_run_id = None
                current_prompt = None
                current_task_name = None
            else:
                if target_run_id:
                    logging.info(
                        f"No active run {target_run_id} found for submission {submission_id}, waiting..."
                    )
                else:
                    logging.info(
                        f"No active run found for submission {submission_id}, waiting..."
                    )
            empty_poll_count = 0
            time.sleep(2)
            continue

        selected_run_id = target_job_collection["run_id"]
        if selected_run_id != current_run_id:
            current_run_id = selected_run_id
            current_prompt = target_job_collection.get("prompt")
            current_task_name = target_job_collection.get("task_name")
            empty_poll_count = 0
            task_name = current_task_name
            robot_tag = target_job_collection["robotTag"]
            status = target_job_collection["status"]
            logging.info(
                f"job_collection id: {current_run_id}, task name: {task_name}, prompt: {current_prompt}, robot tag: {robot_tag}, status: {status}"
            )
            if hasattr(gpu_client, "prepare_task"):
                gpu_client.prepare_task(
                    task_name=current_task_name, prompt=current_prompt
                )

        try:
            job_collection = client.get_all_jobs(current_run_id)
        except Exception as e:
            logging.error(f"Error fetching jobs for run {current_run_id}: {e}")
            time.sleep(2)
            continue

        jobs = job_collection.get("jobs") or []

        has_active_job = False
        exit_code = 0
        for job in jobs:
            status = job["status"]
            if status in ACTIVE_STATES:
                has_active_job = True
                break
            elif status in ["finished", "cancelled", "failed"]:
                exit_code += 1

        if not has_active_job and exit_code == len(jobs):
            empty_poll_count += 1
            logging.info(
                f"No active jobs for run {current_run_id}, poll count: {empty_poll_count}"
            )
            if empty_poll_count >= MAX_EMPTY_POLLS:
                logging.info(
                    f"Run {current_run_id} appears complete, switching back to run polling."
                )
                current_run_id = None
                current_prompt = None
                current_task_name = None
                empty_poll_count = 0
                time.sleep(1)
                continue
            time.sleep(1)
            continue
        else:
            empty_poll_count = 0

        for job in jobs:
            job_id = job["job_id"]
            status = job["status"]
            logging.info(
                f"Job id: {job_id}, status: {status}, remaining jobs: {len(jobs)}"
            )
            if status == "ready":
                device = job.get("device") or {}
                robot_id = device.get("robot_id")
                if not robot_id:
                    logging.warning(
                        f"Job {job_id} is ready but missing robot_id, skipping this poll."
                    )
                    continue
                task_runtime = {}
                if hasattr(gpu_client, "get_task_runtime"):
                    task_runtime = gpu_client.get_task_runtime(current_task_name) or {}
                process_job(
                    client,
                    gpu_client,
                    job_id,
                    robot_id,
                    image_size,
                    task_runtime.get("image_type", image_type),
                    task_runtime.get("action_type", action_type),
                    task_runtime.get("duration", duration),
                    prompt=current_prompt,
                    task_name=current_task_name,
                    resize_name=task_runtime.get("resize_name", resize_name),
                )

        time.sleep(1)
