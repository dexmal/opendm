import datetime
import os
import re
import sys

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from robot.interface_client import InterfaceClient
from robot.job_worker import job_loop
from runner import InferenceRunner
from utils.constants import get_robot_image_config, get_task_metadata


def get_runtime_profile_from_config(cfg: DictConfig, robot_type: str) -> dict:
    robot_profiles = cfg.get("robot_profiles") or {}
    profile = robot_profiles.get(robot_type)
    if profile is None:
        available = (
            list(robot_profiles.keys()) if hasattr(robot_profiles, "keys") else []
        )
        raise ValueError(
            f"Unknown runtime profile for robot: {robot_type}. Available: {available}"
        )
    profile = OmegaConf.to_container(profile, resolve=True)
    profile["runtime_args"] = dict(profile.get("runtime_args") or {})
    return profile


def to_plain_dict(value) -> dict:
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return dict(OmegaConf.to_container(value, resolve=True))


def get_task_override(cfg: DictConfig, task_name: str | None) -> dict:
    if not task_name:
        return {}
    task_overrides = cfg.get("task_overrides") or {}
    if task_name not in task_overrides:
        return {}
    return to_plain_dict(task_overrides.get(task_name))


def build_effective_runtime(
    cfg: DictConfig,
    robot_type: str,
    task_name: str | None,
    prompt: str,
    run_label: str,
) -> dict:
    runtime_profile = get_runtime_profile_from_config(cfg, robot_type)
    task_override = get_task_override(cfg, task_name)

    default_image_type, image_mapping = get_robot_image_config(robot_type)
    runtime_args = dict(runtime_profile.get("runtime_args") or {})
    runtime_args.update(to_plain_dict(cfg.get("runtime_args")))
    runtime_args.update(to_plain_dict(task_override.get("runtime_args")))

    action_playback_target_steps = task_override.get("action_playback_target_steps")
    if action_playback_target_steps is None:
        action_playback_target_steps = cfg.get("action_playback_target_steps")
    if action_playback_target_steps is None:
        action_playback_target_steps = runtime_profile.get(
            "action_playback_target_steps", 0
        )

    debug = bool(task_override.get("debug", cfg.get("debug", False)))
    runtime_args.setdefault("debug", debug)
    if robot_type in ("arx5", "ur5"):
        runtime_args.setdefault(
            "action_playback_target_steps", action_playback_target_steps
        )

    # Checkpoint / norm / opendm root are model-level settings. Generalist task
    # overrides only tune runtime behavior for a shared model.
    checkpoint = cfg.get("checkpoint")
    norm_stats = cfg.get("norm_stats")
    if not norm_stats:
        norm_stats = runtime_args.pop("norm_stats", None)

    policy_type = cfg.get("policy_type") or "dm05"
    if str(policy_type).lower() == "auto":
        policy_type = "dm05"
    if not norm_stats and str(policy_type).startswith("dm05"):
        norm_stats = os.path.join(str(checkpoint), "norm_stats.json")

    policy_robot_type = cfg.get("policy_robot_type") or runtime_profile.get(
        "policy_robot_type", robot_type
    )
    action_type = task_override.get(
        "action_type",
        cfg.get("action_type")
        or runtime_profile.get("action_type")
        or ("joint" if robot_type in ["aloha", "w1"] else "leftpos"),
    )
    action_horizon = task_override.get("action_horizon")
    if action_horizon is None:
        action_horizon = cfg.get("action_horizon") or runtime_profile.get(
            "action_horizon", 15
        )
    duration = task_override.get("duration")
    if duration is None:
        duration = cfg.get("duration") or runtime_profile.get("duration", 0.1)
    image_type = task_override.get(
        "image_type", cfg.get("image_type") or default_image_type
    )
    resize_name = task_override.get("resize_name", cfg.get("resize_name"))
    if not resize_name:
        resize_name = runtime_args.pop("resize_name", None)
    code_root = (
        cfg.get("code_root")
        or cfg.get("opendm_root")
        or runtime_profile.get("code_root")
        or os.environ.get("OPENDM_ROOT")
        or None
    )
    if isinstance(code_root, str) and not code_root.strip():
        code_root = None
    return {
        "task_name": task_name,
        "prompt": prompt,
        "run_label": run_label,
        "robot_type": robot_type,
        "policy_robot_type": policy_robot_type,
        "policy_type": policy_type,
        "checkpoint": checkpoint,
        "norm_stats": norm_stats,
        "code_root": code_root,
        "action_type": action_type,
        "action_horizon": int(action_horizon),
        "action_playback_target_steps": int(action_playback_target_steps or 0),
        "duration": float(duration),
        "image_type": list(image_type),
        "image_mapping": image_mapping,
        "resize_name": resize_name,
        "runtime_args": runtime_args,
        "debug": debug,
        "debug_image_limit": task_override.get(
            "debug_image_limit", cfg.get("debug_image_limit", 3)
        ),
    }


def apply_code_root(code_root: str | None) -> str | None:
    if not code_root:
        return None
    code_root = os.path.abspath(os.path.expanduser(str(code_root)))
    os.environ["OPENDM_ROOT"] = code_root
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    return code_root


def safe_path_part(value: str | None, default: str = "unknown") -> str:
    value = str(value or "").strip() or default
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)


def build_log_dir(log_base_dir: str, runtime: dict) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = safe_path_part(runtime["run_label"])
    task_name = safe_path_part(runtime.get("task_name"), "generalist")
    run_id = safe_path_part(runtime.get("run_id"), "no_run_id")
    return os.path.join(log_base_dir, run_label, task_name, run_id, timestamp)


def create_runner_from_runtime(runtime: dict, image_size, log_base_dir: str):
    code_root = apply_code_root(runtime.get("code_root"))
    log_dir = build_log_dir(log_base_dir, runtime)
    task_name = runtime.get("task_name")
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "runtime.log"), level="DEBUG", enqueue=False)

    logger.info(f"Task: {task_name or '<generalist>'}")
    logger.info(f"Checkpoint: {runtime['checkpoint']}")
    logger.info(f"Robot type: {runtime['robot_type']}")
    logger.info(f"Policy robot type: {runtime['policy_robot_type']}")
    logger.info(f"Policy type: {runtime['policy_type']}")
    logger.info(f"Action type: {runtime['action_type']}")
    logger.info(f"Action horizon: {runtime['action_horizon']}")
    logger.info(
        f"Action playback target steps: {runtime['action_playback_target_steps']}"
    )
    logger.info(f"Duration: {runtime['duration']}")
    logger.info(f"Image type: {runtime['image_type']}")
    logger.info(f"Resize name: {runtime['resize_name'] or '<omitted>'}")
    logger.info(f"Norm stats: {runtime['norm_stats']}")
    logger.info(f"Code root: {code_root}")
    logger.info(f"Debug: {runtime['debug']}")
    logger.info(f"Runtime args: {runtime['runtime_args']}")
    logger.info(f"Log dir: {log_dir}")

    # Import policy after code_root/OPENDM_ROOT is on sys.path so opendm
    # resolves from the configured checkout.
    from policies import get_policy

    policy = get_policy(
        ckpt_path=runtime["checkpoint"],
        policy_type=runtime["policy_type"],
        prompt=runtime["prompt"],
        robot_type=runtime["policy_robot_type"],
        action_type=runtime["action_type"],
        action_horizon=runtime["action_horizon"],
        task_name=runtime["task_name"] or runtime["run_label"],
        image_shape=tuple(image_size),
        norm_stats=runtime["norm_stats"],
        **runtime["runtime_args"],
    )

    return InferenceRunner(
        policy=policy,
        robot_type=runtime["robot_type"],
        action_type=runtime["action_type"],
        task_name=runtime["task_name"] or runtime["run_label"],
        image_type=runtime["image_type"],
        image_mapping=runtime["image_mapping"],
        action_horizon=runtime["action_horizon"],
        debug=runtime["debug"],
        debug_image_limit=runtime["debug_image_limit"],
        log_dir=log_dir,
    )


class TaskAwareRunner:
    """Apply per-task runtime overrides while sharing one generalist model."""

    def __init__(self, cfg: DictConfig, robot_type: str, image_size, log_base_dir: str):
        self.cfg = cfg
        self.robot_type = robot_type
        self.image_size = image_size
        self.log_base_dir = log_base_dir
        self.runtimes = {}
        self.default_runner = None
        self.default_run_label = f"{robot_type}_generalist"
        self.target_run_id = str(cfg.get("run_id", "") or "").strip()
        self.task_log_dirs = {}
        self.default_runtime = build_effective_runtime(
            cfg,
            robot_type=robot_type,
            task_name=None,
            prompt="",
            run_label=self.default_run_label,
        )
        self.default_runtime["run_id"] = self.target_run_id
        self.default_runner = create_runner_from_runtime(
            self.default_runtime,
            image_size=image_size,
            log_base_dir=log_base_dir,
        )

    def _runtime_for_task(self, task_name: str | None, prompt: str | None) -> dict:
        if not task_name or not get_task_override(self.cfg, task_name):
            return self.default_runtime
        if task_name not in self.runtimes:
            self.runtimes[task_name] = build_effective_runtime(
                self.cfg,
                robot_type=self.robot_type,
                task_name=task_name,
                prompt=prompt or "",
                run_label=self.default_run_label,
            )
            self.runtimes[task_name]["run_id"] = self.target_run_id
        elif prompt:
            self.runtimes[task_name]["prompt"] = prompt
        return self.runtimes[task_name]

    def _ensure_runner(self):
        if self.default_runner is None:
            self.default_runner = create_runner_from_runtime(
                self.default_runtime,
                image_size=self.image_size,
                log_base_dir=self.log_base_dir,
            )
        return self.default_runner

    def _apply_runtime(self, runtime: dict, prompt: str | None, task_name: str | None):
        runner = self._ensure_runner()
        runner.action_type = runtime["action_type"]
        runner.image_type = runtime["image_type"]
        runner.image_mapping = runtime["image_mapping"]
        runner.action_horizon = runtime["action_horizon"]
        if prompt:
            runner.set_task_context(task_name=task_name, prompt=prompt)
        elif task_name:
            runner.set_task_context(task_name=task_name)

        self._apply_debug_context(runner, runtime, task_name)

        policy = runner.policy
        if hasattr(policy, "action_type"):
            policy.action_type = runtime["action_type"]
        if hasattr(policy, "action_horizon"):
            policy.action_horizon = runtime["action_horizon"]
        if hasattr(policy, "action_playback_target_steps"):
            policy.action_playback_target_steps = runtime[
                "action_playback_target_steps"
            ]
        # Keep logical-step history hop locked to the slot grid. Prefer explicit
        # history_action_step_increment, then playback. Do not fall through to
        # action_horizon when logical-step history is already configured.
        if hasattr(policy, "sync_history_action_step_increment"):
            history_step = runtime["runtime_args"].get("history_action_step_increment")
            playback = int(runtime.get("action_playback_target_steps") or 0)
            horizon = int(runtime.get("action_horizon") or 0)
            use_logical = bool(getattr(policy, "use_logical_step_history", False))
            existing = int(getattr(policy, "history_action_step_increment", 0) or 0)
            if history_step is not None and int(history_step) > 0:
                hop = int(history_step)
            elif playback > 0:
                hop = playback
            elif use_logical and existing > 0:
                hop = existing
            else:
                hop = horizon
            if hop > 0:
                policy.sync_history_action_step_increment(hop)
        elif hasattr(policy, "history_action_step_increment"):
            history_step = runtime["runtime_args"].get("history_action_step_increment")
            policy.history_action_step_increment = int(
                history_step
                or runtime["action_playback_target_steps"]
                or runtime["action_horizon"]
            )
        return runner

    def _apply_debug_context(
        self, runner, runtime: dict, task_name: str | None
    ) -> None:
        if not task_name:
            return
        key = task_name
        reset_counters = False
        if key not in self.task_log_dirs:
            task_runtime = dict(runtime)
            task_runtime["task_name"] = task_name
            task_runtime["run_id"] = self.target_run_id
            self.task_log_dirs[key] = build_log_dir(self.log_base_dir, task_runtime)
            os.makedirs(self.task_log_dirs[key], exist_ok=True)
            logger.add(
                os.path.join(self.task_log_dirs[key], "runtime.log"),
                level="DEBUG",
                enqueue=False,
            )
            logger.info(f"Task debug log dir: {self.task_log_dirs[key]}")
            reset_counters = True
        log_dir = self.task_log_dirs[key]
        if hasattr(runner, "set_debug_context"):
            runner.set_debug_context(log_dir=log_dir, reset_counters=reset_counters)

    def _runner_for_task(self, task_name: str | None, prompt: str | None):
        runtime = self._runtime_for_task(task_name, prompt)
        return self._apply_runtime(runtime, prompt, task_name)

    def prepare_task(self, task_name: str = None, prompt: str = None) -> None:
        self._runner_for_task(task_name, prompt)

    def get_task_runtime(self, task_name: str = None) -> dict:
        runtime = self._runtime_for_task(task_name, None)
        return {
            "image_type": runtime["image_type"],
            "action_type": runtime["action_type"],
            "duration": runtime["duration"],
            "resize_name": runtime["resize_name"],
        }

    def reset_policy(self):
        if self.default_runner is not None:
            self.default_runner.reset_policy()

    def infer(self, state, prompt: str = None, task_name: str = None):
        return self._runner_for_task(task_name, prompt).infer(
            state,
            prompt=prompt,
            task_name=task_name,
        )


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    task_name = cfg.get("task_name")
    if task_name:
        metadata = get_task_metadata(task_name)
        prompt = metadata["prompt"]
        robot_type = metadata["robot_type"]
        run_label = task_name
    else:
        robot_type = cfg.get("robot_type")
        if not robot_type:
            raise ValueError("Either task_name or robot_type is required")
        robot_type = str(robot_type).lower()
        prompt = ""
        run_label = f"{robot_type}_generalist"

    # Setup log directory (from config or default)
    log_base_dir = cfg.get("log_dir", "./logs")
    if task_name:
        runtime = build_effective_runtime(cfg, robot_type, task_name, prompt, run_label)
        runtime["run_id"] = str(cfg.get("run_id", "") or "").strip()
        runner = create_runner_from_runtime(runtime, cfg.image_size, log_base_dir)
        image_type = runtime["image_type"]
        action_type = runtime["action_type"]
        duration = runtime["duration"]
        resize_name = runtime["resize_name"]
    else:
        runner = TaskAwareRunner(cfg, robot_type, cfg.image_size, log_base_dir)
        default_runtime = runner.default_runtime
        image_type = default_runtime["image_type"]
        action_type = default_runtime["action_type"]
        duration = default_runtime["duration"]
        resize_name = default_runtime["resize_name"]

    # Online mode
    logger.info("Starting Online mode...")
    user_id = cfg.get("user_id", "")
    submission_id = cfg.get("submission_id", cfg.get("job_collection_id", ""))
    run_id = cfg.get("run_id", "")
    if not user_id:
        raise ValueError("user_id is required for online mode")
    if not submission_id:
        raise ValueError("submission_id is required for online mode")
    if run_id:
        logger.info(f"Target run_id: {run_id}")
    client = InterfaceClient(user_id)
    job_loop(
        client,
        runner,
        submission_id,
        cfg.image_size,
        image_type,
        action_type,
        duration,
        resize_name=resize_name,
        target_robot_type=robot_type,
        target_task_name=task_name,
        target_run_id=run_id,
    )


if __name__ == "__main__":
    main()
