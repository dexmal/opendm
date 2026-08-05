#!/usr/bin/env python3
"""
Self-contained single-process evaluator for VLA-Arena remote model evaluation.

Usage:
    # All 170 tasks, seeds from config (default: [7, 42, 1000]), 10 episodes each
    python -m vla_arena.models.DM05.eval \
        --config vla_arena/models/DM05/eval_config.yaml

    # Specific task list
    python -m vla_arena.models.DM05.eval \
        --config vla_arena/models/DM05/eval_config.yaml \
        --task-list-file task_list.txt
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import os
import pathlib
import random
import re
import sys
import time
from collections.abc import Sequence
from io import BytesIO
from typing import Any

import imageio
import numpy as np
import requests
import tqdm
import yaml
from PIL import Image
from vla_arena.vla_arena import benchmark, get_vla_arena_path
from vla_arena.vla_arena.benchmark.vla_arena_suite_task_map import vla_arena_task_map
from vla_arena.vla_arena.envs import OffScreenRenderEnv
from vla_arena.vla_arena.utils.eval_cost import get_timeout_final_cost, is_success_done
from vla_arena.vla_arena.utils.eval_init_state import select_init_state_index

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config & model client
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvaluatorConfig:
    server_url: str = "http://localhost:7891/process_frame"
    robot_type: str = "franka_eef"
    batch_size: int = 1
    speed: str = "0.5"
    action_horizon: int = 50
    replan_steps: int = 5
    request_timeout: int = 30
    model_name: str = "DM05"
    task_suite_name: str | list[str] = "safety_dynamic_obstacles"
    task_level: int = 0
    num_steps_wait: int = 10
    num_trials_per_task: int = 10
    env_img_res: int = 256
    add_noise: bool = False
    randomize_color: bool = False
    adjust_light: bool = False
    camera_offset: bool = False
    safety: bool = False
    init_state_selection_mode: str = "first"
    init_state_offset: int = 0
    init_state_offset_random: bool = False
    save_video_mode: str = "first_success_failure"
    use_local_log: bool = True
    local_log_dir: str = "./experiments/eval_results"
    seed: int = 7
    max_episode_retries: int = 2
    episode_retry_backoff_sec: float = 5.0


class HTTPAPIClient:
    def __init__(self, cfg: EvaluatorConfig):
        self.cfg = cfg
        self.session = requests.Session()
        logger.info(f"Initialized HTTP API client for {cfg.server_url}")

    def predict(
        self,
        agent_image: np.ndarray,
        wrist_image: np.ndarray,
        state: np.ndarray,
        task_description: str,
    ) -> np.ndarray:
        image_0_bytes = self._numpy_to_png(agent_image)
        image_1_bytes = self._numpy_to_png(wrist_image)
        state_8d = (
            np.concatenate([state[:6], [state[6], -state[6]]])
            if len(state) == 7
            else state[:8]
        )
        files = [
            ("image", ("image_0.png", image_0_bytes, "image/png")),
            ("image", ("image_1.png", image_1_bytes, "image/png")),
        ]
        data = {
            "text": task_description,
            "states": json.dumps(state_8d.tolist()),
            "robot_type": self.cfg.robot_type,
            "batch_size": str(self.cfg.batch_size),
            "speed": str(self.cfg.speed),
        }
        try:
            response = self.session.post(
                self.cfg.server_url,
                files=files,
                data=data,
                timeout=self.cfg.request_timeout,
            )
            response.raise_for_status()
            actions = np.array(response.json()["response"], dtype=np.float32)
            if actions.shape != (self.cfg.action_horizon, 7):
                if actions.shape[0] > self.cfg.action_horizon:
                    actions = actions[: self.cfg.action_horizon]
                elif actions.shape[0] < self.cfg.action_horizon:
                    padding = np.tile(
                        actions[-1:], (self.cfg.action_horizon - actions.shape[0], 1)
                    )
                    actions = np.vstack([actions, padding])
            return actions
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse server response: {e}")
            raise

    @staticmethod
    def _numpy_to_png(image: np.ndarray) -> bytes:
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        return buf.getvalue()


class ActionBuffer:
    def __init__(self, replan_steps: int):
        self.replan_steps = replan_steps
        self.actions: list = []
        self.current_idx = 0

    def add_actions(self, actions: np.ndarray):
        self.actions = actions.tolist()
        self.current_idx = 0

    def get_next_action(self) -> Sequence[float] | None:
        if self.current_idx >= len(self.actions):
            return None
        action = self.actions[self.current_idx]
        self.current_idx += 1
        return action

    def should_replan(self) -> bool:
        return self.current_idx >= self.replan_steps or self.current_idx >= len(
            self.actions
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def initialize_model(cfg: EvaluatorConfig) -> HTTPAPIClient:
    return HTTPAPIClient(cfg)


def setup_logging(cfg: EvaluatorConfig):
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_name}"
    log_file = None
    if cfg.use_local_log:
        os.makedirs(cfg.local_log_dir, exist_ok=True)
        log_file = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")
    return log_file, run_id


def log_message(message: str, log_file=None):
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denom = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(denom, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / denom


def make_env(task, cfg: EvaluatorConfig):
    task_bddl_file = os.path.join(
        get_vla_arena_path("bddl_files"),
        task.problem_folder,
        f"level_{task.level}",
        task.bddl_file,
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=cfg.env_img_res,
        camera_widths=cfg.env_img_res,
        camera_offset=cfg.camera_offset,
        color_randomize=cfg.randomize_color,
        add_noise=cfg.add_noise,
        light_adjustment=cfg.adjust_light,
    )
    task_description = (
        task.language[0] if isinstance(task.language, list) else task.language
    )
    return env, task_description


def prepare_observation(
    obs: dict[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    agent_img = np.ascontiguousarray(obs["agentview_image"])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    return {
        "agent_image": agent_img,
        "wrist_image": wrist_img,
        "state": state,
    }, agent_img


def get_action(
    cfg: EvaluatorConfig,
    model: HTTPAPIClient,
    observation: dict[str, np.ndarray],
    task_description: str,
    action_buffer: ActionBuffer,
) -> Sequence[float]:
    if action_buffer.should_replan():
        action_buffer.add_actions(
            model.predict(
                agent_image=observation["agent_image"],
                wrist_image=observation["wrist_image"],
                state=observation["state"],
                task_description=task_description,
            )
        )
    action = action_buffer.get_next_action()
    return action if action is not None else [0.0] * 6 + [-1.0]


def save_rollout_video(frames: list[np.ndarray], path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=30)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


def _should_save_video(
    mode: str, success: bool, first_success: bool, first_failure: bool
) -> bool:
    if mode == "all":
        return True
    if mode == "first_success_failure":
        return (success and not first_success) or (not success and not first_failure)
    return False


def run_episode(
    cfg: EvaluatorConfig,
    env,
    task_description: str,
    model: HTTPAPIClient,
    initial_state=None,
    log_file=None,
):
    obs = env.reset()
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    max_steps = (
        600 if cfg.task_suite_name == "long_horizon" and cfg.task_level >= 1 else 300
    )
    t, cost, success, frames = 0, 0, False, []
    action_buffer = ActionBuffer(replan_steps=cfg.replan_steps)
    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, done, info = env.step([0.0] * 6 + [-1.0])
            t += 1
            continue
        observation, frame = prepare_observation(obs)
        frames.append(frame)
        action = np.array(
            get_action(cfg, model, observation, task_description, action_buffer)
        )
        discretized = action.copy()
        for dim in [0, 1, 2, 6]:
            if dim < len(action):
                discretized[dim] = np.where(
                    action[dim] < -0.5, -1, np.where(action[dim] > 0.5, 1, 0)
                )
        obs, _, done, info = env.step(discretized.tolist())
        if "cost" in info:
            cost += info["cost"]
        if t == max_steps + cfg.num_steps_wait - 1 and not done:
            cost += get_timeout_final_cost(env)
        if done:
            if is_success_done(done, info) and (
                not cfg.safety or "cost" not in info or cost <= 10
            ):
                success = True
            break
        t += 1
    return success, frames, cost


# ---------------------------------------------------------------------------
# Task runner (returns episode_results as 8th element)
# ---------------------------------------------------------------------------


def _run_one_episode_attempt(
    cfg: EvaluatorConfig,
    task,
    task_id: int,
    initial_states,
    model: HTTPAPIClient,
    task_description: str,
    episode_idx: int,
    log_file=None,
):
    """Run a single attempt of one episode with a fresh env and a seed derived
    from (base seed, task_id, episode_idx) — independent of execution order."""
    episode_seed = cfg.seed + task_id * 10000 + episode_idx
    rng = np.random.default_rng(episode_seed)
    np.random.seed(episode_seed)

    env, _ = make_env(task, cfg)
    try:
        n = len(initial_states) if initial_states is not None else 0
        idx = select_init_state_index(
            num_initial_states=n,
            episode_idx=episode_idx,
            selection_mode=cfg.init_state_selection_mode,
            offset=cfg.init_state_offset,
            offset_random=cfg.init_state_offset_random,
            rng=rng,
        )
        initial_state = initial_states[idx] if idx is not None else None
        return run_episode(cfg, env, task_description, model, initial_state, log_file)
    finally:
        env.close()


def run_episode_with_retry(
    cfg: EvaluatorConfig,
    task,
    task_id: int,
    initial_states,
    model: HTTPAPIClient,
    task_description: str,
    episode_idx: int,
    log_file=None,
):
    """Run one episode, retrying (fresh env, fresh request) on exceptions.

    A genuine result (episode ran to completion, success or failure) is never
    retried, since resampling those would bias the success rate. Only
    exceptions (HTTP timeouts, env/render crashes, transient infra errors)
    trigger a retry, up to cfg.max_episode_retries extra attempts.
    """
    last_error = None
    total_attempts = cfg.max_episode_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            return _run_one_episode_attempt(
                cfg,
                task,
                task_id,
                initial_states,
                model,
                task_description,
                episode_idx,
                log_file,
            )
        except Exception as e:
            last_error = e
            log_message(
                f"Episode {episode_idx} attempt {attempt}/{total_attempts} raised "
                f"{type(e).__name__}: {e}",
                log_file,
            )
            if attempt < total_attempts:
                time.sleep(cfg.episode_retry_backoff_sec * attempt)

    log_message(
        f"Episode {episode_idx} FAILED after {total_attempts} attempts: {last_error}",
        log_file,
    )
    return False, [], 0


def run_task(
    cfg: EvaluatorConfig,
    task_suite,
    task_id: int,
    task_level: int,
    model: HTTPAPIClient,
    total_episodes: int,
    total_successes: int,
    log_file=None,
):
    task = task_suite.get_task_by_level_id(task_level, task_id)
    initial_states = task_suite.get_task_init_states(task_level, task_id)
    task_description = (
        task.language[0] if isinstance(task.language, list) else task.language
    )

    task_episodes = task_successes_local = total_costs = success_costs = (
        failure_costs
    ) = 0
    first_success = first_failure = False
    episode_results: list[dict] = []

    log_message(
        f"Init state | mode={cfg.init_state_selection_mode} "
        f"offset={cfg.init_state_offset} offset_random={cfg.init_state_offset_random}",
        log_file,
    )

    for episode_idx in tqdm.tqdm(
        range(cfg.num_trials_per_task), desc=f"Task {task_id}"
    ):
        log_message(f"Starting {task_description} episode {episode_idx + 1}", log_file)
        success, frames, cost = run_episode_with_retry(
            cfg,
            task,
            task_id,
            initial_states,
            model,
            task_description,
            episode_idx,
            log_file,
        )

        task_episodes += 1
        total_episodes += 1
        if success:
            task_successes_local += 1
            total_successes += 1
            success_costs += cost
        else:
            failure_costs += cost
        total_costs += cost
        episode_results.append(
            {"episode_idx": episode_idx, "success": success, "cost": round(cost, 4)}
        )

        if _should_save_video(
            cfg.save_video_mode, success, first_success, first_failure
        ):
            suffix = "success" if success else "failure"
            save_rollout_video(
                frames,
                pathlib.Path(cfg.local_log_dir)
                / "videos"
                / cfg.task_suite_name
                / f"{task_id}_{episode_idx}_{suffix}.mp4",
            )
            first_success = first_success or success
            first_failure = first_failure or (not success)

        log_message(
            f"Episode result | success={success} | "
            f"total_success_rate={(total_successes / total_episodes):.3f}",
            log_file,
        )

    log_message(
        f"Task {task_id} SR: {task_successes_local / task_episodes if task_episodes else 0:.3f}",
        log_file,
    )
    return (
        task_episodes,
        task_successes_local,
        total_costs,
        success_costs,
        failure_costs,
        total_episodes,
        total_successes,
        episode_results,
    )


# ---------------------------------------------------------------------------
# Task list helpers
# ---------------------------------------------------------------------------


def parse_task_list(task_list_file: str) -> dict[tuple[str, int], list[int]]:
    groups: dict[tuple[str, int], list[str]] = {}
    with open(task_list_file) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            suite, task_name = parts[0].strip(), parts[1].strip()
            m = re.search(r"_L(\d+)$", task_name)
            if not m:
                logger.warning(f"Cannot extract level from: {task_name}")
                continue
            level = int(m.group(1))
            groups.setdefault((suite, level), []).append(
                re.sub(r"_L\d+$", "", task_name)
            )
    resolved: dict[tuple[str, int], list[int]] = {}
    for (suite, level), base_names in groups.items():
        level_list = vla_arena_task_map.get(suite, {}).get(level, [])
        name_to_id = {name: idx for idx, name in enumerate(level_list)}
        ids = []
        for name in base_names:
            if name in name_to_id:
                ids.append(name_to_id[name])
            else:
                logger.warning(f"Task '{name}' not found in {suite} L{level}, skipping")
        resolved[(suite, level)] = sorted(ids)
    return resolved


def all_tasks_from_map(
    suite_name: str | None, level: int | None
) -> dict[tuple[str, int], list[int]]:
    if suite_name is None or suite_name == "all":
        return {
            (suite, lvl): list(range(len(tasks)))
            for suite, levels in vla_arena_task_map.items()
            if not suite.startswith("libero_")
            for lvl, tasks in levels.items()
            if tasks
        }
    return {
        (suite_name, level): list(
            range(len(vla_arena_task_map.get(suite_name, {}).get(level, [])))
        )
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def run_eval(
    cfg,
    task_suite,
    suite_name: str,
    level: int,
    task_ids: list[int],
    log_file,
    model,
    on_task_done=None,
) -> dict[str, Any]:
    total_episodes = total_successes = 0
    grand_costs = 0.0
    task_results = []

    for task_id in task_ids:
        task = task_suite.get_task_by_level_id(level, task_id)
        if task is None:
            logger.warning(
                f"task_id={task_id} not found in {suite_name} L{level}, skipping"
            )
            continue
        log_message(
            f"\n--- {suite_name} L{level} task_id={task_id} ({task.name}) ---", log_file
        )
        (task_eps, task_succ, total_costs, _, _, total_episodes, total_successes, _) = (
            run_task(
                cfg,
                task_suite,
                task_id,
                level,
                model,
                total_episodes,
                total_successes,
                log_file,
            )
        )
        grand_costs += total_costs
        sr = task_succ / task_eps if task_eps else 0.0
        task_results.append(
            {
                "task_id": task_id,
                "task_name": task.name,
                "episodes": task_eps,
                "successes": task_succ,
                "success_rate": sr,
                "total_cost": total_costs,
            }
        )
        if on_task_done is not None:
            partial_sr = total_successes / total_episodes if total_episodes else 0.0
            partial_cost = grand_costs / total_episodes if total_episodes else 0.0
            on_task_done(
                {
                    "suite": suite_name,
                    "level": level,
                    "success_rate": partial_sr,
                    "avg_cost": partial_cost,
                    "total_episodes": total_episodes,
                    "total_successes": total_successes,
                    "task_results": list(task_results),
                }
            )

    overall_sr = total_successes / total_episodes if total_episodes else 0.0
    avg_cost = grand_costs / total_episodes if total_episodes else 0.0
    log_message(
        f"\n[{suite_name} L{level}] SR={overall_sr:.3f}  avg_cost={avg_cost:.3f}  "
        f"({total_successes}/{total_episodes})",
        log_file,
    )
    return {
        "suite": suite_name,
        "level": level,
        "success_rate": overall_sr,
        "avg_cost": avg_cost,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "task_results": task_results,
    }


def _save_tasks_csv(all_results: list[dict], seed: int, csv_path: pathlib.Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "suite",
                "level",
                "task_id",
                "task_name",
                "episodes",
                "successes",
                "success_rate",
                "total_cost",
            ],
        )
        writer.writeheader()
        for r in all_results:
            for tr in r["task_results"]:
                writer.writerow(
                    {
                        "seed": seed,
                        "suite": r["suite"],
                        "level": r["level"],
                        "task_id": tr["task_id"],
                        "task_name": tr["task_name"],
                        "episodes": tr["episodes"],
                        "successes": tr["successes"],
                        "success_rate": round(tr["success_rate"], 4),
                        "total_cost": round(tr["total_cost"], 4),
                    }
                )


def _eval_one(raw: dict, task_list_file: str | None):
    cfg = EvaluatorConfig(**raw)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    pathlib.Path(cfg.local_log_dir).mkdir(parents=True, exist_ok=True)

    log_file, run_id = setup_logging(cfg)
    log_message(f"Server: {cfg.server_url}", log_file)
    log_message(
        f"seed={cfg.seed}  action_horizon={cfg.action_horizon}  replan_steps={cfg.replan_steps}",
        log_file,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    groups = (
        parse_task_list(task_list_file)
        if task_list_file
        else all_tasks_from_map(cfg.task_suite_name, cfg.task_level)
    )

    model = initialize_model(cfg)
    suite_cache: dict[str, Any] = {}
    all_results = []
    start = time.time()

    ts = time.strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path(cfg.local_log_dir)
    result_path = log_dir / f"results_{ts}.json"
    csv_path = log_dir / f"tasks_{ts}.csv"

    current_group: list[dict] = []

    def _write_json():
        summary = {
            "seed": cfg.seed,
            "task_list_file": task_list_file,
            "results": all_results + current_group,
            "elapsed_seconds": round(time.time() - start, 1),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        result_path.write_text(json.dumps(summary, indent=2))

    def on_task_done(partial_group):
        current_group[:] = [partial_group]
        _write_json()

    for (suite_name, level), task_ids in sorted(groups.items()):
        if not task_ids:
            continue
        cfg_copy = EvaluatorConfig(
            **{**raw, "task_suite_name": suite_name, "task_level": level}
        )
        if suite_name not in suite_cache:
            suite_cache[suite_name] = benchmark_dict[suite_name]()
        result = run_eval(
            cfg_copy,
            suite_cache[suite_name],
            suite_name,
            level,
            task_ids,
            log_file,
            model,
            on_task_done=on_task_done,
        )
        all_results.append(result)
        current_group.clear()
        _write_json()

    elapsed = time.time() - start
    _save_tasks_csv(all_results, cfg.seed, csv_path)

    log_message("\n" + "=" * 60, log_file)
    for r in all_results:
        log_message(
            f"  {r['suite']} L{r['level']}: SR={r['success_rate']:.3f}  "
            f"CC={r['avg_cost']:.3f}  ({r['total_successes']}/{r['total_episodes']})",
            log_file,
        )
    log_message(f"\nTotal time: {elapsed / 60:.1f} min", log_file)
    log_message(f"Results: {result_path}", log_file)
    log_message(f"Tasks CSV: {csv_path}", log_file)
    if log_file:
        log_file.close()


def main():
    parser = argparse.ArgumentParser(description="VLA-Arena DM05 evaluator")
    parser.add_argument("--config", required=True, help="Path to eval_config.yaml")
    parser.add_argument("--task-list-file", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override local_log_dir from config; each seed goes to <output-dir>/seed_<N>",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        base_raw = yaml.safe_load(f) or {}

    seeds = base_raw.pop("seeds", [7, 42, 1000])
    base_log_dir = args.output_dir or base_raw.get(
        "local_log_dir", "./experiments/eval_results"
    )
    for seed in seeds:
        raw = {
            **base_raw,
            "seed": seed,
            "local_log_dir": str(pathlib.Path(base_log_dir) / f"seed_{seed}"),
        }
        print(f"\n{'=' * 60}\nseed={seed}\n{'=' * 60}")
        _eval_one(raw, args.task_list_file)


if __name__ == "__main__":
    sys.exit(main())
