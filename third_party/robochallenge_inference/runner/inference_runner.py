import datetime
import os
import pickle
import time
from typing import Any

import numpy as np
from loguru import logger
from utils.constants import IMAGE_MAPPING


class InferenceRunner:
    """Inference runner for the current OpenDM policy API."""

    def __init__(
        self,
        policy,
        robot_type: str,
        action_type: str,
        task_name: str,
        image_type: list[str],
        image_mapping: dict[str, str] = None,
        action_horizon: int = 15,
        debug: bool = False,
        debug_image_limit: int = 3,
        log_dir: str = None,
    ):
        self.policy = policy
        self.robot_type = robot_type.lower()
        self.action_type = action_type
        self.task_name = task_name
        self.image_type = image_type
        self.action_horizon = action_horizon
        self.debug = bool(debug)
        # debug_image_limit < 0 means unlimited platform image snapshots.
        self.debug_image_limit = int(debug_image_limit or 0)
        self.debug_image_count = 0
        self.debug_step_count = 0
        self.debug_step_dir = None
        self.platform_image_dir = None
        self.log_dir = log_dir or "./logs"
        if self.debug:
            self.debug_step_dir = os.path.join(self.log_dir, "debug_steps")
            os.makedirs(self.debug_step_dir, exist_ok=True)
        if self.debug and self.debug_image_limit != 0:
            self.platform_image_dir = os.path.join(self.log_dir, "platform_images")
            os.makedirs(self.platform_image_dir, exist_ok=True)

        # robot_type -> image_key mapping
        self.image_mapping = image_mapping or IMAGE_MAPPING[self.robot_type]

        logger.info(
            f"InferenceRunner initialized: robot={robot_type}, task={task_name}"
        )

    def set_debug_context(self, log_dir: str, reset_counters: bool = False) -> None:
        self.log_dir = log_dir or self.log_dir
        if reset_counters:
            self.debug_image_count = 0
            self.debug_step_count = 0
        self.debug_step_dir = None
        self.platform_image_dir = None
        if self.debug:
            self.debug_step_dir = os.path.join(self.log_dir, "debug_steps")
            os.makedirs(self.debug_step_dir, exist_ok=True)
        if self.debug and self.debug_image_limit != 0:
            self.platform_image_dir = os.path.join(self.log_dir, "platform_images")
            os.makedirs(self.platform_image_dir, exist_ok=True)

    def reset_policy(self):
        self.policy.reset()

    def set_task_context(self, task_name: str = None, prompt: str = None) -> None:
        if task_name:
            self.task_name = task_name
            if hasattr(self.policy, "task_name"):
                self.policy.task_name = task_name
        if prompt:
            self.policy.prompt = prompt

    def infer(
        self,
        state: dict[str, Any],
        prompt: str = None,
        task_name: str = None,
    ) -> list[list[float]]:
        """
        Run the policy on one platform state.

        Args:
            state: dictionary containing "images" and "action"

        Returns:
            List of action sequences
        """
        self.set_task_context(task_name=task_name, prompt=prompt)
        debug_record = self._new_debug_record(state, prompt=prompt, task_name=task_name)
        self._save_platform_images(state.get("images", {}))

        if not hasattr(self.policy, "run_policy"):
            raise TypeError(
                "InferenceRunner expects policy.run_policy(state, image_type). "
                f"Got policy={type(self.policy).__name__}"
            )

        inference_start = time.perf_counter()
        actions = self.policy.run_policy(state, self.image_type)
        inference_time = time.perf_counter() - inference_start
        model_latency_sec = self._get_model_latency_sec()
        model_latency_ms = (
            None if model_latency_sec is None else round(model_latency_sec * 1000.0, 3)
        )
        wrapper_overhead_ms = (
            None
            if model_latency_sec is None
            else round(max(inference_time - model_latency_sec, 0.0) * 1000.0, 3)
        )
        logger.info(
            "Inference time: {:.4f}s, model_latency_ms={}, wrapper_overhead_ms={}",
            inference_time,
            model_latency_ms,
            wrapper_overhead_ms,
        )
        logger.debug(f"Actions:\n{actions}")
        self._save_debug_record(
            debug_record,
            policy_api="run_policy",
            final_actions=np.asarray(actions),
            inference_time=inference_time,
            model_latency_sec=model_latency_sec,
            model_latency_ms=model_latency_ms,
            wrapper_overhead_ms=wrapper_overhead_ms,
        )

        return actions

    def _get_model_latency_sec(self) -> float | None:
        for obj in (self.policy, getattr(self.policy, "infer_cfg", None)):
            if obj is None:
                continue
            latency = getattr(obj, "last_model_latency_sec", None)
            if latency is None:
                continue
            try:
                return float(latency)
            except (TypeError, ValueError):
                return None
        return None

    def _new_debug_record(
        self,
        state: dict[str, Any],
        prompt: str = None,
        task_name: str = None,
    ) -> dict[str, Any] | None:
        if not self.debug:
            return None
        self.debug_step_count += 1
        images = state.get("images", {})
        return {
            "step": self.debug_step_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "robot_type": self.robot_type,
            "action_type": self.action_type,
            "task_name": task_name or self.task_name,
            "prompt": prompt or getattr(self.policy, "prompt", None),
            "image_type": list(self.image_type),
            "image_mapping": dict(self.image_mapping),
            "action_horizon": self.action_horizon,
            "platform_image_sources": list(images.keys()),
            "platform_image_bytes": dict(images),
            "raw_state": np.asarray(state.get("action")).copy(),
        }

    def _save_debug_record(self, record: dict[str, Any] | None, **updates) -> None:
        if not record or not self.debug_step_dir:
            return
        try:
            record.update(updates)
            step = int(record["step"])
            path = os.path.join(self.debug_step_dir, f"step_{step:04d}.pkl")
            with open(path, "wb") as f:
                pickle.dump(record, f)
            logger.info(f"Saved debug step record to {path}")
        except Exception as exc:
            logger.warning(f"Failed to save debug step record: {exc}")

    def _save_platform_images(self, images_dict: dict[str, bytes]) -> None:
        if not self.platform_image_dir:
            return
        if (
            self.debug_image_limit > 0
            and self.debug_image_count >= self.debug_image_limit
        ):
            return
        try:
            import cv2

            self.debug_image_count += 1
            for source, image_data in images_dict.items():
                image = cv2.imdecode(
                    np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                if image is None:
                    logger.warning(f"Failed to decode platform image source={source}")
                    continue
                safe_source = str(source).replace("/", "_")
                path = os.path.join(
                    self.platform_image_dir,
                    f"{self.debug_image_count:04d}_{safe_source}.jpg",
                )
                cv2.imwrite(path, image)
            logger.info(f"Saved platform images to {self.platform_image_dir}")
        except Exception as exc:
            logger.warning(f"Failed to save platform images: {exc}")
