"""Output postprocessing helpers used by OpenDM policies."""

import numpy as np


def wrap_angle_to_pi(angle: float) -> float:
    """Normalize radians to (-pi, pi]."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def ur5_roll_pi_anchor(ref_roll: float) -> float:
    """Pick +pi or -pi on the same side as ``ref_roll``.

    After wrapping ``ref_roll`` to (-pi, pi]: non-negative -> +pi, negative -> -pi.
    """
    wrapped = wrap_angle_to_pi(ref_roll)
    return float(np.pi if wrapped >= 0.0 else -np.pi)


def apply_w1_gripper_trick(actions: np.ndarray, task_name: str) -> np.ndarray:
    """W1 gripper postprocessing used by the RoboChallenge W1 deployment."""
    left_gripper_idx = 6
    right_gripper_idx = 13
    if actions.shape[1] <= right_gripper_idx:
        return actions

    actions = actions.copy()
    task_name = str(task_name or "")
    if task_name == "place_objects_into_desk_drawer":
        actions[:, left_gripper_idx] = np.where(
            actions[:, left_gripper_idx] < 0.015,
            0.0,
            actions[:, left_gripper_idx],
        )
        actions[:, right_gripper_idx] = np.where(
            actions[:, right_gripper_idx] < 0.015,
            0.0,
            actions[:, right_gripper_idx],
        )
    elif task_name == "sweep_the_trash":
        actions[:, left_gripper_idx] = np.where(
            actions[:, left_gripper_idx] < 0.02,
            0.0,
            actions[:, left_gripper_idx],
        )
        actions[:, right_gripper_idx] = np.where(
            actions[:, right_gripper_idx] < 0.032,
            0.0,
            actions[:, right_gripper_idx],
        )
    elif task_name == "put_the_shoes_back":
        actions[:, left_gripper_idx] = np.where(
            actions[:, left_gripper_idx] < 0.01,
            0.0,
            actions[:, left_gripper_idx],
        )
        actions[:, right_gripper_idx] = np.where(
            actions[:, right_gripper_idx] < 0.01,
            0.0,
            actions[:, right_gripper_idx],
        )
    elif task_name == "hold_the_tray_with_both_hands":
        actions[:, left_gripper_idx] = np.where(
            actions[:, left_gripper_idx] < 0.02,
            0.0,
            actions[:, left_gripper_idx],
        )
        actions[:, right_gripper_idx] = np.where(
            actions[:, right_gripper_idx] < 0.02,
            0.0,
            actions[:, right_gripper_idx],
        )
        actions[:, left_gripper_idx] = np.clip(
            actions[:, left_gripper_idx] - 0.003,
            0.0,
            None,
        )
        actions[:, right_gripper_idx] = np.clip(
            actions[:, right_gripper_idx] - 0.003,
            0.0,
            None,
        )
    else:
        actions[:, left_gripper_idx] = np.clip(
            actions[:, left_gripper_idx] - 0.003,
            0.0,
            None,
        )
        actions[:, right_gripper_idx] = np.clip(
            actions[:, right_gripper_idx] - 0.003,
            0.0,
            None,
        )
    return actions
