from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .opendm_policy import OpenDMPolicy


def get_policy(ckpt_path: str, policy_type: str = None, **kwargs) -> "OpenDMPolicy":
    """Create the opendm-backed DM05 policy for the current robot type."""
    policy_type = (policy_type or "dm05").lower()
    if policy_type != "dm05":
        raise ValueError(f"Only policy_type='dm05' is supported, got {policy_type!r}")

    robot_type = str(kwargs.get("robot_type", "")).lower()
    if robot_type not in ("arx5", "ur5", "aloha", "w1", "dm05_dosw1"):
        raise ValueError(
            f"Cannot infer DM05 policy implementation for robot_type={robot_type!r}"
        )

    from .opendm_policy import OpenDMPolicy

    return OpenDMPolicy(ckpt_path=ckpt_path, **kwargs)
