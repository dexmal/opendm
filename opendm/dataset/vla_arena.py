"""Vla Arena dataset registration."""

from opendm.constants.robot import RobotStateDesc, RobotType
from opendm.dataset.register import register_dataset

VLA_ARENA_EEF_STATE_DESC = [RobotStateDesc.EEF] * 6 + [
    RobotStateDesc.GRIPPER
]  # 7D action

register_dataset(
    {
        "L0_L": {
            "jsonl_dir": "./data/vla_arena_L0_L/jsonl",
            "image_dir": "./data/vla_arena_L0_L/images",
            "image_keys": ["images_1", "images_2"],
            "image_prompts": ["Head", "Left wrist"],
            "robot_type": RobotType.FRANKA,
            "state_desc": VLA_ARENA_EEF_STATE_DESC,
        },
    },
    prefix="vla_arena_eef",
)
