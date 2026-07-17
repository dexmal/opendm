"""LIBERO dataset registration."""

from opendm.constants.robot import ROBOT_STATE_DESCS, RobotType
from opendm.dataset.register import register_dataset

register_dataset(
    {
        "generalist": {
            "jsonl_dir": "./data/robotwin2.0",
            "image_dir": "./data/robotwin2.0/video",
            "image_keys": ["images_1", "images_2", "images_3"],
            "robot_type": RobotType.ALOHA_ROBOTWIN2,
            "state_desc": ROBOT_STATE_DESCS[RobotType.ALOHA_ROBOTWIN2],
        },
    },
    prefix="robotwin2",
)
