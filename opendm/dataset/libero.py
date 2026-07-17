"""LIBERO dataset registration."""

from opendm.constants.robot import RobotStateDesc, RobotType
from opendm.dataset.register import register_dataset

LIBERO_7D_STATE_DESC = [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER]
LIBERO_8D_STATE_DESC = [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER] * 2

register_dataset(
    {
        "goal": {
            "jsonl_dir": "./data/libero/libero_goal",
            "image_dir": "./data/libero/libero_goal/video",
            "image_keys": ["images_1", "images_2"],
            "robot_type": RobotType.FRANKA,
            "state_desc": LIBERO_7D_STATE_DESC,
        },
        "10": {
            "jsonl_dir": "./data/libero/libero_10",
            "image_dir": "./data/libero/libero_10/video",
            "image_keys": ["images_1", "images_2"],
            "robot_type": RobotType.FRANKA,
            "state_desc": LIBERO_7D_STATE_DESC,
        },
        "spatial": {
            "jsonl_dir": "./data/libero/libero_spatial",
            "image_dir": "./data/libero/libero_spatial/video",
            "image_keys": ["images_1", "images_2"],
            "robot_type": RobotType.FRANKA,
            "state_desc": LIBERO_7D_STATE_DESC,
        },
        "object": {
            "jsonl_dir": "./data/libero/libero_object",
            "image_dir": "./data/libero/libero_object/video",
            "image_keys": ["images_1", "images_2"],
            "robot_type": RobotType.FRANKA,
            "state_desc": LIBERO_7D_STATE_DESC,
        },
        "pi0_all": {
            "jsonl_dir": "./data/libero/libero_pi0_all",
            "image_dir": "./data/libero/libero_pi0_all/image",
            "image_keys": ["images_1", "images_2"],
            "robot_type": RobotType.FRANKA,
            "state_desc": LIBERO_8D_STATE_DESC,
        },
    },
    prefix="libero",
)
