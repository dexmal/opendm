"""Demo dataset registration."""

from opendm.constants.robot import RobotStateDesc
from opendm.dataset.register import register_dataset

ALOHA_STATE_DESC = (
    [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
)

register_dataset(
    {
        "demo": {
            "jsonl_dir": "./assets/demo/",
            "image_dir": "./assets/demo/",
            "image_keys": ["images_1", "images_2", "images_3"],
            "state_desc": ALOHA_STATE_DESC,
        },
    },
)
