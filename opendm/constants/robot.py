from enum import Enum

# Soft tokens per history image (``<unused0>`` placeholders / pooled vision tokens).
HISTORY_TOKENS_PER_IMAGE = 16
# 2D adaptive-pool spatial size; ``HISTORY_POOL_SIZE ** 2 == HISTORY_TOKENS_PER_IMAGE``.
HISTORY_POOL_SIZE = int(HISTORY_TOKENS_PER_IMAGE**0.5)
assert HISTORY_POOL_SIZE * HISTORY_POOL_SIZE == HISTORY_TOKENS_PER_IMAGE


class ActionMode(Enum):
    ABSOLUTE = "absolute"
    RELATIVE = "relative"


class RobotStateDesc(Enum):
    JOINT = "joint"
    EEF = "eef"
    GRIPPER = "gripper"


class RobotType(Enum):
    DOS_W1 = "DOS W1"
    FRANKA = "Franka"
    ALOHA = "Aloha"
    ALOHA_ROBOTWIN2 = "Aloha RoboTwin2"
    SO101 = "SO101"
    ARX5 = "ARX5"
    UR5 = "UR5"


ROBOT_STATE_DESCS = {
    RobotType.DOS_W1: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA_ROBOTWIN2: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.SO101: [RobotStateDesc.JOINT] * 5 + [RobotStateDesc.GRIPPER],
    RobotType.ARX5: [RobotStateDesc.JOINT] * 6 + [RobotStateDesc.GRIPPER],
    RobotType.UR5: [RobotStateDesc.EEF] * 6 + [RobotStateDesc.GRIPPER],
}
