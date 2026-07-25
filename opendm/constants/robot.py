from enum import Enum


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
    ALOHA_ROBOTWIN2 = "Aloha RoboTwin2"
    SO101 = "SO101"


ROBOT_STATE_DESCS = {
    RobotType.DOS_W1: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.ALOHA_ROBOTWIN2: [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER],
    RobotType.SO101: [RobotStateDesc.JOINT] * 5 + [RobotStateDesc.GRIPPER],
}
