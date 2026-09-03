"""Centralized constants definition for Table30 v2."""

# ========== Robot Configuration ==========
# robot_type -> image_type list
IMAGE_TYPE_MAP = {
    "arx5": ["cam_arm", "cam_global", "cam_side"],
    "aloha": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "ur5": ["cam_arm", "cam_global"],
    "w1": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
}

# robot_type -> image_key mapping (for model input).
# ARX5/UR5 slot order must match the successful dexbotic single-arm policy
# (dm05_single_arm_policy.IMAGE_MAPPING), NOT the platform image_type list order:
#   arx5: cam_global -> image_0, cam_side -> image_1, cam_arm -> image_2
#   ur5:  cam_global -> image_0, cam_arm  -> image_1
IMAGE_MAPPING = {
    "arx5": {
        "cam_global": "image_0",
        "cam_side": "image_1",
        "cam_arm": "image_2",
        "right_hand": "image_0",
        "high": "image_1",
        "left_hand": "image_2",
    },
    "aloha": {
        "cam_high": "image_0",
        "cam_left_wrist": "image_1",
        "cam_right_wrist": "image_2",
        "high": "image_0",
        "left_hand": "image_1",
        "right_hand": "image_2",
    },
    "ur5": {
        "cam_global": "image_0",
        "cam_arm": "image_1",
        "right_hand": "image_0",
        "left_hand": "image_1",
    },
    "w1": {
        "cam_high": "image_0",
        "cam_left_wrist": "image_1",
        "cam_right_wrist": "image_2",
        "high": "image_0",
        "left_hand": "image_1",
        "right_hand": "image_2",
    },
}


def get_robot_image_config(robot_type: str):
    robot_type = robot_type.lower()
    if robot_type not in IMAGE_TYPE_MAP:
        raise ValueError(
            f"Unknown robot: {robot_type}. Available: {list(IMAGE_TYPE_MAP.keys())}"
        )
    return IMAGE_TYPE_MAP[robot_type], IMAGE_MAPPING[robot_type]
