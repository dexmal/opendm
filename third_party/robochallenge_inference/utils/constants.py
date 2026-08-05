"""Centralized constants definition for Table30 v2."""

import numpy as np

# ========== Task Metadata ==========
# task_name -> prompt, robot_type, task_id mapping
TASK_METADATA = {
    "arrange_flowers": {
        "prompt": "Put the 4 flowers into the vase.",
        "robot_type": "arx5",
        "task_id": "9d9d",
    },
    "arrange_fruits": {
        "prompt": "Arrange the fruit in the basket.",
        "robot_type": "ur5",
        "task_id": "bf87",
    },
    "fold_the_clothes": {
        "prompt": "Fold the T-shirts and stack them neatly in the upper-left corner of the table.",
        "robot_type": "w1",
        "task_id": "6dcb",
    },
    "hang_the_cup": {
        "prompt": "Hang the cup on the rack.",
        "robot_type": "arx5",
        "task_id": "3624",
    },
    "hold_the_tray_with_both_hands": {
        "prompt": "Place the ball from the desk onto the small tray, and then move the "
        "small tray onto the large tray.",
        "robot_type": "w1",
        "task_id": "2b13",
    },
    "item_classification": {
        "prompt": "Place the stationery in the yellow box and the electronics in the blue box.",
        "robot_type": "ur5",
        "task_id": "cb19",
    },
    "lint_roller_remove_dirt": {
        "prompt": "Use a lint roller to remove the debris from the clothing.",
        "robot_type": "aloha",
        "task_id": "f31b",
    },
    "pack_the_items": {
        "prompt": "Box up the tablet and its accessories.",
        "robot_type": "aloha",
        "task_id": "671c",
    },
    "pack_the_toothbrush_holder": {
        "prompt": "Put the toothbrush and toothpaste into the toiletries case in sequence, "
        "close the case, and then place it into the basket.",
        "robot_type": "aloha",
        "task_id": "6ccd",
    },
    "paint_jam": {
        "prompt": "Spread the bread with jam.",
        "robot_type": "aloha",
        "task_id": "4bca",
    },
    "pick_out_the_green_blocks": {
        "prompt": "Find all the green blocks and put them into the basket.",
        "robot_type": "arx5",
        "task_id": "2acf",
    },
    "place_objects_into_desk_drawer": {
        "prompt": "Open the drawer, put the bottle opener inside, and close the drawer.",
        "robot_type": "w1",
        "task_id": "6ab4",
    },
    "press_the_button": {
        "prompt": "Press the buttons in the following sequence: pink, blue, green, and then yellow.",
        "robot_type": "arx5",
        "task_id": "bec8",
    },
    "put_in_pen_container": {
        "prompt": "Put the pens on the desk into the pen holder.",
        "robot_type": "w1",
        "task_id": "2945",
    },
    "put_the_books_back": {
        "prompt": "Place the books back onto the bookshelf.",
        "robot_type": "aloha",
        "task_id": "ae1f",
    },
    "put_the_pencil_case_into_the_schoolbag": {
        "prompt": "Put the pencil case into the backpack.",
        "robot_type": "aloha",
        "task_id": "fea0",
    },
    "put_the_shoes_back": {
        "prompt": "Pair the two pairs of shoes on the desk and place them on the shoe rack",
        "robot_type": "w1",
        "task_id": "0965",
    },
    "scoop_with_a_small_spoon": {
        "prompt": "Scoop beans and place them into the empty bowl.",
        "robot_type": "aloha",
        "task_id": "e358",
    },
    "shred_paper": {
        "prompt": "Put the paper into the shredder.",
        "robot_type": "ur5",
        "task_id": "44df",
    },
    "stack_bowls": {
        "prompt": "Put the blue bowl into the beige bowl, and put the green bowl into the blue bowl.",
        "robot_type": "w1",
        "task_id": "8fc9",
    },
    "stamp_positioning": {
        "prompt": "Stamp the signature area on the paper.",
        "robot_type": "aloha",
        "task_id": "3809",
    },
    "sweep_the_trash": {
        "prompt": "Sweep the trash on the table into the dustpan.",
        "robot_type": "w1",
        "task_id": "b2e8",
    },
    "tidy_up_the_makeup_table": {
        "prompt": "Sort and organize the cosmetics on the table.",
        "robot_type": "w1",
        "task_id": "3232",
    },
    "tie_a_knot": {
        "prompt": "Tie a knot with the string on the table.",
        "robot_type": "w1",
        "task_id": "c6b3",
    },
    "turn_on_the_light_switch": {
        "prompt": "Turn on the lamp.",
        "robot_type": "arx5",
        "task_id": "d96c",
    },
    "untie_the_shoelaces": {
        "prompt": "Remove the laces from the shoes, then place them on the table.",
        "robot_type": "w1",
        "task_id": "1c7d",
    },
    "water_the_flowers": {
        "prompt": "Water the potted plants.",
        "robot_type": "arx5",
        "task_id": "f5b4",
    },
    "wipe_the_blackboard": {
        "prompt": "Wipe the blackboard clean.",
        "robot_type": "aloha",
        "task_id": "43ea",
    },
    "wipe_the_table": {
        "prompt": "Wipe the stains off the desk with a rag.",
        "robot_type": "arx5",
        "task_id": "90e1",
    },
    "wrap_with_a_soft_cloth": {
        "prompt": "Bundle the objects together using the cloth on the table.",
        "robot_type": "aloha",
        "task_id": "abc7",
    },
}

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


def get_task_metadata(task_name: str):
    if task_name not in TASK_METADATA:
        raise ValueError(
            f"Unknown task: {task_name}. Available: {list(TASK_METADATA.keys())}"
        )
    return TASK_METADATA[task_name]


def get_robot_image_config(robot_type: str):
    robot_type = robot_type.lower()
    if robot_type not in IMAGE_TYPE_MAP:
        raise ValueError(
            f"Unknown robot: {robot_type}. Available: {list(IMAGE_TYPE_MAP.keys())}"
        )
    return IMAGE_TYPE_MAP[robot_type], IMAGE_MAPPING[robot_type]


# ========== Joint Limits ==========
ANGLE2RAD = np.pi / 180.0
ALOHA_JOINT_MIN = [
    -150 * ANGLE2RAD,
    0 * ANGLE2RAD,
    -170 * ANGLE2RAD,
    -100 * ANGLE2RAD,
    -70 * ANGLE2RAD,
    -180 * ANGLE2RAD,
    0,
    -150 * ANGLE2RAD,
    0 * ANGLE2RAD,
    -170 * ANGLE2RAD,
    -99 * ANGLE2RAD,
    -70 * ANGLE2RAD,
    -180 * ANGLE2RAD,
    0,
]
ALOHA_JOINT_MAX = [
    150 * ANGLE2RAD,
    180 * ANGLE2RAD,
    0 * ANGLE2RAD,
    100 * ANGLE2RAD,
    70 * ANGLE2RAD,
    180 * ANGLE2RAD,
    0.1,
    150 * ANGLE2RAD,
    180 * ANGLE2RAD,
    0 * ANGLE2RAD,
    99 * ANGLE2RAD,
    70 * ANGLE2RAD,
    180 * ANGLE2RAD,
    0.1,
]
