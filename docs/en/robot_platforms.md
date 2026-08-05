# DM0.5 Physical Robot Modifications

This document records only the changes made to the factory configurations of the physical robots used by OpenDM. Hardware, accessories, and mounting details not listed here remain unchanged from the corresponding factory setup.

中文版本：[DM0.5 真机机型改动说明](../zh/robot_platforms.md)

## AgileX COBOT Magic

Base platform: [AgileX COBOT Magic official page](https://global.agilex.ai/products/cobot-magic)

The corresponding robot type in the code is `Aloha`. Its `state_desc` is the same as `RobotType.ALOHA_ROBOTWIN2`: six left-arm joints, the left gripper, six right-arm joints, and the right gripper, for 14 dimensions in total.

The OpenDM unit differs from the factory configuration as follows:

1. The arm-mounted cameras on both follower arms are replaced with [RealSense D405](https://www.realsenseai.com/products/stereo-depth-camera-d405/) cameras.
2. Each D405 uses the project-specific long camera bracket. The 3D model is available at [realsense-d405-wrist-camera-bracket-long.step](../../assets/embodiments/agilex_cobot_magic/realsense-d405-wrist-camera-bracket-long.step).
3. The factory aluminum-extrusion main-view camera mount between the two follower arms is retained. Only the camera mounting height is changed to `177.2 cm` above the floor.
4. The angle between the main-view camera and the vertical aluminum extrusion is changed to `27.5°`.

![AgileX COBOT Magic main-view camera mounting schematic](../image/cobot-magic-camera-mount.svg)

The height and angle definitions match the annotations in the physical reference photo. The schematic does not specify the extrusion profile, bracket design, or installation procedure.

## Dexmal DOS-W1

Base platform: [Dexmal Mirror (DOS-W1) official page](https://dev.dexmal.com/product/dos-w1)

The corresponding robot type in the code is `DOS W1`.

The OpenDM unit retains the factory cameras and camera-mounting scheme. Only the main-view camera position is adjusted:

1. The main-view camera mounting height is changed to `177.2 cm` above the floor.
2. The angle between the camera and the vertical aluminum extrusion is changed to `27.5°`.

All other settings remain unchanged from the factory configuration.
