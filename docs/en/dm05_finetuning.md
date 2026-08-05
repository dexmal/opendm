# DM05 SFT and Validation Guide

This guide shows how to run a complete DM05 SFT workflow in OpenDM. Start with the built-in `assets/demo` data and `playground/dm05_sft_demo.py`, then replace the dataset configuration with your own robot data when the pipeline is verified.

## 1. Prepare the Base Model

Run from the OpenDM repository root:

```bash
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

## 2. Understand the Demo SFT Entry

OpenDM provides `playground/dm05_sft_demo.py` as a ready-to-run SFT entry. It is launched through `script/dm05_launcher.sh` and presets the demo training setup:

- `dataset_name`: `demo`
- `image_keys`: `images_1`, `images_2`, `images_3`
- `output_action_dim`: `14`
- `base_lr`: `2.5e-5`
- `per_device_train_batch_size`: `8`
- `num_train_steps`: `50000`
- `chunk_size`: inherited from the DM05 default, `50`

The built-in demo dataset is registered in `opendm/dataset/demo.py` and uses:

```text
assets/demo/
├── episode0.jsonl
├── index_cache.json
└── images/episode0/...
```

## 3. Check the Data Format

OpenDM reads robot demonstrations from JSONL files. Each line is one frame:

```json
{
  "images_1": {"type": "image", "url": "./images/episode0/cam_high/0.jpg"},
  "images_2": {"type": "image", "url": "./images/episode0/cam_left_wrist/0.jpg"},
  "images_3": {"type": "image", "url": "./images/episode0/cam_right_wrist/0.jpg"},
  "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  "prompt": "Pick up the object",
  "is_robot": true
}
```

Field notes:

- `images_1`, `images_2`, and `images_3` must match the dataset `image_keys`.
- `url` is relative to the registered `image_dir`.
- `state` is the current robot state. Its dimension order must match `state_desc`.
- `action` is optional. If it exists, OpenDM builds the training target from `action`; otherwise it builds the target from future `state` values.
- `prompt` is the task instruction.
- `index_cache.json` is optional. If it is missing, OpenDM scans the JSONL files and generates it automatically.

The default `action_mode` is `relative`, so the action and state dimensions must match. Gripper dimensions remain absolute according to `state_desc`. Each episode needs at least two frames.

## 4. Run DM05 SFT on Demo Data

Use `script/dm05_launcher.sh` with `--exp playground/dm05_sft_demo.py`.

For a short smoke test:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --nproc_per_node 1 \
  --task train \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --data-config.compute-norm-stats-max-batches 1 \
  --trainer-config.num-train-steps 10 \
  --trainer-config.save-steps 10 \
  --trainer-config.output-dir ./user_checkpoints/dm05_sft_demo_smoke
```

This smoke run only shortens the training and save steps. It does not override `chunk_size`, so it keeps the DM05 default value `50`. Override `--model-config.chunk-size` only when your data and controller need a different action horizon, and keep the same value for training and inference.

For a normal SFT run, keep the demo entry defaults and adjust the GPU count/output directory as needed:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --nproc_per_node 8 \
  --task train \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.output-dir ./user_checkpoints/dm05_sft_demo
```

During training, OpenDM needs normalization statistics for `state` and `action`. If the matching file does not exist, OpenDM computes it from the current experiment data and saves it under `./norm_stats/`. Data with the same `robot_type` shares one profile, while different robots are stored under `norm_stats_by_robot` in the same file. The complete file is copied into saved checkpoints as `norm_stats.json`.

This is important because inference must use the same norm stats to normalize the input `state` and denormalize the predicted `action`.

## 5. Use Your Own Data

After the demo SFT workflow runs successfully, register your own dataset. For example, create `opendm/dataset/my_robot.py`:

```python
from opendm.constants.robot import RobotStateDesc
from opendm.dataset.register import register_dataset

MY_ROBOT_STATE_DESC = (
    [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
)

register_dataset(
    {
        "my_robot": {
            "jsonl_dir": "./assets/my_robot/",
            "image_dir": "./assets/my_robot/",
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "state_desc": MY_ROBOT_STATE_DESC,
        },
    }
)
```

Then launch the same SFT entry and override the dataset name:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --nproc_per_node 8 \
  --task train \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.output-dir ./user_checkpoints/dm05_my_robot
```

`playground/dm05_sft_demo.py` can be used as an SFT template, but custom robot data usually requires checking and adjusting:

- `dataset_name`: your registered dataset name.
- `image_keys`: image fields in the JSONL data.
- `image_prompts`: prompt labels zipped with loaded images in order.
- `state_desc`: dimension order of `state` and `action`.
- `output_action_dim`: last dimension of the predicted action.
- `chunk_size`: action horizon for your controller. If unsure, start with the default `50`.
- `base_lr`, batch size, and training steps: adjust based on dataset size and GPU memory.

Make sure `image_keys`, `state_desc`, action dimension, and `chunk_size` stay consistent between training and inference.

## 6. Inference

See the [DM05 Inference Guide](dm05_inference.md) for the custom SFT checkpoint command, service validation, HTTP API usage, and fast backend setup.

## Checklist

- Launch SFT with `script/dm05_launcher.sh --exp playground/dm05_sft_demo.py`.
- Dataset name is registered and passed through `--data-config.dataset-name`.
- JSONL image keys match `image_keys`.
- `state_desc` length matches the state/action dimension.
- Training and inference use the same `chunk_size`.
- Inference checkpoint contains `norm_stats.json`, or `./norm_stats/` contains the matching file for the same dataset and `chunk_size`.
