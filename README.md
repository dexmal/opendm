# OpenDM

![DM0.5](docs/image/header.png)

<p align="center">
  <a href="https://www.dexmal.com/blog/dm0.5/index_en.html"><img src="https://img.shields.io/badge/📖-Tech_Blog-blue" alt="Tech Blog"></a>
  <a href="https://huggingface.co/collections/Dexmal/dm05"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow" alt="Hugging Face"></a>
  <a href="https://www.modelscope.cn/collections/Dexmal/DM05"><img src="https://img.shields.io/badge/%F0%9F%A4%96-ModelScope-624AFF" alt="ModelScope"></a>
  <a href="https://maas.dexmal.com/"><img src="https://img.shields.io/badge/MaaS-Online-brightgreen.svg" alt="MaaS"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  English | <a href="README.zh-CN.md">简体中文</a>
</p>

## Introduction

DM0.5 is Dexmal's next-generation Vision-Language-Action model (VLA) for open-world robot control. It builds on the native embodied modeling approach introduced by DM0, with systematic upgrades for open-ended instructions, long-horizon tasks, dynamic disturbances, and multi-embodiment robot control.

OpenDM provides DM0.5 model weights, training and inference scripts, dataset registration examples, and evaluation workflows for researchers and developers to train, fine-tune, evaluate, and deploy the model.

## News

- [2026-07-24] DM0.5 has added the SO101 pick cube fine-tuned checkpoint and the LoRA SFT workflow. See the [DM05 SO101 LoRA Training Guide](docs/en/dm05_so101_lora_training.md).
- [2026-07-17] DM0.5 has open-sourced the RoboTwin2.0 generalist model checkpoint, along with the supervised fine-tuning (SFT) code built upon the DM0.5 pretrained model. See the [DM05 RoboTwin2.0 Training and Evaluation Guide](docs/en/dm05_robotwin2.md).
- [2026-07-09] DM0.5 is officially released. Read the [technical blog](https://www.dexmal.com/blog/dm0.5/index_en.html) for more details.


## Models

| Model | Description | Checkpoint |
| --- | --- | --- |
| DM05 | Base DM0.5 model for fine-tuning | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05) |
| DM05-libero | LIBERO fine-tuned DM0.5 model for evaluation | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-libero) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-libero) |
| DM05-robotwin2 | RoboTwin2.0 fine-tuned DM0.5 model for evaluation | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-robotwin2) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-robotwin2) |
| DM05-SO101-Pick-Cube | SO101 fine-tuned DM0.5 model for evaluation | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-SO101-Pick-Cube) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-SO101-Pick-Cube) |

Example checkpoint download:

```bash
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

## Benchmark Results

### LIBERO Results

| Method | Spatial | Object | Goal | Long | Average | Reference |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DM0.5 | 99.0 | 99.8 | 99.6 | 97.4 | 99.0 | [Training & Evaluation](docs/en/dm05_libero.md) |


### RoboTwin 2.0 Results

| Method | Clean | Randomized | Average | Reference |
| --- | ---: | ---: | ---: | --- |
| DM0.5 | 93.6 | 93.3 | 93.5 | [Training & Evaluation](docs/en/dm05_robotwin2.md) |


## Quick Start

We recommend using Docker to set up the runtime environment first, which helps avoid version mismatches across CUDA, PyTorch, flash-attn, and other dependencies on the host machine.

### Requirements

```text
System requirements:
Ubuntu 20.04 / 22.04
NVIDIA GPU
NVIDIA Driver
Docker
NVIDIA Container Toolkit
Conda (optional, only required for local pip installation)

Recommended GPUs:
RTX 4090, A100, H100, H20
8 GPUs are recommended for training, and 1 GPU is sufficient for deployment inference.
```

### Docker Installation

```bash
git clone https://github.com/dexmal/opendm.git
cd opendm

docker run -it --rm --gpus all --network host \
  --name opendm \
  --shm-size=16g \
  -v "$PWD":/app/opendm \
  -w /app/opendm \
  dexmal/opendm:latest /bin/bash

# Run from the OpenDM repository root inside the container.
conda activate opendm
pip install -e .
```

### Local Installation

```bash
conda create -n opendm python=3.10 -y
conda activate opendm

pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

pip install ninja packaging
MAX_JOBS=2 pip install flash-attn --no-build-isolation

# Enter the OpenDM repository root.
cd opendm
pip install -e .
```

## Inference

After installing the environment and initializing the source code, you can start the model inference service. The service loads the specified checkpoint and exposes an HTTP endpoint for benchmark clients or other applications to request action predictions. Use a checkpoint that contains `norm_stats.json`, or make sure the matching stats already exist under `./norm_stats/`.

```bash
script/dm05_launcher.sh \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.port 7891
```

Arguments:

- `--task`: task type. Use `inference` for inference.
- `--nproc_per_node`: number of GPUs on a single node. 1 GPU is sufficient for inference.
- `--model-config.model-name-or-path`: model checkpoint path.
- `--model-config.chunk-size`: action chunk length.
- `--inference-config.port`: inference service port.

During inference, the service first looks for `norm_stats.json` in the checkpoint directory. If it is not found, it falls back to the matching file under `./norm_stats/`, which is normally generated during training for the same dataset, action mode, and chunk size.

After the service starts, send a test request to verify that the endpoint returns a valid response:

```bash
bash tests/curl_demo.sh http://SERVER_IP:7891/process_frame
```

`/process_frame` accepts a `multipart/form-data` request:

- `text`: task instruction.
- `states`: JSON array of the current robot state. The dimension and order must match the model's training and normalization statistics.
- `image`: image files, one field per configured image key. The order must match `--inference-config.image-keys`.
- `robot_type`: optional built-in robot type. Currently only `DOS W1` is supported. It provides the robot state description when relative actions need to be converted back to absolute actions.
- `control_mode` and `speed`: text conditioning fields required when directly serving the pretrained `Dexmal/DM05` model. They are normally not required for SFT checkpoints unless your SFT data was trained with the same fields.

A successful response has the following shape.

```text
{
  "response": [
    [0.012, -0.034, 0.18, "..."],
    [0.015, -0.031, 0.17, "..."],
    ...
  ]
}
```

## Training

### Data Preparation

Prepare data files and dataset configuration according to the dexbotic [Data Guide](https://github.com/dexmal/dexbotic/blob/main/docs/Data.md). Make sure `--data-config.dataset-name` in the training command matches the registered dataset name.

The training script selects a dataset through `--data-config.dataset-name`. Before training, register your dataset in the project dataset registry. We recommend using an existing file such as `opendm/dataset/demo.py` as a reference, then creating a new dataset config file such as `opendm/dataset/my_robot.py` and updating the dataset name, data paths, image keys, and state description.

```python
# opendm/dataset/my_robot.py

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
            "state_desc": MY_ROBOT_STATE_DESC,
        },
    }
)
```

Field descriptions:

- `my_robot`: dataset name registered in the dataset registry. Use it with `--data-config.dataset-name my_robot`.
- `jsonl_dir`: directory containing training `jsonl` files.
- `image_dir`: directory containing image files.
- `image_keys`: image field names to load from the dataset.
- `state_desc`: semantic description of each state/action dimension, such as robot joints and grippers.

During training, if the corresponding normalization statistics file does not exist, the script automatically computes it from the current dataset, action mode, and chunk size, then saves it under `./norm_stats/`.

### Start Training

After environment setup, source initialization, and data preparation, start model training. The training script reads the specified dataset configuration, loads the base checkpoint, and starts training according to the configuration.

```bash
script/dm05_launcher.sh \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50
```

Arguments:

- `--task train`: run in training mode.
- `--nproc_per_node 8`: number of training processes on a single node, usually matching the number of GPUs.
- `--data-config.dataset-name my_robot`: dataset name for training. It must match the project dataset configuration.
- `--model-config.model-name-or-path ./checkpoints/DM05`: initial model checkpoint path.
- `--model-config.chunk-size 50`: action chunk length predicted by the model.

Training logs will include data loading, model initialization, loss values, and checkpoint saving. Before running a full training job, verify that the data path, model checkpoint path, and GPU count are correctly configured.

## DM05 SFT with Demo and Custom Data

Start by running a complete DM05 SFT workflow with the built-in demo data and `playground/dm05_sft_demo.py`. After you are familiar with the data format, normalization statistics, training, inference, and service validation flow, replace the demo dataset with your own robot data for SFT. See [DM05 SFT and Validation Guide](docs/en/dm05_finetuning.md).

## Benchmark Fine-Tuning Reference

Use the benchmark fine-tuning guides as end-to-end references for fine-tuning DM05. They cover data and model preparation, SFT training, inference service startup, and benchmark evaluation.

- LIBERO: [DM05 LIBERO Training and Evaluation Guide](docs/en/dm05_libero.md)
- RoboTwin2.0: [DM05 RoboTwin2.0 Training and Evaluation Guide](docs/en/dm05_robotwin2.md)
- SO101: [DM05 SO101 LoRA Training Guide](docs/en/dm05_so101_lora_training.md)

## Guides

- Download models: see [Models](#models) or visit [Dexmal Hugging Face](https://huggingface.co/Dexmal).
- Prepare data: see the [Data Guide](https://github.com/dexmal/dexbotic/blob/main/docs/Data.md).
- Start inference service: see [Inference](#inference).
- DM05 SFT with demo or custom data: see [DM05 SFT and Validation Guide](docs/en/dm05_finetuning.md).
- Benchmark training and evaluation: see the [DM05 LIBERO Training and Evaluation Guide](docs/en/dm05_libero.md) and [DM05 RoboTwin2.0 Training and Evaluation Guide](docs/en/dm05_robotwin2.md); for LoRA SFT, see [DM05 LIBERO LoRA Training](docs/en/dm05_libero_lora_training.md) and [DM05 SO101 LoRA Training Guide](docs/en/dm05_so101_lora_training.md).

## Community and Support

- Learn more about Dexmal products and model updates on the [Dexmal website](https://www.dexmal.com/).
- Get DM model weights from [Dexmal Hugging Face](https://huggingface.co/Dexmal).
- If you encounter issues, please report them through [GitHub Issues](https://github.com/dexmal/opendm/issues).
- For further discussion, scan the [WeChat QR code](docs/image/wechat.jpeg) to contact us.

We will continue to release more model weights, technical documentation, and examples. If this project is helpful to you, please consider giving us a star on GitHub [![GitHub](https://img.shields.io/github/stars/dexmal/opendm?color=5B5BD6)](https://github.com/dexmal/opendm). Your support helps us move forward.

## License

This project is licensed under the [Apache-2.0](LICENSE).
