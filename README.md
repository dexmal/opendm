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

- [2026-08-03] Published the [physical robot modification guide](docs/en/robot_platforms.md) for AgileX COBOT Magic and DOS-W1, documenting camera changes and the robot-name mapping used by the algorithm.
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
| DM05-VLA-Arena | VLA-Arena fine-tuned DM0.5 model for evaluation | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-Vla-Arena) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-Vla-Arena) |
| DM05-Table30v2 | RoboChallenge Table 30 v2 DM0.5 model collection for evaluation | [🤗 Hugging Face](https://huggingface.co/collections/Dexmal/dm05-table30v2) / [🤖 ModelScope](https://www.modelscope.cn/collections/Dexmal/DM05-Table30v2) |

Example checkpoint download:

```bash
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

## Benchmark Results

<table>
  <thead>
    <tr>
      <th></th>
      <th>Benchmark</th>
      <th>Metric</th>
      <th>DM0.5</th>
      <th>Pi0</th>
      <th>Pi0.5</th>
      <th>GROOT-N1.7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6"><strong>Simulated Tasks</strong></td>
      <td><strong><a href="docs/en/dm05_libero.md">LIBERO</a></strong></td>
      <td><strong>SR</strong></td>
      <td align="right"><strong>99.0%</strong></td>
      <td align="right">94.4%</td>
      <td align="right">96.9%</td>
      <td align="right">97.0%</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="docs/en/dm05_robotwin2.md">RoboTwin2.0</a></strong></td>
      <td><strong>Clean</strong></td>
      <td align="right"><strong>93.6%</strong></td>
      <td align="right">65.9%</td>
      <td align="right">82.7%</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td><strong>Rand</strong></td>
      <td align="right"><strong>93.3%</strong></td>
      <td align="right">58.4%</td>
      <td align="right">76.8%</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td rowspan="3"><strong><a href="docs/en/dm05_vla_arena.md">VLA-Arena</a></strong></td>
      <td><strong>L0</strong></td>
      <td align="right"><strong>89.0%</strong></td>
      <td align="right">82.3%</td>
      <td align="right">64.3%</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td><strong>L1</strong></td>
      <td align="right"><strong>53.6%</strong></td>
      <td align="right">32.2%</td>
      <td align="right">35.6%</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td><strong>L2</strong></td>
      <td align="right"><strong>44.1%</strong></td>
      <td align="right">11.4%</td>
      <td align="right">24.5%</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Real-World Tasks</strong></td>
      <td rowspan="2"><strong><a href="docs/en/dm05_robochallenge.md">RoboChallenge<br>Table30V2</a></strong></td>
      <td><strong>Score</strong></td>
      <td align="right"><strong>54.42</strong></td>
      <td align="right">-</td>
      <td align="right">31.48</td>
      <td align="right">-</td>
    </tr>
    <tr>
      <td><strong>SR</strong></td>
      <td align="right"><strong>43.0%</strong></td>
      <td align="right">-</td>
      <td align="right">14.3%</td>
      <td align="right">-</td>
    </tr>
  </tbody>
</table>

Click a benchmark name to view the corresponding DM05 training and evaluation guide.


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

The base environment below covers training and the default inference backend.
The fast backend additionally requires TensorRT Python/runtime, Triton, and
PyTorch FlexAttention support.

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

The commands above create the base OpenDM environment. Before using
`--inference-config.backend fast`, continue with the fast-backend environment
layer below.

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

### Fast Backend Environment Layer

The Docker and local installation steps above are not enough for
`--inference-config.backend fast`. Activate the same `opendm` environment and
install the fast inference dependency layer:

```bash
pip install -e ".[fast-infer]"
```

The `fast-infer` extra installs `onnx`, `triton==3.6.0`, and `tensorrt`. Fast
startup is not a best-effort acceleration toggle: OpenDM builds or loads a
TensorRT vision engine, dispatches Triton prefix/suffix kernels, and forces the
LLM attention backend to `flex_attention`. TensorRT, Triton, and PyTorch
FlexAttention support are therefore required prerequisites for fast inference.

Before launching the fast backend, verify the active environment:

```bash
python -c "import tensorrt"
python -c "import triton"
python -c "import torch.nn.attention.flex_attention"
```

Use a PyTorch build that provides `torch.nn.attention.flex_attention` (for
example `torch>=2.5`). Also expect the first fast launch for each
checkpoint/image layout to spend extra time exporting ONNX and building the
TensorRT engine before the HTTP service becomes ready.

## Inference

After downloading the DM05 base pretrained checkpoint, start its default
inference service with:

```bash
script/dm05_launcher.sh \
  --exp opendm/exp/dm05_exp.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-prompts "Head" "Left wrist" "Right wrist" \
  --inference-config.port 7891
```

This example uses three images and a 14-dimensional state/action. See the
[DM05 Inference Guide](docs/en/dm05_inference.md) for robot profile selection,
HTTP request fields, fine-tuned checkpoint commands, fast backend setup,
runtime constraints, and troubleshooting.
Use `/v1/infer` for new integrations. The older `/process_frame` multipart API remains available as a legacy compatibility path and will be phased out over time.

## Training

### Data Preparation

Prepare data files and register the dataset according to the OpenDM [Data Guide](docs/en/data.md). Make sure `--data-config.dataset-name` in the training command matches the registered dataset name.

The training script selects a dataset through `--data-config.dataset-name`. Before training, register your dataset in the project dataset registry. We recommend using an existing file such as `opendm/dataset/demo.py` as a reference, then creating a new dataset config file such as `opendm/dataset/my_robot.py` and updating the dataset name, data paths, image keys, and state description.

```python
# opendm/dataset/my_robot.py

from opendm.constants.robot import RobotStateDesc, RobotType
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
            "robot_type": RobotType.ALOHA,
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
- `image_prompts`: prompt labels zipped with loaded images in order (e.g. Head / Left wrist).
- `robot_type`: robot embodiment used to select the state description and matching normalization profile.
- `state_desc`: semantic description of each state/action dimension, such as robot joints and grippers.

During training, if the corresponding normalization statistics file does not exist, the script automatically computes it from the current experiment data, action mode, and chunk size, then saves it under `./norm_stats/`. Data sources for the same robot type share one profile within an experiment; different robot types are stored separately in the same file.

### Start Training

After environment setup, source initialization, and data preparation, start model training. The training script reads the specified dataset configuration, loads the base checkpoint, and starts training according to the configuration.

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --trainer-config.num-train-steps 50000
```

Arguments:

- `--exp playground/dm05_sft_demo.py`: this example uses the DM05 SFT demo configuration as its training entry point. Copy and adapt this configuration when your dataset requires different settings.
- `--task train`: run in training mode.
- `--nproc_per_node 8`: number of training processes on a single node, usually matching the number of GPUs.
- `--data-config.dataset-name my_robot`: dataset name for training. It must match the project dataset configuration.
- `--model-config.model-name-or-path ./checkpoints/DM05`: initial model checkpoint path.
- `--model-config.chunk-size 50`: action chunk length predicted by the model.
- `--trainer-config.num-train-steps 50000`: total number of training steps.

#### Enable Weights & Biases Logging

W&B logging is optional and is enabled only when a project name is provided. OpenDM already includes the `wandb` dependency.

1. Authenticate on the training machine:

   ```bash
   wandb login
   ```

   For a non-interactive job, set `WANDB_API_KEY` instead. Do not commit the API key to the repository.

2. Add the following option to the existing training command:

   ```text
   --trainer-config.wandb-project <project-name>
   ```

   Replace `<project-name>` with the W&B project to use, for example `dm05-sft`. Remove this option to disable W&B logging.

Training logs will include data loading, model initialization, loss values, and checkpoint saving. Before running a full training job, verify that the data path, model checkpoint path, and GPU count are correctly configured.

## DM05 SFT with Demo and Custom Data

Start by running a complete DM05 SFT workflow with the built-in demo data and `playground/dm05_sft_demo.py`. After you are familiar with the data format, normalization statistics, training, inference, and service validation flow, replace the demo dataset with your own robot data for SFT. See [DM05 SFT and Validation Guide](docs/en/dm05_finetuning.md).

## Benchmark Fine-Tuning Reference

Use the benchmark fine-tuning guides as end-to-end references for data preparation, SFT training, and benchmark evaluation. Start the service with the [DM05 Inference Guide](docs/en/dm05_inference.md).

- LIBERO: [DM05 LIBERO Training and Evaluation Guide](docs/en/dm05_libero.md)
- RoboTwin2.0: [DM05 RoboTwin2.0 Training and Evaluation Guide](docs/en/dm05_robotwin2.md)
- VLA-Arena: [DM05 VLA-Arena Training and Evaluation Guide](docs/en/dm05_vla_arena.md)
- SO101: [DM05 SO101 LoRA Training Guide](docs/en/dm05_so101_lora_training.md)
- RoboChallenge Table 30 v2: [DM05 RoboChallenge Table 30 v2 Inference Guide](docs/en/dm05_robochallenge.md)

## Guides

- Download models: see [Models](#models) or visit [Dexmal Hugging Face](https://huggingface.co/Dexmal).
- Review physical robot changes: see the [AgileX COBOT Magic and DOS-W1 Modification Guide](docs/en/robot_platforms.md).
- Prepare data: see the [OpenDM Data Guide](docs/en/data.md).
- Start inference service: see the [DM05 Inference Guide](docs/en/dm05_inference.md).
- DM05 SFT with demo or custom data: see [DM05 SFT and Validation Guide](docs/en/dm05_finetuning.md).
- Benchmark training and evaluation: see the [DM05 LIBERO Training and Evaluation Guide](docs/en/dm05_libero.md), [DM05 RoboTwin2.0 Training and Evaluation Guide](docs/en/dm05_robotwin2.md), and [DM05 RoboChallenge Table 30 v2 Inference Guide](docs/en/dm05_robochallenge.md); for LoRA SFT, see [DM05 LIBERO LoRA Training](docs/en/dm05_libero_lora_training.md) and [DM05 SO101 LoRA Training Guide](docs/en/dm05_so101_lora_training.md).

## Community and Support

- Learn more about Dexmal products and model updates on the [Dexmal website](https://www.dexmal.com/).
- Get DM model weights from [Dexmal Hugging Face](https://huggingface.co/Dexmal).
- If you encounter issues, please report them through [GitHub Issues](https://github.com/dexmal/opendm/issues).
- For further discussion, scan the [WeChat QR code](docs/image/wechat.jpeg) to contact us.

We will continue to release more model weights, technical documentation, and examples. If this project is helpful to you, please consider giving us a star on GitHub [![GitHub](https://img.shields.io/github/stars/dexmal/opendm?color=5B5BD6)](https://github.com/dexmal/opendm). Your support helps us move forward.

## License

This project is licensed under the [Apache-2.0](LICENSE).
