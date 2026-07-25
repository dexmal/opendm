# OpenDM

![DM0.5](docs/image/header-zh.png)

<p align="center">
  <a href="https://www.dexmal.com/blog/dm0.5/index.html"><img src="https://img.shields.io/badge/📖-Tech_Blog-blue" alt="Tech Blog"></a>
  <a href="https://huggingface.co/collections/Dexmal/dm05"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-yellow" alt="Hugging Face"></a>
  <a href="https://www.modelscope.cn/collections/Dexmal/DM05"><img src="https://img.shields.io/badge/%F0%9F%A4%96-ModelScope-624AFF" alt="ModelScope"></a>
  <a href="https://maas.dexmal.com/"><img src="https://img.shields.io/badge/MaaS-Online-brightgreen.svg" alt="MaaS"></a>
  <a href="#许可"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

<p align="center">
  <a href="README.md">English</a> | 简体中文
</p>

## 简介

DM0.5 是 Dexmal 面向开放世界机器人控制发布的新一代视觉-语言-动作模型（VLA）。它继承了 DM0 的原生具身建模路线，并进一步面向开放指令、长程任务、动态干扰和多机器人本体控制进行系统升级。

OpenDM 提供 DM0.5 的模型权重、训练与推理脚本、数据注册示例和评测流程，便于研究者和开发者进行持续训练、微调、评测和部署。

## 最新动态

- [2026-07-24] DM0.5 已新增 SO101 pick cube 微调 checkpoint 和 LoRA SFT 流程。参考 [DM05 SO101 LoRA 训练指南](docs/zh/dm05_so101_lora_training.md)。
- [2026-07-17] DM0.5 已开源 RoboTwin2.0 generalist 模型 checkpoint，以及基于 DM0.5 预训练模型的监督微调（SFT）代码。参考 [DM05 RoboTwin2.0 训练与评测指南](docs/zh/dm05_robotwin2.md)。
- [2026-07-09] DM0.5 正式发布。更多模型细节请阅读[技术博客](https://www.dexmal.com/blog/dm0.5/index.html)。

## 模型

| 模型 | 描述 | 权重地址 |
| --- | --- | --- |
| DM05 | 用于微调的 DM0.5 基础模型 | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05) |
| DM05-libero | 用于 LIBERO 评测的 DM0.5 微调模型 | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-libero) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-libero) |
| DM05-robotwin2 | 用于 RoboTwin2.0 评测的 DM0.5 微调模型 | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-robotwin2) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-robotwin2) |
| DM05-SO101-Pick-Cube | 用于 SO101 评测的 DM0.5 微调模型 | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-SO101-Pick-Cube) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-SO101-Pick-Cube) |

模型下载示例：

```bash
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

## Benchmark 结果

### LIBERO

| 方法 | Spatial | Object | Goal | Long | 平均 | 参考 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| DM0.5 | 99.0 | 99.8 | 99.6 | 97.4 | 99.0 | [训练与评测](docs/zh/dm05_libero.md) |

### RoboTwin 2.0

| 方法 | Clean | Randomized | 平均 | 参考 |
| --- | ---: | ---: | ---: | --- |
| DM0.5 | 93.6 | 93.3 | 93.5 | [训练与评测](docs/zh/dm05_robotwin2.md) |

## 快速开始

推荐优先使用 Docker 准备运行环境，避免宿主机 CUDA、PyTorch、flash-attn 等依赖版本不一致。

### 环境要求

```text
系统要求：
Ubuntu 20.04 / 22.04
NVIDIA GPU
NVIDIA Driver
Docker
NVIDIA Container Toolkit
Conda（可选）仅本地 pip 安装方式需要

推荐 GPU：
RTX 4090, A100, H100, H20
训练建议使用 8 卡，部署推理使用 1 卡即可
```

### Docker 安装

```bash
git clone https://github.com/dexmal/opendm.git
cd opendm

docker run -it --rm --gpus all --network host \
  --name opendm \
  --shm-size=16g \
  -v "$PWD":/app/opendm \
  -w /app/opendm \
  dexmal/opendm:latest /bin/bash

# 在容器内的 OpenDM 仓库根目录运行。
conda activate opendm
pip install -e .
```

### 本地安装

```bash
conda create -n opendm python=3.10 -y
conda activate opendm

pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

pip install ninja packaging
MAX_JOBS=2 pip install flash-attn --no-build-isolation

# 进入 OpenDM 仓库根目录。
cd opendm
pip install -e .
```

## 推理

完成环境安装和源码初始化后，可以启动模型推理服务。推理服务会加载指定 checkpoint，并对外提供 HTTP 接口，供评测基准或其他客户端请求动作预测。请使用包含 `norm_stats.json` 的 checkpoint，或确保 `./norm_stats/` 下已有匹配的统计文件。

```bash
script/dm05_launcher.sh \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --inference-config.port 7891
```

参数说明：

- `--task`：任务类型，推理时使用 `inference`。
- `--nproc_per_node`：单节点使用的 GPU 数量，推理使用 1 卡即可。
- `--model-config.model-name-or-path`：模型 checkpoint 路径。
- `--model-config.chunk-size`：动作块（action chunk）长度。
- `--inference-config.port`：推理服务端口。

推理时，服务会优先读取 checkpoint 目录下的 `norm_stats.json`。如果不存在，则回退到 `./norm_stats/` 下与当前数据集、action mode 和 chunk size 匹配的统计文件；该文件通常在训练阶段自动生成。

服务启动后，可以使用测试脚本发送一次请求，确认接口能够正常返回结果：

```bash
bash tests/curl_demo.sh http://SERVER_IP:7891/process_frame
```

`/process_frame` 接收 `multipart/form-data` 请求：

- `text`：任务指令。
- `states`：当前机器人状态的 JSON array，维度和顺序需要与模型训练和归一化统计保持一致。
- `image`：图像文件，每个配置的图像键对应一个 `image` 字段，顺序需要与 `--inference-config.image-keys` 一致。
- `robot_type`：可选的内置机器人类型，目前仅支持 `DOS W1`。当 relative action 需要转换回 absolute action 时，它用于提供机器人 state 描述。
- `control_mode` 和 `speed`：直接服务 pretrained `Dexmal/DM05` 模型时需要传入的文本条件字段。SFT checkpoint 通常不需要这两个字段，除非你的 SFT 数据训练时也使用了相同字段。

测试脚本正确返回形如以下的结果。

```text
{
  "response": [
    [0.012, -0.034, 0.18, "..."],
    [0.015, -0.031, 0.17, "..."],
    ...
  ]
}
```

## 训练

### 数据准备

按照 dexbotic [数据使用指南](https://github.com/dexmal/dexbotic/blob/main/docs/Data.md)准备数据文件和数据集配置，并确保训练命令中的 `--data-config.dataset-name` 与实际注册的数据集名称一致。

训练脚本通过 `--data-config.dataset-name` 指定数据集名称。启动训练前，需要先在项目数据注册表中注册对应数据集。建议参考已有的 `opendm/dataset/demo.py`，复制一份新的数据集配置文件，例如 `opendm/dataset/my_robot.py`，然后修改数据集名称、数据路径、图像字段和状态描述。

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

字段说明：

- `my_robot`：注册到数据集表中的数据集名称，训练时通过 `--data-config.dataset-name my_robot` 使用。
- `jsonl_dir`：训练数据的 `jsonl` 文件目录。
- `image_dir`：图像文件目录。
- `image_keys`：数据中需要读取的图像字段名。
- `state_desc`：状态 / 动作各维度对应的机器人关节、夹爪等含义。

训练启动时，如果对应的归一化参数文件不存在，脚本会根据当前数据集、action mode 和 chunk size 自动计算，并保存到 `./norm_stats/`。

### 启动训练

完成环境安装、源码初始化和数据准备后，可以启动模型训练。训练脚本会读取指定数据集配置，加载基础模型 checkpoint，并按照配置启动训练。

```bash
script/dm05_launcher.sh \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50
```

参数说明：

- `--task train`：指定当前任务为训练模式。
- `--nproc_per_node 8`：单机启动的训练进程数，通常对应使用的 GPU 数量。
- `--data-config.dataset-name my_robot`：指定训练数据集名称，需要与项目中的数据配置保持一致。
- `--model-config.model-name-or-path ./checkpoints/DM05`：指定初始模型 checkpoint 路径。
- `--model-config.chunk-size 50`：指定模型一次预测的动作块（action chunk）长度。

训练开始后，日志会输出数据加载、模型初始化、loss、checkpoint 保存等信息。实际训练前请确认数据路径、模型权重路径和 GPU 数量均已正确配置。

## DM05 SFT 与自定义数据微调

建议先使用内置 demo 数据和 `playground/dm05_sft_demo.py` 跑通一次完整的 DM05 SFT 流程，熟悉数据格式、归一化统计、训练、推理和服务验证后，再替换为自己的机器人数据进行 SFT。参考 [DM05 SFT 与验证指南](docs/zh/dm05_finetuning.md)。

## Benchmark 微调参考流程

如需端到端微调 DM05，可以参考 benchmark 微调指南，这些流程覆盖数据与模型准备、SFT 训练、推理服务启动和 benchmark 评测。

- LIBERO：[DM05 LIBERO 训练与评测指南](docs/zh/dm05_libero.md)
- RoboTwin2.0：[DM05 RoboTwin2.0 训练与评测指南](docs/zh/dm05_robotwin2.md)
- SO101：[DM05 SO101 LoRA 训练指南](docs/zh/dm05_so101_lora_training.md)

## 使用指南

- 下载模型：参考[模型](#模型)或访问 [Dexmal Hugging Face](https://huggingface.co/Dexmal)
- 准备数据：参考[数据使用指南](https://github.com/dexmal/dexbotic/blob/main/docs/Data.md)
- 启动推理服务：参考[推理](#推理)
- 使用 demo 或自有数据进行 DM05 SFT：参考[DM05 SFT 与验证指南](docs/zh/dm05_finetuning.md)
- Benchmark 训练和评测：参考[DM05 LIBERO 训练与评测指南](docs/zh/dm05_libero.md)和[DM05 RoboTwin2.0 训练与评测指南](docs/zh/dm05_robotwin2.md)；LoRA SFT 参考[DM05 LIBERO LoRA 训练](docs/zh/dm05_libero_lora_training.md)和[DM05 SO101 LoRA 训练指南](docs/zh/dm05_so101_lora_training.md)

## 社区与支持

- 了解更多 Dexmal 产品与模型动态，请访问 [Dexmal 官网](https://www.dexmal.com/)。
- 获取 DM 模型权重，请访问 [Dexmal Hugging Face](https://huggingface.co/Dexmal)。
- 如果你在使用中遇到问题，欢迎通过 [GitHub Issues](https://github.com/dexmal/opendm/issues) 反馈。
- 如需进一步沟通，也可以扫描[微信二维码](docs/image/wechat.jpeg)与我们联系。

我们将持续开放更多模型权重、技术文档和示例。如果这个项目对你有帮助，欢迎在 GitHub 上给我们一颗星 [![GitHub](https://img.shields.io/github/stars/dexmal/opendm?color=5B5BD6)](https://github.com/dexmal/opendm)，你的支持是我们前进的动力。

## 许可

本项目采用 [Apache-2.0 许可证](LICENSE)。
