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

- [2026-08-26] 发布 [DM05-MEM-Robodojo-Sim](https://huggingface.co/Dexmal/DM05-MEM-Robodojo-Sim)，面向 RoboDojo-Sim 中 ARX X5 双臂操作任务的微调模型，配套评测接入见 [XPolicyLab PR #101](https://github.com/XPolicyLab/XPolicyLab/pull/101)。
- [2026-08-03] 已发布 AgileX COBOT Magic 与 DOS-W1 的[真机机型改动说明](docs/zh/robot_platforms.md)，记录相机改动及机型名称映射。
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
| DM05-VLA-Arena | 用于 VLA-Arena 评测的 DM0.5 微调模型 | [训练与评测](docs/zh/dm05_vla_arena.md) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-Vla-Arena) |
| DM05-Table30v2 | 用于 RoboChallenge Table 30 v2 评测的 DM0.5 模型集合 | [🤗 Hugging Face](https://huggingface.co/collections/Dexmal/dm05-table30v2) / [🤖 ModelScope](https://www.modelscope.cn/collections/Dexmal/DM05-Table30v2) |
| DM05-MEM-Robodojo-Sim | 用于 RoboDojo-Sim 中 ARX X5 双臂操作任务的 DM0.5 微调模型 | [🤗 Hugging Face](https://huggingface.co/Dexmal/DM05-MEM-Robodojo-Sim) / [🤖 ModelScope](https://modelscope.cn/models/Dexmal/DM05-MEM-Robodojo-Sim) |

模型下载示例：

```bash
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

## Benchmark 结果

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
      <td rowspan="8"><strong>Simulated Tasks</strong></td>
      <td><strong><a href="docs/zh/dm05_libero.md">LIBERO</a></strong></td>
      <td><strong>SR</strong></td>
      <td align="right"><strong>99.0%</strong></td>
      <td align="right">94.4%</td>
      <td align="right">96.9%</td>
      <td align="right">97.0%</td>
    </tr>
    <tr>
      <td rowspan="2"><strong><a href="docs/zh/dm05_robotwin2.md">RoboTwin2.0</a></strong></td>
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
      <td rowspan="3"><strong><a href="docs/zh/dm05_vla_arena.md">VLA-Arena</a></strong></td>
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
      <td rowspan="2"><strong><a href="https://github.com/XPolicyLab/XPolicyLab/pull/101">RoboDojo-Sim</a></strong></td>
      <td><strong>Score</strong></td>
      <td align="right"><strong>24.90</strong></td>
      <td align="right">3.48</td>
      <td align="right">11.41</td>
      <td align="right">2.85</td>
    </tr>
    <tr>
      <td><strong>SR</strong></td>
      <td align="right"><strong>19.34%</strong></td>
      <td align="right">1.53%</td>
      <td align="right">6.91%</td>
      <td align="right">1.31%</td>
    </tr>
    <tr>
      <td rowspan="2"><strong>Real-World Tasks</strong></td>
      <td rowspan="2"><strong><a href="docs/zh/dm05_robochallenge.md">RoboChallenge<br>Table30V2</a></strong></td>
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

点击表格中的 Benchmark 名称，可查看 DM05 在对应数据集下的训练、评测文档或评测接入。

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

下面的基础环境只覆盖训练和 default backend 推理。fast backend 还需要单独安装
TensorRT Python/runtime、Triton，以及支持 PyTorch FlexAttention 的环境。

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

以上命令只会创建基础 OpenDM 环境。如果要使用
`--inference-config.backend fast`，还需要继续安装下面的 fast backend 环境层。

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

### Fast Backend 环境层

上面的 Docker / 本地安装还不足以运行 `--inference-config.backend fast`。请在同一个
`opendm` 环境中继续安装 fast 推理依赖层：

```bash
pip install -e ".[fast-infer]"
```

`fast-infer` extra 会安装 `onnx`、`triton==3.6.0` 和 `tensorrt`。Fast 启动不是“能用就
加速、不能用就回退”的可选优化：OpenDM 会直接构建或加载 TensorRT vision engine，
调用 Triton prefix/suffix kernels，并强制把 LLM attention backend 切到
`flex_attention`。因此 TensorRT、Triton 和 PyTorch FlexAttention 支持都是 fast 推理
的前置条件。

启动 fast backend 前，先在当前环境确认：

```bash
python -c "import tensorrt"
python -c "import triton"
python -c "import torch.nn.attention.flex_attention"
```

请使用提供 `torch.nn.attention.flex_attention` 的 PyTorch 版本，例如 `torch>=2.5`。
同时要预留首次 fast 启动时间：每个 checkpoint / image layout 第一次启动时，服务会先
导出 ONNX 并构建 TensorRT engine，之后 HTTP 服务才会就绪。

## 推理

下载 DM05 基础预训练模型后，可以使用 default backend 启动推理服务：

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

该示例使用三路图像输入和 14 维 state/action。不同机器人 profile、HTTP 请求字段、
微调模型启动命令、fast backend 配置、运行约束和问题排查参考
[DM05 推理指南](docs/zh/dm05_inference.md)。
新的接入方优先使用 `/v1/infer`。旧的 `/process_frame` multipart 接口仍作为 legacy 兼容路径保留，但会逐步被替换。

## 训练

### 数据准备

按照 OpenDM [数据使用指南](docs/zh/data.md)准备数据文件并注册数据集，并确保训练命令中的 `--data-config.dataset-name` 与实际注册的数据集名称一致。

训练脚本通过 `--data-config.dataset-name` 指定数据集名称。启动训练前，需要先在项目数据注册表中注册对应数据集。建议参考已有的 `opendm/dataset/demo.py`，复制一份新的数据集配置文件，例如 `opendm/dataset/my_robot.py`，然后修改数据集名称、数据路径、图像字段和状态描述。

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

字段说明：

- `my_robot`：注册到数据集表中的数据集名称，训练时通过 `--data-config.dataset-name my_robot` 使用。
- `jsonl_dir`：训练数据的 `jsonl` 文件目录。
- `image_dir`：图像文件目录。
- `image_keys`：数据中需要读取的图像字段名。
- `image_prompts`：与加载图像顺序对应的 prompt 标签（如 Head / Left wrist）。
- `robot_type`：数据对应的机型，用于选择 state 描述和该机型的归一化统计。
- `state_desc`：状态 / 动作各维度对应的机器人关节、夹爪等含义。

训练启动时，如果对应的归一化参数文件不存在，脚本会根据当前实验数据、action mode 和 chunk size 自动计算，并保存到 `./norm_stats/`。同一实验中相同机型的数据共享一组统计值，不同机型分别写入同一文件。

### 启动训练

完成环境安装、源码初始化和数据准备后，可以启动模型训练。训练脚本会读取指定数据集配置，加载基础模型 checkpoint，并按照配置启动训练。

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

参数说明：

- `--exp playground/dm05_sft_demo.py`：本示例使用 DM05 SFT demo 配置作为训练入口；如果自定义数据需要不同配置，可以复制并调整该入口。
- `--task train`：指定当前任务为训练模式。
- `--nproc_per_node 8`：单机启动的训练进程数，通常对应使用的 GPU 数量。
- `--data-config.dataset-name my_robot`：指定训练数据集名称，需要与项目中的数据配置保持一致。
- `--model-config.model-name-or-path ./checkpoints/DM05`：指定初始模型 checkpoint 路径。
- `--model-config.chunk-size 50`：指定模型一次预测的动作块（action chunk）长度。
- `--trainer-config.num-train-steps 50000`：总训练步数。

#### 启用 Weights & Biases 训练记录

W&B 是可选功能，只有传入项目名称时才会启用。OpenDM 已包含 `wandb` 依赖。

1. 在训练机器上完成认证：

   ```bash
   wandb login
   ```

   对于非交互式任务，可以改为设置 `WANDB_API_KEY`。不要将 API key 提交到代码仓库。

2. 在现有训练命令中增加以下参数：

   ```text
   --trainer-config.wandb-project <project-name>
   ```

   将 `<project-name>` 替换为要使用的 W&B 项目名称，例如 `dm05-sft`。删除该参数即可关闭 W&B。

训练开始后，日志会输出数据加载、模型初始化、loss、checkpoint 保存等信息。实际训练前请确认数据路径、模型权重路径和 GPU 数量均已正确配置。

## DM05 SFT 与自定义数据微调

建议先使用内置 demo 数据和 `playground/dm05_sft_demo.py` 跑通一次完整的 DM05 SFT 流程，熟悉数据格式、归一化统计、训练、推理和服务验证后，再替换为自己的机器人数据进行 SFT。参考 [DM05 SFT 与验证指南](docs/zh/dm05_finetuning.md)。

## Benchmark 微调参考流程

如需端到端微调 DM05，可以参考 benchmark 微调指南，其中包含数据准备、SFT 训练和 benchmark 评测。推理服务统一参考 [DM05 推理指南](docs/zh/dm05_inference.md)。

- LIBERO：[DM05 LIBERO 训练与评测指南](docs/zh/dm05_libero.md)
- RoboTwin2.0：[DM05 RoboTwin2.0 训练与评测指南](docs/zh/dm05_robotwin2.md)
- VLA-Arena：[DM05 VLA-Arena 训练与评测指南](docs/zh/dm05_vla_arena.md)
- SO101：[DM05 SO101 LoRA 训练指南](docs/zh/dm05_so101_lora_training.md)
- RoboChallenge Table 30 v2：[DM05 RoboChallenge Table 30 v2 推理指南](docs/zh/dm05_robochallenge.md)

## 使用指南

- 下载模型：参考[模型](#模型)或访问 [Dexmal Hugging Face](https://huggingface.co/Dexmal)
- 查看真机配置改动：参考 [AgileX COBOT Magic 与 DOS-W1 机型改动说明](docs/zh/robot_platforms.md)
- 准备数据：参考 [OpenDM 数据使用指南](docs/zh/data.md)
- 启动推理服务：参考 [DM05 推理指南](docs/zh/dm05_inference.md)
- 使用 demo 或自有数据进行 DM05 SFT：参考[DM05 SFT 与验证指南](docs/zh/dm05_finetuning.md)
- Benchmark 训练和评测：参考[DM05 LIBERO 训练与评测指南](docs/zh/dm05_libero.md)、[DM05 RoboTwin2.0 训练与评测指南](docs/zh/dm05_robotwin2.md)和[DM05 RoboChallenge Table 30 v2 推理指南](docs/zh/dm05_robochallenge.md)；LoRA SFT 参考[DM05 LIBERO LoRA 训练](docs/zh/dm05_libero_lora_training.md)和[DM05 SO101 LoRA 训练指南](docs/zh/dm05_so101_lora_training.md)

## 社区与支持

- 了解更多 Dexmal 产品与模型动态，请访问 [Dexmal 官网](https://www.dexmal.com/)。
- 获取 DM 模型权重，请访问 [Dexmal Hugging Face](https://huggingface.co/Dexmal)。
- 如果你在使用中遇到问题，欢迎通过 [GitHub Issues](https://github.com/dexmal/opendm/issues) 反馈。
- 如需进一步沟通，也可以扫描[微信二维码](docs/image/wechat.jpeg)与我们联系。

我们将持续开放更多模型权重、技术文档和示例。如果这个项目对你有帮助，欢迎在 GitHub 上给我们一颗星 [![GitHub](https://img.shields.io/github/stars/dexmal/opendm?color=5B5BD6)](https://github.com/dexmal/opendm)，你的支持是我们前进的动力。

## 许可

本项目采用 [Apache-2.0 许可证](LICENSE)。
