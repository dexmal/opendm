# DM05 RoboTwin 2.0 训练与评测指南

本文档介绍如何使用 `DM05` 完成 RoboTwin 2.0 场景下的模型训练、推理服务启动以及 benchmark 评测。

## 参考指标

| 方法 | Clean | Randomized | Average |
| --- | ---: | ---: | ---: |
| DM0.5 | 93.6 | 93.3 | 93.5 |

## RoboTwin 2.0 训练

### 前提条件

开始训练前，请确认已经完成以下准备：

- 已按照官方步骤完成 OpenDM 环境安装和源码初始化。
- 训练、推理和评测需要使用 NVIDIA GPU，推荐使用 A100、H100、H20、RTX 4090 等 GPU。
- 如果环境中没有 `hf` 命令，请先安装 Hugging Face CLI：

```bash
pip install -U huggingface_hub
```

### 数据准备

RoboTwin 2.0 完整数据集和 DM05 基础模型可从 Hugging Face 下载：

- RoboTwin 2.0 完整数据集：[Dexmal/robotwin2-full](https://huggingface.co/datasets/Dexmal/robotwin2-full)
- DM05 模型：[Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

在 OpenDM 仓库根目录准备数据和模型：

```bash
cd opendm

# 下载 RoboTwin 2.0 完整数据集压缩包。
mkdir -p data/.hf_downloads/robotwin
hf download Dexmal/robotwin2-full robotwin2.tar.gz \
  --repo-type dataset \
  --local-dir data/.hf_downloads/robotwin

# 压缩包内包含一个顶层数据目录，将其中的 jsonl 和 video 目录整理到
# OpenDM 注册文件指定的位置。
mkdir -p data/robotwin2.0
tar -xzf data/.hf_downloads/robotwin/robotwin2.tar.gz \
  -C data/robotwin2.0 \
  --strip-components=1

# 下载 DM05 基础模型。
hf download Dexmal/DM05 --local-dir checkpoints/DM05
```

确认解压后的数据目录结构如下：

```text
data/robotwin2.0/
├── jsonl/
│   ├── adjust_bottle/
│   │   ├── clean/
│   │   └── randomized/
│   └── ...
└── video/
    ├── adjust_bottle/
    │   ├── clean/
    │   └── randomized/
    └── ...
```

OpenDM 在 `opendm/dataset/robotwin2.py` 中注册了名为
`robotwin2_generalist` 的数据集，注册配置包括：

- 数据根目录为 `./data/robotwin2.0`；
- `images_1`、`images_2`、`images_3` 分别对应头部、左腕和右腕 RGB 图像；
- 使用 ALOHA RoboTwin2 embodiment，state 和 action 均为 14 维；
- 使用绝对关节位置动作。

训练启动时，如果 `./norm_stats/` 下不存在匹配的归一化参数，OpenDM
会自动计算并保存。保存 checkpoint 时，训练使用的归一化参数也会复制到
checkpoint 目录下的 `norm_stats.json`。推理会优先读取该文件，因此请将它与模型权重一起保留。

### 启动训练

在 OpenDM 仓库根目录运行：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robotwin2.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name robotwin2_generalist \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50
```

参数说明：

- `--exp playground/dm05_robotwin2.py`：RoboTwin 2.0 训练和推理入口，预设了该
  embodiment 使用的数据集、action mode、优化器和训练参数。
- `--task train`：启动训练。
- `--nproc_per_node 8`：单节点使用的 GPU 数量，参考配置推荐使用 8 卡。
- `--data-config.dataset-name robotwin2_generalist`：
  `opendm/dataset/robotwin2.py` 中注册的数据集名称。
- `--model-config.model-name-or-path ./checkpoints/DM05`：DM05 基础模型路径。
- `--model-config.chunk-size 50`：action chunk 长度，训练、推理和评测必须保持一致。

默认训练输出目录为 `user_checkpoints/dm05_robotwin2`。如需调整 batch size、
保存间隔、输出目录或总训练步数，可通过对应的命令行参数覆盖默认配置。

## RoboTwin 2.0 推理

### 启动推理服务

从 Hugging Face 下载 [Dexmal/DM05-robotwin2](https://huggingface.co/Dexmal/DM05-robotwin2)
checkpoint：

```bash
hf download Dexmal/DM05-robotwin2 \
  --local-dir checkpoints/DM05-robotwin2-bf16
```

checkpoint 中必须包含与 RoboTwin 2.0 训练配置匹配的 `norm_stats.json`。
在 OpenDM 仓库根目录运行：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robotwin2.py \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-robotwin2-bf16 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-keys images_1 images_2 images_3 \
  --inference-config.port 7891
```

参数说明：

- `--exp playground/dm05_robotwin2.py`：使用与训练阶段相同的 RoboTwin 2.0 入口。
- `--task inference`：启动 HTTP 推理服务。
- `--nproc_per_node 1`：推理使用 1 卡即可。
- `--model-config.model-name-or-path`：已发布或本地训练得到的 RoboTwin 2.0 checkpoint 路径。
- `--model-config.chunk-size 50`：需要与训练配置以及 benchmark 的
  `action_horizon` 保持一致。
- `--inference-config.output-action-dim 14`：两个 ALOHA 机械臂各包含 6 个关节和 1 个夹爪值。
- `--inference-config.image-keys images_1 images_2 images_3`：三路图像输入，顺序需要与
  benchmark 的 `cameras` 配置一致。
- `--inference-config.port 7891`：HTTP 服务端口。

benchmark 评测期间需要保持推理服务运行。

## RoboTwin 2.0 评测

### 准备 dexbotic-benchmark

条件允许时，建议使用一张 GPU 运行推理服务，另一张 GPU 运行 RoboTwin 2.0
仿真环境。推荐使用官方 Docker 镜像作为 benchmark 客户端：

```bash
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark
git submodule update --init --recursive RoboTwin
docker pull dexmal/dexbotic_benchmark
```

RoboTwin 2.0 还需要 assets、object-data texture library 和 embodiment 文件。
请按照 [RoboTwin 安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#4-download-assets-robotwin-od-texture-library-and-embodiments)
完成下载后再启动评测。

### 配置评测任务

使用 `evaluation/configs/robotwin2/adjust_bottle.yaml` 作为参考配置，并将
`base_url` 修改为推理服务地址：

```yaml
# Basic experiment configuration (keep unchanged)
policy_name: dexbotic
task_name: adjust_bottle
task_config: demo_clean
ckpt_setting: dexbotic
seed: 0
instruction_type: seen

# Add Parameters You Need
base_url: http://localhost:7891
output_dir: ./result_test/robotwin2_evaluation
cameras: "head_camera_rgb,left_camera_rgb,right_camera_rgb"
action_horizon: 50
action_mode: absolute
```

重要配置项：

- `task_name`：RoboTwin 2.0 的 50 个任务之一，每个任务需要独立评测。
- `task_config`：Clean 设置使用 `demo_clean`，Randomized 设置使用
  `demo_randomized`。
- `base_url`：已启动的 DM05 推理服务地址；远程服务通常填写
  `http://<SERVER_IP>:7891`。
- `cameras`：必须保持头部、左腕、右腕的顺序，以对应 `images_1`、
  `images_2`、`images_3`。
- `action_horizon`：每次请求返回并执行的动作数，需要与模型的 chunk size 50 保持一致。
- `action_mode`：绝对关节位置使用 `absolute`，相对关节位置使用 `relative`；当前
  checkpoint 使用 `absolute`。
- `output_dir`：评测结果和 rollout 视频的保存根目录。

benchmark runner 会对所选任务和设置评测 100 个 episode。要复现汇总的 Clean 和
Randomized 指标，需要分别在两种设置下评测全部 50 个任务，再汇总各任务成功率。

### 使用 Docker 启动评测

在 `dexbotic-benchmark` 仓库根目录，将 `ROBOTWIN_ASSETS` 设置为已下载的
RoboTwin assets 绝对路径，然后运行：

```bash
ROBOTWIN_ASSETS=/absolute/path/to/robotwin/assets

docker run --rm --gpus all --network host \
  -v "$ROBOTWIN_ASSETS":"$ROBOTWIN_ASSETS" \
  -v "$ROBOTWIN_ASSETS":/app/assets \
  -v "$ROBOTWIN_ASSETS":/app/RoboTwin/assets \
  -v "$PWD/evaluation":/app/evaluation \
  -v "$PWD/scripts":/app/scripts \
  -v "$PWD/result_test":/app/result_test \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  dexmal/dexbotic_benchmark \
  bash scripts/env_sh/robotwin2.sh \
  evaluation/configs/robotwin2/adjust_bottle.yaml
```

### 使用本地环境启动评测

如需调试仿真环境，请参考 `dexbotic-benchmark/docs/local_install.md` 安装
RoboTwin 环境。激活 `RoboTwin` Conda 环境后运行：

```bash
cd dexbotic-benchmark
conda activate RoboTwin

# 推荐使用官方 shell 脚本。
bash scripts/env_sh/robotwin2.sh \
  evaluation/configs/robotwin2/adjust_bottle.yaml

# 也可以直接使用 Python evaluator。
python evaluation/run_robotwin2_evaluation.py \
  --config evaluation/configs/robotwin2/adjust_bottle.yaml

# 还可以在运行时覆盖部分配置。
python evaluation/run_robotwin2_evaluation.py \
  --config evaluation/configs/robotwin2/adjust_bottle.yaml \
  --set base_url http://localhost:7891 \
  --set output_dir ./result_test/robotwin2_evaluation
```

### 确认评测结果

评测结果会按照以下层级保存在 `output_dir` 下：

```text
<output_dir>/<task_name>/<task_config>/<timestamp>/
├── _result.txt
└── *.mp4
```

`_result.txt` 记录所选任务和设置的成功率。如果 RoboTwin 任务配置启用了视频记录，
同一个时间戳目录下还会包含 rollout 视频。
