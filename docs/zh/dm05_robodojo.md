# DM05 RoboDojo-Sim 训练与评测指南

本文档介绍如何使用 `DM05-MEM` 在 RoboDojo-Sim `cover_blocks` 上完成 mem SFT 训练、启动推理服务，以及对接评测。推理和评测使用本入口训练得到的 checkpoint。

本文提供 `cover_blocks` 单任务微调参考流程。README 榜单结果对应已发布的 RoboDojo generalist 模型 [DM05-MEM-Robodojo-Sim](https://huggingface.co/Dexmal/DM05-MEM-Robodojo-Sim)，其使用方式见模型说明；本文的训练配置及评测 YAML 适用于本入口训练得到的 checkpoint。

## RoboDojo-Sim 训练

### 前提条件

开始训练前，请确认已经完成以下准备：

- **已按照官方步骤完成 OpenDM 环境安装和源码初始化。**
- 训练、推理、评测需要使用 GPU 资源，推荐使用 A100、H100、H20、4090 等 GPU 卡。在 RTX 4090 等显存较小的 GPU 上执行 Full SFT 时，FSDP 初始化阶段的显存峰值可能导致训练失败，建议采用 LoRA 微调。
- 如果环境中没有 `hf` 命令，请先安装 Hugging Face CLI：

```bash
pip install -U huggingface_hub
```

### 数据准备

RoboDojo-Sim 数据和基础模型可从 Hugging Face 下载：

- RoboDojo-Sim 数据集：[Dexmal/robodojo-sim](https://huggingface.co/datasets/Dexmal/robodojo-sim)
- DM05-MEM 模型：[Dexmal/DM05-MEM](https://huggingface.co/Dexmal/DM05-MEM)

在 OpenDM 工程根目录下准备数据和模型。Hugging Face 仓库 `Dexmal/robodojo-sim` 提供 `robodojo_sim.tar.gz.part-aa` 和 `robodojo_sim.tar.gz.part-ab`。解压后目录为 `./data/robodojo_sim`（`jsonl/<task>/`、`video/<task>/`）：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

mkdir -p data/.hf_downloads/robodojo_sim
hf download Dexmal/robodojo-sim \
  --repo-type dataset \
  --local-dir data/.hf_downloads/robodojo_sim

# 拼接 gzip 分卷。解压后得到 data/robodojo_sim/。
cat data/.hf_downloads/robodojo_sim/robodojo_sim.tar.gz.part-* \
  | tar -xzf - -C data

hf download Dexmal/DM05-MEM --local-dir checkpoints/DM05-MEM
```

确认数据目录结构如下：

```text
data/robodojo_sim/
├── jsonl/
│   ├── cover_blocks/
│   │   └── episode_*.jsonl
│   └── ...
└── video/
    ├── cover_blocks/
    │   └── episode_*/
    │       ├── cam_head.mp4
    │       ├── cam_left_wrist.mp4
    │       └── cam_right_wrist.mp4
    └── ...
```

JSONL 中的视频 `url` 形如 `./cover_blocks/episode_0000000/cam_head.mp4`，相对于注册的 `image_dir`（`./data/robodojo_sim/video`）解析。当前 SFT 入口只注册了 `cover_blocks`。

准备数据集注册文件，并确认归一化参数生成方式：

```text
# 使用官方提供的 RoboDojo 注册文件
opendm/dataset/robodojo.py
```

说明：

- `opendm/dataset/robodojo.py` 用于注册 RoboDojo-Sim 数据集。
- 训练启动时，如果对应的归一化参数文件不存在，脚本会根据当前数据集、action mode 和 action chunk 长度自动计算并保存到 `./norm_stats/`。
- checkpoint 保存时会同时把训练使用的归一化参数复制为 checkpoint 目录下的 `norm_stats.json`。推理会优先读取 checkpoint 目录下的 `norm_stats.json`。
- 训练命令中的 `--data-config.dataset-name` 需要与注册名称一致。官方示例名称为 `robodojo_sim_cover_blocks`。
- 该数据集 `fps` 为 25。训练入口从主视角 `images_1` 按 1 FPS 回看 20 帧历史，JSONL 不需要 `history_images`。

### 启动训练

确认数据、模型和注册文件都准备完成后，启动训练：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_robodojo_cover_blocks.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name robodojo_sim_cover_blocks \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --trainer-config.num-train-steps 30000
```

参数说明：

- `--exp playground/dm05_mem_sft_robodojo_cover_blocks.py`：RoboDojo-Sim `cover_blocks` mem SFT 入口，预设了绝对动作、20 帧历史、chunk size 25 和 Aloha 机型。
- `--task train`：指定当前任务为训练模式。
- `--nproc_per_node 8`：单节点使用的 GPU 数量，推荐 8 卡。
- `--data-config.dataset-name robodojo_sim_cover_blocks`：指定训练使用的数据集名称。
- `--model-config.model-name-or-path ./checkpoints/DM05-MEM`：指定 DM05-MEM 基础模型路径。
- `--trainer-config.num-train-steps 30000`：总训练步数。

## RoboDojo-Sim 推理

使用本入口训练得到的 checkpoint 启动服务。HTTP API 和 fast backend 配置参考 [DM05 推理指南](dm05_inference.md)。本入口为 20 个历史槽、`fast_prefix_len=2048`；FastInfer 需要 23 图 TensorRT engine。

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_robodojo_cover_blocks.py \
  --task inference \
  --model-config.model-name-or-path ./user_checkpoints/dm05_mem_sft_robodojo_cover_blocks/checkpoint-30000 \
  --inference-config.port 7891
```

将 `model-name-or-path` 换成实际保存的 step 目录。

## RoboDojo-Sim 评测

### 准备阶段

1. 建议使用至少 2 卡完成评测过程，一卡用于策略服务，另一卡用于仿真环境。支持 A100、H100、H20、4090 等 GPU 卡。
2. RoboDojo-Sim 排行榜评测接入见 [XPolicyLab PR #101](https://github.com/XPolicyLab/XPolicyLab/pull/101)，环境客户端与策略服务通过 XPolicyLab 协议通信，而不是 OpenDM HTTP `/v1/infer`。
3. 按照 XPolicyLab `policy/OpenDM/README.md` 安装策略环境，并将本入口训练得到的权重放到 `policy/OpenDM/checkpoints/mm-robodojo/`。将 `XPolicyLab/` 与 RoboDojo 仿真侧的 `env_cfg/`、`scripts/`、`src/eval_client/`、`task/` 放在一起。

### 评测阶段

#### 修改评测配置

评测前必须修改 XPolicyLab `policy/OpenDM/deploy.yml`，与本入口的训练配置对齐。`eval.sh` 读取该文件启动策略服务，未改 yml 时不能直接评测本入口的 checkpoint。

```yaml
action_chunk_size: 25
action_steps: 25
history_enabled: true
history_slots: 20
history_fps: 1.0
runtime_fps: 25.0
model_action_mode: absolute
robot_type: Aloha
speed: "0.5"
model_max_length: 1536
```

需要与训练保持一致的项包括：

- `action_chunk_size`、`action_steps`：与训练 `chunk_size` 相同，本入口为 25。
- `history_slots`：与训练历史帧数相同，本入口为 20。
- `runtime_fps`：与数据集注册 `fps` 相同，为 25。
- `model_action_mode`、`robot_type`：本入口为绝对动作、`Aloha`。

#### 启动评测

在 `XPolicyLab/policy/OpenDM` 下运行（策略 GPU 0、仿真 GPU 1 示例）：

```bash
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo cover_blocks mm-robodojo \
  arx_x5 joint 0 0 1 opendm RoboDojo
```

参数含义与共享约定见 [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab)。本入口对应 Dual ARX5（`arx_x5`）、`joint`、绝对关节。一个 policy server 同时只服务一个评测 client。
