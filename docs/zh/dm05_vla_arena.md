# DM05 VLA-Arena 训练与评测指南

本文档介绍如何使用 `DM05` 完成 VLA-Arena 场景下的模型训练、推理服务启动以及 benchmark 评测流程。

## VLA-Arena 训练

### 前提条件

开始训练前，请确认已经完成以下准备：

- **已按照官方步骤完成 OpenDM 环境安装和源码初始化。**
- 训练和推理需要使用 GPU 资源，推荐使用 A100, H100, H20, 4090 等 GPU 卡。

### 数据准备

VLA-Arena 数据和基础模型可从 Hugging Face 下载：

- VLA-Arena 数据集：[Dexmal/vla_arena_L0_L](https://huggingface.co/datasets/Dexmal/vla_arena_L0_L)
- DM05 模型：[Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

在 OpenDM 工程根目录下准备数据和模型：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

# 下载 VLA-Arena 数据集
huggingface-cli download Dexmal/vla_arena_L0_L --repo-type dataset \
  --local-dir ./data/vla_arena_L0_L

# 解压分片 tar 包
cd ./data/vla_arena_L0_L
cat vla_arena_L0_L.tar.gz.part-* | tar -xzf -
cd -

# 下载 DM05 基础模型
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

确认 VLA-Arena 数据目录已准备完成。数据集结构如下：

```
├── data
│   └── vla_arena_L0_L
│       ├── jsonl
│       │   ├── episode_00000.jsonl
│       │   ├── episode_00001.jsonl
│       │   └── ...
│       └── images
│           ├── episode_00000
│           │   ├── episode_00000_top_0000.png
│           │   ├── episode_00000_wrist_0000.png
│           │   └── ...
│           ├── episode_00001
│           └── ...
└── ...
```

准备数据集注册文件，并确认归一化参数生成方式：

```
# VLA-Arena 数据集注册文件
opendm/dataset/vla_arena.py
```

说明：

- `opendm/dataset/vla_arena.py` 用于注册 VLA-Arena 数据集，默认注册的数据集名称为 `vla_arena_eef_L0_L`。
- 训练启动时，如果对应的归一化参数文件不存在，脚本会根据当前数据集、action mode 和 action chunk 长度自动计算并保存到 `./norm_stats/`。
- checkpoint 保存时会同时把训练使用的归一化参数复制为 checkpoint 目录下的 `norm_stats.json`。推理会优先读取 checkpoint 目录下的 `norm_stats.json`；如果不存在，则根据当前数据集、action mode 和 action chunk 长度到 `./norm_stats/` 查找对应文件。
- 训练命令中的 `--data-config.dataset-name` 需要与 `opendm/dataset/vla_arena.py` 中注册的数据集名称保持一致。

### 启动训练

确认数据、模型和注册文件都准备完成后，启动 VLA-Arena 训练：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

script/dm05_launcher.sh \
  --exp ./playground/dm05_vla_arena.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name vla_arena_eef_L0_L \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 60000
```

参数说明：

- `--exp ./playground/dm05_vla_arena.py`：使用 VLA-Arena 场景的训练入口，该入口预设了 VLA-Arena 需要的 action mode 和默认 chunk size。
- `--task train`：指定当前任务为训练模式。
- `--nproc_per_node 8`：单节点使用的 GPU 数量，推荐 8 卡。
- `--data-config.dataset-name vla_arena_eef_L0_L`：指定训练使用的 VLA-Arena 数据集名称。
- `--model-config.model-name-or-path ./checkpoints/DM05`：指定基础模型 checkpoint 路径。
- `--trainer-config.num-train-steps 60000`：总训练步数。

## VLA-Arena 推理

**完成环境安装和源码初始化后**，可以利用上一步训练好的 DM05 VLA-Arena 模型 checkpoint 启动模型推理服务。

### 前提条件

推理前，请确认已经**按照官方步骤完成 OpenDM 环境安装和源码初始化。** 推理需要至少 1 张 GPU。

### 启动推理

- 推理会优先读取模型 checkpoint 目录下的 `norm_stats.json`；请确认该 checkpoint 来自与当前推理配置一致的 VLA-Arena 训练流程。

```bash
# 启动推理服务
script/dm05_launcher.sh \
  --exp playground/dm05_vla_arena.py \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint \
  --inference-config.output-action-dim 7
```

参数说明：

- `--exp playground/dm05_vla_arena.py`：使用 VLA-Arena 场景的推理入口，与训练阶段保持一致。
- `--task inference`：任务类型，推理时使用 inference。
- `--nproc_per_node 1`：单节点使用的 GPU 数量，推理使用 1 卡就可以。
- `--model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint`：模型 checkpoint 路径。
- `--inference-config.output-action-dim 7`：VLA-Arena 动作输出维度。

推理服务启动后需要保持运行，评测脚本会通过 HTTP 接口请求该服务完成动作预测。

## VLA-Arena 评测

### 准备阶段

1. 按照上述步骤启动 DM05 推理服务并保持运行。
2. 克隆 VLA-Arena 官方代码库：

```bash
git clone https://github.com/PKU-Alignment/VLA-Arena
cd VLA-Arena
```

3. 安装 VLA-Arena 依赖：

```bash
sudo apt-get install -y libosmesa6-dev libglfw3 libgl1-mesa-glx libglib2.0-0

pip install robosuite==1.5.1 bddl numpy==1.26.4 requests tqdm pyyaml "imageio[ffmpeg]" pillow

pip install -e .

export MUJOCO_GL=osmesa
```

4. 将 OpenDM 提供的 DM05 评测文件复制到 VLA-Arena 代码库中：

```bash
# 在 VLA-Arena 仓库根目录运行。
mkdir -p vla_arena/models/DM05

# 从 OpenDM tools 目录复制
cp <opendm路径>/third_party/vla_arena/eval.py vla_arena/models/DM05/eval.py
cp <opendm路径>/third_party/vla_arena/eval_config.yaml vla_arena/models/DM05/eval_config.yaml
```

### 评测阶段

#### 修改评测配置

打开 `vla_arena/models/DM05/eval_config.yaml`，将 `server_url` 修改为已启动的推理服务地址：

```yaml
# ----- HTTP inference server -----
server_url: "http://<SERVER_IP>:7891/process_frame"
request_timeout: 30          # 秒

# ----- Model inference parameters -----
action_horizon: 20
replan_steps: 10
robot_type: "Franka"
batch_size: 1
speed: "0.5"

# ----- Task selection -----
# "all" 评测全部 170 个任务（11 个 suite × 3 个 level）
task_suite_name: "all"
task_level: 0

# ----- Episode settings -----
num_trials_per_task: 10
seeds: [7, 42, 1000]

# ----- Output -----
model_name: "DM05"
local_log_dir: "./experiments/eval_results"
save_video_mode: "first_success_failure"
```

可修改的配置内容包括：

- `server_url`：已启动的 DM05 推理服务地址，通常为 `http://<SERVER_IP>:7891/process_frame`。
- `task_suite_name`：评测的任务集。设为 `"all"` 评测全部 170 个任务，也可以指定具体 suite 名称，如 `"safety_static_obstacles"`。
- `task_level`：评测的任务难度级别，仅在 `task_suite_name` 不为 `"all"` 时生效。
- `seeds`：随机种子列表，每个种子会完整跑一轮评测。
- `num_trials_per_task`：每个任务评测的 episode 数量。
- `local_log_dir`：评测结果保存目录。

#### 启动 VLA-Arena 评测

在 VLA-Arena 仓库根目录下执行：

```bash
# 在 VLA-Arena 仓库根目录运行。
cd VLA-Arena

python -m vla_arena.models.DM05.eval \
  --config vla_arena/models/DM05/eval_config.yaml \
  --output-dir ./experiments/eval_results
```

如需只评测指定任务列表而非全部任务，可通过 `--task-list-file` 传入任务列表文件：

```bash
python -m vla_arena.models.DM05.eval \
  --config vla_arena/models/DM05/eval_config.yaml \
  --task-list-file <任务列表文件路径> \
  --output-dir ./experiments/eval_results
```

#### 确认评测结果

评测结果保存在 `<output-dir>/seed_<N>/` 目录下，主要文件：

- `results_<timestamp>.json`：各任务及汇总的成功率和 cost。
- `tasks_<timestamp>.csv`：各任务结果的 CSV 格式汇总。
- `videos/<suite>/<task_id>_<episode>_<success|failure>.mp4`：rollout 视频。
