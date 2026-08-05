# DM05 LIBERO 训练与评测指南

本文档介绍如何使用 `DM05` 完成 LIBERO 场景下的模型训练、推理服务启动以及 benchmark 评测流程。

## Libero 训练
### 前提条件
开始训练前，请确认已经完成以下准备：
  - **已按照官方步骤完成 OpenDM 环境安装和源码初始化。**
  - 训练、推理、评测需要使用 GPU 资源，推荐使用 A100, H100, H20, 4090 等 GPU 卡

### 数据准备
LIBERO 数据和基础模型可从 Hugging Face 下载：
  - LIBERO 数据集：[Dexmal/libero](https://huggingface.co/datasets/Dexmal/libero)
  - DM05 模型：[Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

在 OpenDM 工程根目录下准备数据和模型：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

# 下载并整理 LIBERO 数据
script/libero_runner.sh dataset

# 下载 DM05 checkpoint
script/libero_runner.sh model
```

  - `script/libero_runner.sh dataset` 会从 `Dexmal/libero` 下载数据到 `./data/.hf_downloads/libero`，自动处理 `libero.tar.part-*` 分片、解压并整理到 `./data/libero`。
  - `script/libero_runner.sh model` 默认会从 `Dexmal/DM05` 下载基础模型，并保存到 `./checkpoints/DM05`。如果需要使用其他模型仓库或保存目录，可通过 `--model-repo` 和 `--model-dir` 覆盖。

确认 LIBERO 数据目录已准备完成。数据集结构可参考 Hugging Face 中的说明，包含：

```
├── data
│   ├── libero
│   │   ├── libero_pi0_all
│   │   │   ├── image
│   │   │   └── jsonl
│   │   ├── libero_10
│   │   ├── libero_goal
│   │   ├── libero_object
│   │   └── libero_spatial
└── ...

```

准备数据集注册文件，并确认归一化参数生成方式：

```
# 使用官方提供的 LIBERO 注册文件
opendm/dataset/libero.py
```

  说明：
  - `opendm/dataset/libero.py` 用于注册 LIBERO 数据集。
  - 训练启动时，如果对应的归一化参数文件不存在，脚本会根据当前数据集、action mode 和 action chunk 长度自动计算并保存到 `./norm_stats/`。
  - checkpoint 保存时会同时把训练使用的归一化参数复制为 checkpoint 目录下的 `norm_stats.json`。推理会优先读取 checkpoint 目录下的 `norm_stats.json`；如果不存在，则根据当前数据集、action mode 和 action chunk 长度到 `./norm_stats/` 查找对应文件。
  - 训练命令中的 `--data-config.dataset-name` 需要与 `opendm/dataset/libero.py` 中注册的数据集名称保持一致。

如果使用官方 LIBERO 注册文件，示例数据集名称可使用： `libero_pi0_all`



### 启动训练
确认数据、模型和注册文件都准备完成后，启动 LIBERO 训练：

```bash
# 在 OpenDM 仓库根目录运行。
cd opendm

script/dm05_launcher.sh \
  --exp playground/dm05_libero.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name libero_pi0_all \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 10 \
  --trainer-config.num-train-steps 100000
```
参数说明：
- `--exp playground/dm05_libero.py`：使用 LIBERO 场景的训练入口，该入口预设了 LIBERO 需要的 action mode、state 输入方式和默认 chunk size。
- `--task train`：指定当前任务为训练模式。
- `--nproc_per_node 8`：单节点使用的 GPU 数量，推荐 8 卡。
- `--data-config.dataset-name libero_pi0_all`：指定训练使用的 LIBERO 数据集名称。
- `--model-config.model-name-or-path ./checkpoints/DM05`：指定基础模型 checkpoint 路径。
- `--model-config.chunk-size 10`：action chunk 长度，需要与训练和推理配置保持一致。
- `--trainer-config.num-train-steps 100000`：总训练步数。

## Libero 推理

发布版 LIBERO checkpoint、完整服务命令、fast backend 配置和 HTTP API 使用方法统一参考 [DM05 推理指南](dm05_inference.md)。

## Libero 评测
### 准备阶段
1. 建议使用至少 2 卡完成评测过程，一卡用于推理服务，另一卡用于启动 benchmark 评测。支持 A100, H100, H20, 4090 等 GPU 卡。
2. 按照 [DM05 推理指南](dm05_inference.md)启动服务并保持运行，benchmark 评测会通过 HTTP 接口请求该服务完成动作预测。
3. 配置 benchmark 环境，详细步骤与更多测试方法参考 dexbotic-benchmark 。 推荐使用官方提供的 Docker 镜像作为 benchmark 运行环境。该镜像作为评测客户端使用，负责运行 LIBERO 环境，并通过 HTTP 请求前面启动的 DM05 推理服务
```bash
# 获取评测脚本和配置文件
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark

# 只评测 LIBERO 时，仅初始化 LIBERO 子模块即可
git submodule update --init --recursive libero

# 拉取官方 benchmark Docker 镜像，作为评测客户端运行环境
docker pull dexmal/dexbotic_benchmark
```
  - 推荐优先使用 Docker 方式运行 benchmark。若需要在宿主机上直接调试 LIBERO 环境、评测脚本或依赖版本，可参考 dexbotic-benchmark docs/local_install.md 中的 LIBERO Environment Setup 安装说明。


### 评测阶段

#### 修改评测配置
  - 使用并修改 DM05 LIBERO 示例配置 `evaluation/configs/libero/example_dm05_libero.yaml`

```bash
benchmark: libero_spatial
num_trails_per_task: 50
num_steps_wait: 10
seed: 7

base_url: http://localhost:7891
replan_step: 10

send_state: true
send_image:
  - image
  - wrist_image
discrete_gripper: false
use_text_template: false

output_dir: "results/example_dm05_libero_spatial"
```

可修改替换的配置内容包括
- `benchmark`：选择要评测的 LIBERO 任务集，可选 libero_spatial、libero_goal、libero_object、libero_10。
- `base_url`：准备阶段提供的推理服务地址。通常为 http://<SERVER_IP>:7891。
- `replan_step`：每次调用模型后复用动作序列的步数。可根据官方配置或实验设置调整。
- `output_dir`：评测结果保存目录。启动脚本会基于该配置生成最终结果目录。

#### 启动 LIBERO 评测
- **推荐优先使用 Docker 方式运行评测。**


方式一：Docker 方式，在 dexbotic-benchmark 工程根目录下执行：
```bash
docker run --rm --gpus all --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  dexmal/dexbotic_benchmark \
  bash /workspace/scripts/env_sh/libero.sh \
  /workspace/evaluation/configs/libero/example_dm05_libero.yaml
```
  - 该方式会在 Docker 容器中运行 LIBERO 评测环境，并通过 `example_dm05_libero.yaml` 中的  `base_url` 请求已经启动的 DM05 推理服务。

方式二：本地环境方式
如果已经按照 dexbotic-benchmark docs/local_install.md  配置好本地 libero_env 环境，可以直接在宿主机运行：
```bash
cd dexbotic-benchmark
conda activate libero_env

# 推荐使用官方 shell 脚本启动：
bash scripts/env_sh/libero.sh \
  evaluation/configs/libero/example_dm05_libero.yaml

# 也可以直接使用 Python 脚本启动
python evaluation/run_libero_evaluation.py \
  --config evaluation/configs/libero/example_dm05_libero.yaml

# 也可以覆盖部分配置参数的方式执行
python evaluation/run_libero_evaluation.py \
  --config evaluation/configs/libero/example_dm05_libero.yaml \
  --set base_url http://localhost:7891 \
  --set output_dir results/example_dm05_libero_spatial
```
  - 该方式适合需要在本地调试 LIBERO 环境、评测脚本或依赖版本的场景。

#### 确认评测结果
评测结果会保存在 output_dir 下，主要文件：
- results.json
- config.yaml
- logs/evaluation.log
- videos/*.mp4

评测结束后，可以在 results.json 中查看详细评测以及汇总结果。
