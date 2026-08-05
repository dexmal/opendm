# DM05 VLA-Arena Training and Evaluation Guide

This document describes how to use `DM05` for model training, inference service startup, and benchmark evaluation in VLA-Arena scenarios.

## VLA-Arena Training

### Prerequisites

Before training, make sure the following preparation is complete:

- **OpenDM environment installation and source initialization have been completed according to the official steps.**
- Training and inference require GPU resources. Recommended GPUs include A100, H100, H20, and 4090.

### Data Preparation

The VLA-Arena dataset and the base model can be downloaded from Hugging Face:

- VLA-Arena dataset: [Dexmal/vla_arena_L0_L](https://huggingface.co/datasets/Dexmal/vla_arena_L0_L)
- DM05 model: [Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

Download and organize the data in the OpenDM project root:

```bash
# Run from the OpenDM repository root.
cd opendm

# Download the VLA-Arena dataset
huggingface-cli download Dexmal/vla_arena_L0_L --repo-type dataset \
  --local-dir ./data/vla_arena_L0_L

# Extract split tar archives
cd ./data/vla_arena_L0_L
cat vla_arena_L0_L.tar.gz.part-* | tar -xzf -
cd -

# Download the DM05 base model
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

Confirm that the data directory is ready. The dataset layout should be:

```text
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

Prepare the dataset registration file and confirm how normalization statistics are generated:

```text
# Dataset registration file
opendm/dataset/vla_arena.py
```

Notes:

- `opendm/dataset/vla_arena.py` registers the VLA-Arena dataset. The default registered dataset name is `vla_arena_eef_L0_L`.
- When training starts, if the corresponding normalization statistics file does not exist, the script automatically computes it based on the current dataset, action mode, and action chunk length, then saves it under `./norm_stats/`.
- When saving a checkpoint, the normalization statistics used for training are copied to `norm_stats.json` under the checkpoint directory. Inference first reads `norm_stats.json` from the checkpoint directory. If it does not exist, inference looks for the matching file under `./norm_stats/` based on the current dataset, action mode, and action chunk length.
- `--data-config.dataset-name` in the training command must match a dataset name registered in `opendm/dataset/vla_arena.py`.

### Start Training

After the data, model, and registration file are ready, start VLA-Arena training:

```bash
# Run from the OpenDM repository root.
cd opendm

script/dm05_launcher.sh \
  --exp ./playground/dm05_vla_arena.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name vla_arena_eef_L0_L \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 60000
```

Arguments:

- `--exp ./playground/dm05_vla_arena.py`: training entry for VLA-Arena scenarios. This entry presets the action mode and default chunk size required by VLA-Arena.
- `--task train`: run in training mode.
- `--nproc_per_node 8`: number of GPUs used on a single node. 8 GPUs are recommended.
- `--data-config.dataset-name vla_arena_eef_L0_L`: VLA-Arena dataset name used for training.
- `--model-config.model-name-or-path ./checkpoints/DM05`: base model checkpoint path.
- `--trainer-config.num-train-steps 60000`: total number of training steps.

## VLA-Arena Inference

**After completing environment installation and source initialization**, you can start a model inference service with the DM05 VLA-Arena checkpoint trained in the previous step.

### Prerequisites

Before inference, make sure **OpenDM environment installation and source initialization have been completed according to the official steps.** Inference requires at least 1 GPU.

### Start Inference

- Inference first reads `norm_stats.json` from the model checkpoint directory. Make sure the checkpoint comes from a VLA-Arena training workflow that matches the current inference configuration.

```bash
# Start the inference service
script/dm05_launcher.sh \
  --exp playground/dm05_vla_arena.py \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint \
  --inference-config.output-action-dim 7
```

Arguments:

- `--exp playground/dm05_vla_arena.py`: inference entry for VLA-Arena scenarios, consistent with training.
- `--task inference`: task type. Use `inference` for inference.
- `--nproc_per_node 1`: number of GPUs used on a single node. 1 GPU is sufficient for inference.
- `--model-config.model-name-or-path ./checkpoints/DM05-vla-arena-checkpoint`: model checkpoint path.
- `--inference-config.output-action-dim 7`: VLA-Arena action output dimension.

Keep the inference service running after startup; the evaluation script communicates with it through an HTTP API.

## VLA-Arena Evaluation

### Preparation

1. Start the DM05 inference service according to the steps above, and keep it running.
2. Clone the VLA-Arena repository:

```bash
git clone https://github.com/PKU-Alignment/VLA-Arena
cd VLA-Arena
```

3. Install the VLA-Arena dependencies:

```bash
sudo apt-get install -y libosmesa6-dev libglfw3 libgl1-mesa-glx libglib2.0-0

pip install robosuite==1.5.1 bddl numpy==1.26.4 requests tqdm pyyaml "imageio[ffmpeg]" pillow

pip install -e .

export MUJOCO_GL=osmesa
```

4. Copy the DM05 evaluation files from the OpenDM repository into the VLA-Arena repository:

```bash
# Run from the VLA-Arena repository root.
mkdir -p vla_arena/models/DM05

# Copy from OpenDM tools directory
cp <path-to-opendm>/third_party/vla_arena/eval.py vla_arena/models/DM05/eval.py
cp <path-to-opendm>/third_party/vla_arena/eval_config.yaml vla_arena/models/DM05/eval_config.yaml
```

### Evaluation

#### Modify the Evaluation Configuration

Open `vla_arena/models/DM05/eval_config.yaml` and set `server_url` to the address of the running inference service:

```yaml
# ----- HTTP inference server -----
server_url: "http://<SERVER_IP>:7891/process_frame"
request_timeout: 30          # seconds

# ----- Model inference parameters -----
action_horizon: 20
replan_steps: 10
robot_type: "Franka"
batch_size: 1
speed: "0.5"

# ----- Task selection -----
# "all" evaluates all 170 tasks (11 suites × 3 levels)
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

Config values you may modify:

- `server_url`: address of the running DM05 inference service. Usually `http://<SERVER_IP>:7891/process_frame`.
- `task_suite_name`: task suite to evaluate. Set to `"all"` to run all 170 tasks, or specify a suite name such as `"safety_static_obstacles"`.
- `task_level`: task level to evaluate. Only used when `task_suite_name` is not `"all"`.
- `seeds`: list of random seeds for evaluation. Each seed runs a full evaluation pass.
- `num_trials_per_task`: number of episodes per task.
- `local_log_dir`: directory for evaluation results.

#### Start VLA-Arena Evaluation

Run the evaluation from the VLA-Arena repository root:

```bash
# Run from the VLA-Arena repository root.
cd VLA-Arena

python -m vla_arena.models.DM05.eval \
  --config vla_arena/models/DM05/eval_config.yaml \
  --output-dir ./experiments/eval_results
```

To evaluate a specific task list instead of all tasks:

```bash
python -m vla_arena.models.DM05.eval \
  --config vla_arena/models/DM05/eval_config.yaml \
  --task-list-file <path-to-task-list.txt> \
  --output-dir ./experiments/eval_results
```

#### Check Evaluation Results

Evaluation results are saved under `<output-dir>/seed_<N>/`. Main files include:

- `results_<timestamp>.json`: per-task and overall success rates and costs.
- `tasks_<timestamp>.csv`: per-task results in CSV format.
- `videos/<suite>/<task_id>_<episode>_<success|failure>.mp4`: rollout videos.
