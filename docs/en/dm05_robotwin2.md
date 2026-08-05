# DM05 RoboTwin 2.0 Training and Evaluation Guide

This document describes how to use `DM05` for model training, inference service
startup, and benchmark evaluation in RoboTwin 2.0 scenarios.

## Reference Results

| Method | Clean | Randomized | Average |
| --- | ---: | ---: | ---: |
| DM0.5 | 93.6 | 93.3 | 93.5 |

## RoboTwin 2.0 Training

### Prerequisites

Before training, make sure the following preparation is complete:

- OpenDM has been installed and initialized according to the official steps.
- Training, inference, and evaluation require NVIDIA GPUs. Recommended GPUs
  include A100, H100, H20, and RTX 4090.
- Install the Hugging Face CLI if the `hf` command is not available:

```bash
pip install -U huggingface_hub
```

### Data Preparation

The complete RoboTwin 2.0 dataset and the DM05 base model can be downloaded
from Hugging Face:

- Complete RoboTwin 2.0 dataset: [Dexmal/robotwin2-full](https://huggingface.co/datasets/Dexmal/robotwin2-full)
- DM05 model: [Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

Prepare the dataset and model from the OpenDM repository root:

```bash
cd opendm

# Download all parts of the RoboTwin 2.0 dataset archive.
mkdir -p data/.hf_downloads/robotwin
hf download Dexmal/robotwin2-full \
  --repo-type dataset \
  --local-dir data/.hf_downloads/robotwin

# Join the split archive and extract its top-level robotwin2.0 directory
# under ./data, matching the path registered by OpenDM.
cat data/.hf_downloads/robotwin/robotwin2.tar.part-* \
  | tar -xf - -C data

# Download the DM05 base checkpoint.
hf download Dexmal/DM05 --local-dir checkpoints/DM05
```

Confirm that the extracted data has the following structure:

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

OpenDM registers the dataset in `opendm/dataset/robotwin2.py` with the name
`robotwin2_generalist`. The registration uses:

- `./data/robotwin2.0` as the dataset root;
- `images_1`, `images_2`, and `images_3` for the head, left-wrist, and
  right-wrist RGB views;
- the ALOHA RoboTwin2 embodiment with a 14-dimensional state and action;
- absolute joint-position actions.

When training starts, OpenDM computes normalization statistics automatically if
the matching file does not already exist under `./norm_stats/`. When a
checkpoint is saved, the statistics used for training are copied to
`norm_stats.json` in the checkpoint directory. Inference reads this checkpoint
file first, so keep it together with the model weights.

### Start Training

Run the following command from the OpenDM repository root:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robotwin2.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name robotwin2_generalist \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --model-config.chunk-size 50 \
  --trainer-config.num-train-steps 100000
```

Arguments:

- `--exp playground/dm05_robotwin2.py`: RoboTwin 2.0 training and inference
  entry point. It presets the dataset, action mode, optimizer, and trainer
  configuration used for this embodiment.
- `--task train`: starts training.
- `--nproc_per_node 8`: number of GPUs on one node. Eight GPUs are recommended
  for the reference configuration.
- `--data-config.dataset-name robotwin2_generalist`: dataset name registered in
  `opendm/dataset/robotwin2.py`.
- `--model-config.model-name-or-path ./checkpoints/DM05`: DM05 base checkpoint.
- `--model-config.chunk-size 50`: action chunk length. Use the same value for
  training, inference, and evaluation.
- `--trainer-config.num-train-steps 100000`: total number of training steps.

By default, training outputs are written under
`user_checkpoints/dm05_robotwin2`. Trainer settings such as batch size, save
interval, output directory, and total training steps can be overridden with
their corresponding command-line options.

## RoboTwin 2.0 Inference

See the [DM05 Inference Guide](dm05_inference.md) for the released RoboTwin 2.0 checkpoint, the complete service command, fast backend setup, and HTTP API usage.

## RoboTwin 2.0 Evaluation

### Prepare dexbotic-benchmark

When possible, use one GPU for the inference service and another for the
RoboTwin 2.0 simulator. The recommended benchmark client is the official Docker
image:

```bash
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark
git submodule update --init --recursive RoboTwin
docker pull dexmal/dexbotic_benchmark
```

RoboTwin 2.0 requires the assets, object-data texture library, and embodiment
files described in the
[RoboTwin installation guide](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#4-download-assets-robotwin-od-texture-library-and-embodiments).
Download these resources before starting evaluation.

### Configure an Evaluation Task

Use `evaluation/configs/robotwin2/adjust_bottle.yaml` as the reference
configuration. Set `base_url` to the inference server address:

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

Important fields:

- `task_name`: one of the 50 RoboTwin 2.0 tasks. Each task is evaluated
  independently.
- `task_config`: use `demo_clean` for the Clean setting and `demo_randomized`
  for the Randomized setting.
- `base_url`: address of the running DM05 inference service. For a remote
  service, use `http://<SERVER_IP>:7891`.
- `cameras`: must remain in head, left-wrist, right-wrist order to match
  `images_1`, `images_2`, and `images_3`.
- `action_horizon`: number of actions returned and executed per request. Keep it
  equal to the model chunk size of 50.
- `action_mode`: use `absolute` for absolute joint positions and `relative` for
  relative joint positions. This checkpoint uses `absolute`.
- `output_dir`: root directory for result files and rollout videos.

The benchmark runner evaluates 100 episodes for the selected task and setting.
To reproduce the aggregate Clean and Randomized scores, evaluate all 50 tasks
under both settings and aggregate their success rates.

### Run Evaluation with Docker

From the `dexbotic-benchmark` repository root, set `ROBOTWIN_ASSETS` to the
absolute path of the downloaded RoboTwin assets and run:

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

### Run Evaluation Locally

For simulator debugging, install the RoboTwin environment by following
`dexbotic-benchmark/docs/local_install.md`. After activating its `RoboTwin`
Conda environment, run:

```bash
cd dexbotic-benchmark
conda activate RoboTwin

# Recommended entry point.
bash scripts/env_sh/robotwin2.sh \
  evaluation/configs/robotwin2/adjust_bottle.yaml

# Or invoke the Python evaluator directly.
python evaluation/run_robotwin2_evaluation.py \
  --config evaluation/configs/robotwin2/adjust_bottle.yaml

# Selected configuration values can also be overridden at runtime.
python evaluation/run_robotwin2_evaluation.py \
  --config evaluation/configs/robotwin2/adjust_bottle.yaml \
  --set base_url http://localhost:7891 \
  --set output_dir ./result_test/robotwin2_evaluation
```

### Check Evaluation Results

Results are written below `output_dir` using the following hierarchy:

```text
<output_dir>/<task_name>/<task_config>/<timestamp>/
├── _result.txt
└── *.mp4
```

`_result.txt` records the success rate for the selected task and setting. When
video logging is enabled by the RoboTwin task configuration, the same timestamp
directory also contains rollout videos.
