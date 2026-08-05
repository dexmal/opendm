# DM05 LIBERO Training and Evaluation Guide

This document describes how to use `DM05` for model training, inference service startup, and benchmark evaluation in LIBERO scenarios.

## LIBERO Training

### Prerequisites

Before training, make sure the following preparation is complete:

- **OpenDM environment installation and source initialization have been completed according to the official steps.**
- Training, inference, and evaluation require GPU resources. Recommended GPUs include A100, H100, H20, and 4090.

### Data Preparation

LIBERO data and the base model can be downloaded from Hugging Face:

- LIBERO dataset: [Dexmal/libero](https://huggingface.co/datasets/Dexmal/libero)
- DM05 model: [Dexmal/DM05](https://huggingface.co/Dexmal/DM05)

Prepare the data and model in the OpenDM project root:

```bash
# Run from the OpenDM repository root.
cd opendm

# Download and organize LIBERO data
script/libero_runner.sh dataset

# Download the DM05 checkpoint
script/libero_runner.sh model
```

- `script/libero_runner.sh dataset` downloads data from `Dexmal/libero` to `./data/.hf_downloads/libero`, handles the `libero.tar.part-*` archive parts, extracts them, and organizes the result under `./data/libero`.
- `script/libero_runner.sh model` downloads the base model from `Dexmal/DM05` by default and saves it to `./checkpoints/DM05`. To use another model repository or save directory, override `--model-repo` and `--model-dir`.

Confirm that the LIBERO data directory is ready. The dataset layout can follow the Hugging Face description:

```text
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

Prepare the dataset registration file and confirm how normalization statistics are generated:

```text
# Official LIBERO registration file
opendm/dataset/libero.py
```

Notes:

- `opendm/dataset/libero.py` registers LIBERO datasets.
- When training starts, if the corresponding normalization statistics file does not exist, the script automatically computes it based on the current dataset, action mode, and action chunk length, then saves it under `./norm_stats/`.
- When saving a checkpoint, the normalization statistics used for training are copied to `norm_stats.json` under the checkpoint directory. Inference first reads `norm_stats.json` from the checkpoint directory. If it does not exist, inference looks for the matching file under `./norm_stats/` based on the current dataset, action mode, and action chunk length.
- `--data-config.dataset-name` in the training command must match a dataset name registered in `opendm/dataset/libero.py`.

If you use the official LIBERO registration file, the example dataset name is `libero_pi0_all`.

### Start Training

After the data, model, and registration file are ready, start LIBERO training:

```bash
# Run from the OpenDM repository root.
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

Arguments:

- `--exp playground/dm05_libero.py`: training entry for LIBERO scenarios. This entry presets the action mode, state input mode, and default chunk size required by LIBERO.
- `--task train`: run in training mode.
- `--nproc_per_node 8`: number of GPUs used on a single node. 8 GPUs are recommended.
- `--data-config.dataset-name libero_pi0_all`: LIBERO dataset name used for training.
- `--model-config.model-name-or-path ./checkpoints/DM05`: base model checkpoint path.
- `--model-config.chunk-size 10`: action chunk length. It must remain consistent between training and inference.
- `--trainer-config.num-train-steps 100000`: total number of training steps.

## LIBERO Inference

See the [DM05 Inference Guide](dm05_inference.md) for the released LIBERO checkpoint, the complete service command, fast backend setup, and HTTP API usage.

## LIBERO Evaluation

### Preparation

1. Use at least 2 GPUs for the evaluation workflow when possible: one GPU for the inference service and another for benchmark evaluation. A100, H100, H20, and 4090 GPUs are supported.
2. Start the service with the [DM05 Inference Guide](dm05_inference.md). Keep it running because benchmark evaluation sends HTTP requests to this service for action prediction.
3. Configure the benchmark environment. For detailed steps and more test methods, see `dexbotic-benchmark`. The official Docker image is recommended as the benchmark client environment. This image runs the LIBERO environment and sends HTTP requests to the DM05 inference service started above.

```bash
# Get evaluation scripts and configuration files
git clone https://github.com/dexmal/dexbotic-benchmark.git
cd dexbotic-benchmark

# If only evaluating LIBERO, initialize only the LIBERO submodule
git submodule update --init --recursive libero

# Pull the official benchmark Docker image as the evaluation client environment
docker pull dexmal/dexbotic_benchmark
```

- Docker is recommended for running the benchmark. If you need to debug the LIBERO environment, evaluation scripts, or dependency versions directly on the host, refer to the LIBERO Environment Setup section in `dexbotic-benchmark/docs/local_install.md`.

### Evaluation

#### Modify the Evaluation Configuration

- Use and modify the DM05 LIBERO example configuration at `evaluation/configs/libero/example_dm05_libero.yaml`.

```yaml
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

Config values you may modify include:

- `benchmark`: LIBERO task suite to evaluate. Options include `libero_spatial`, `libero_goal`, `libero_object`, and `libero_10`.
- `base_url`: inference service address prepared in the previous stage. Usually `http://<SERVER_IP>:7891`.
- `replan_step`: number of steps to reuse the action sequence after each model call. Adjust it according to the official configuration or your experiment setup.
- `output_dir`: directory for evaluation results. The startup script generates the final result directory based on this configuration.

#### Start LIBERO Evaluation

- **Docker is recommended for evaluation.**

Option 1: Docker. Run this command in the `dexbotic-benchmark` project root:

```bash
docker run --rm --gpus all --network host \
  -v "$(pwd)":/workspace \
  -w /workspace \
  dexmal/dexbotic_benchmark \
  bash /workspace/scripts/env_sh/libero.sh \
  /workspace/evaluation/configs/libero/example_dm05_libero.yaml
```

- This runs the LIBERO evaluation environment in Docker and requests the running DM05 inference service through `base_url` in `example_dm05_libero.yaml`.

Option 2: Local environment.
If you have already configured the local `libero_env` environment according to `dexbotic-benchmark/docs/local_install.md`, run the evaluation directly on the host:

```bash
cd dexbotic-benchmark
conda activate libero_env

# Recommended: start with the official shell script
bash scripts/env_sh/libero.sh \
  evaluation/configs/libero/example_dm05_libero.yaml

# Or start with the Python script directly
python evaluation/run_libero_evaluation.py \
  --config evaluation/configs/libero/example_dm05_libero.yaml

# Or override selected configuration values
python evaluation/run_libero_evaluation.py \
  --config evaluation/configs/libero/example_dm05_libero.yaml \
  --set base_url http://localhost:7891 \
  --set output_dir results/example_dm05_libero_spatial
```

- This mode is suitable when you need to debug the LIBERO environment, evaluation scripts, or dependency versions locally.

#### Check Evaluation Results

Evaluation results are saved under `output_dir`. Main files include:

- `results.json`
- `config.yaml`
- `logs/evaluation.log`
- `videos/*.mp4`

After evaluation finishes, view detailed and summary results in `results.json`.
