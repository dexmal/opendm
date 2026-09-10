# DM05 RoboDojo-Sim Training and Evaluation Guide

This document describes how to run mem SFT on RoboDojo-Sim `cover_blocks` from `DM05-MEM`, start the inference service, and connect evaluation. Inference and evaluation use the checkpoint trained by this entry.

This page is the `cover_blocks` single-task SFT reference. README leaderboard numbers are for the released RoboDojo generalist [DM05-MEM-Robodojo-Sim](https://huggingface.co/Dexmal/DM05-MEM-Robodojo-Sim); see that model card for how to use it. The training settings and eval YAML on this page apply to checkpoints trained by this entry.

## RoboDojo-Sim Training

### Prerequisites

Before training, make sure the following preparation is complete:

- **OpenDM environment installation and source initialization have been completed according to the official steps.**
- Training, inference, and evaluation require GPU resources. Recommended GPUs include A100, H100, H20, and 4090. On lower-memory GPUs such as RTX 4090, Full SFT may fail due to peak memory use during FSDP initialization; LoRA fine-tuning is recommended.
- If the environment does not provide the `hf` command, install the Hugging Face CLI first:

```bash
pip install -U huggingface_hub
```

### Data Preparation

RoboDojo-Sim data and the base model can be downloaded from Hugging Face:

- RoboDojo-Sim dataset: [Dexmal/robodojo-sim](https://huggingface.co/datasets/Dexmal/robodojo-sim)
- DM05-MEM model: [Dexmal/DM05-MEM](https://huggingface.co/Dexmal/DM05-MEM)

Prepare the data and model in the OpenDM project root. The Hugging Face repo `Dexmal/robodojo-sim` ships `robodojo_sim.tar.gz.part-aa` and `robodojo_sim.tar.gz.part-ab`. After extract, the tree is `./data/robodojo_sim` (`jsonl/<task>/`, `video/<task>/`):

```bash
# Run from the OpenDM repository root.
cd opendm

mkdir -p data/.hf_downloads/robodojo_sim
hf download Dexmal/robodojo-sim \
  --repo-type dataset \
  --local-dir data/.hf_downloads/robodojo_sim

# Join the gzip split archive. Extraction creates data/robodojo_sim/.
cat data/.hf_downloads/robodojo_sim/robodojo_sim.tar.gz.part-* \
  | tar -xzf - -C data

hf download Dexmal/DM05-MEM --local-dir checkpoints/DM05-MEM
```

Confirm that the data directory has the following structure:

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

Video `url` values in the JSONL look like `./cover_blocks/episode_0000000/cam_head.mp4` and resolve against the registered `image_dir` (`./data/robodojo_sim/video`). The current SFT entry registers `cover_blocks` only.

Prepare the dataset registration file and confirm how normalization statistics are generated:

```text
# Official RoboDojo registration file
opendm/dataset/robodojo.py
```

Notes:

- `opendm/dataset/robodojo.py` registers RoboDojo-Sim datasets.
- When training starts, if the corresponding normalization statistics file does not exist, the script automatically computes it based on the current dataset, action mode, and action chunk length, then saves it under `./norm_stats/`.
- When saving a checkpoint, the normalization statistics used for training are copied to `norm_stats.json` under the checkpoint directory. Inference first reads that file.
- `--data-config.dataset-name` in the training command must match a registered name. The official example name is `robodojo_sim_cover_blocks`.
- The dataset `fps` is 25. The training entry samples 20 main-view history frames from `images_1` at 1 FPS. The JSONL does not need `history_images`.

### Start Training

After the data, model, and registration file are ready, start training:

```bash
# Run from the OpenDM repository root.
cd opendm

script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_robodojo_cover_blocks.py \
  --task train \
  --nproc_per_node 8 \
  --data-config.dataset-name robodojo_sim_cover_blocks \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --trainer-config.num-train-steps 30000
```

Arguments:

- `--exp playground/dm05_mem_sft_robodojo_cover_blocks.py`: RoboDojo-Sim `cover_blocks` mem SFT entry. It presets absolute actions, 20 history frames, chunk size 25, and the Aloha robot type.
- `--task train`: run in training mode.
- `--nproc_per_node 8`: number of GPUs used on a single node. 8 GPUs are recommended.
- `--data-config.dataset-name robodojo_sim_cover_blocks`: dataset name used for training.
- `--model-config.model-name-or-path ./checkpoints/DM05-MEM`: DM05-MEM base model path.
- `--trainer-config.num-train-steps 30000`: total number of training steps.

## RoboDojo-Sim Inference

Start the service with a checkpoint produced by this entry. See the [DM05 Inference Guide](dm05_inference.md) for the HTTP API and fast backend setup. This entry uses 20 history slots and `fast_prefix_len=2048`; FastInfer needs a 23-image TensorRT engine.

```bash
# Run from the OpenDM repository root.
cd opendm

script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_robodojo_cover_blocks.py \
  --task inference \
  --model-config.model-name-or-path ./user_checkpoints/dm05_mem_sft_robodojo_cover_blocks/checkpoint-30000 \
  --inference-config.port 7891
```

Replace `model-name-or-path` with the step directory you actually saved.

## RoboDojo-Sim Evaluation

### Preparation

1. Use at least 2 GPUs for the evaluation workflow when possible: one GPU for the policy server and another for the simulator. A100, H100, H20, and 4090 GPUs are supported.
2. Leaderboard evaluation is documented in [XPolicyLab PR #101](https://github.com/XPolicyLab/XPolicyLab/pull/101). The environment client talks to the policy server through the XPolicyLab protocol, not OpenDM HTTP `/v1/infer`.
3. Follow `policy/OpenDM/README.md` in XPolicyLab to install the policy environment, and place the weights trained by this entry at `policy/OpenDM/checkpoints/mm-robodojo/`. Mount `XPolicyLab/` beside the RoboDojo simulator `env_cfg/`, `scripts/`, `src/eval_client/`, and `task/` directories.

### Evaluation

#### Edit the eval config

Before running eval, edit XPolicyLab `policy/OpenDM/deploy.yml` so it matches this entry's training config. `eval.sh` starts the policy server from that file. Do not evaluate this entry's checkpoint until the YAML is updated.

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

Keep these fields aligned with training:

- `action_chunk_size` and `action_steps`: same as training `chunk_size`, 25 for this entry.
- `history_slots`: same as the training history length, 20 for this entry.
- `runtime_fps`: same as the registered dataset `fps`, 25.
- `model_action_mode` and `robot_type`: absolute actions and `Aloha` for this entry.

#### Start evaluation

Run from `XPolicyLab/policy/OpenDM` (policy GPU 0 and simulator GPU 1):

```bash
EVAL_ENV_TYPE=sim bash eval.sh \
  RoboDojo cover_blocks mm-robodojo \
  arx_x5 joint 0 0 1 opendm RoboDojo
```

Argument meanings and shared conventions are in the [XPolicyLab README](https://github.com/XPolicyLab/XPolicyLab). This entry supports Dual ARX5 (`arx_x5`), `joint` control, and absolute joints. One policy-server instance currently serves one evaluation client.
