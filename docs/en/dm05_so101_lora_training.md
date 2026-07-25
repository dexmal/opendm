# DM05 SO101 LoRA Training

This page is the developer reference for running DM05 LoRA SFT on SO101 data
from `playground/dm05_so101_lora.py`.

## When to Use LoRA

Use LoRA when you want to adapt DM05 to SO101 without updating the full model.
This approach allows efficient fine-tuning for the SO101 pick cube task while
preserving the base model's general capabilities.

| Item | Configuration value |
| --- | --- |
| Hardware | 8x GPU (configurable) |
| Per-device batch | 8 |
| Gradient accumulation | 1 |
| Train steps | 10,000 (default) |
| Save interval | 1,000 steps |
| Optimizer | AdamW over LoRA and trainable weights |
| LR / warmup | `1e-4` / `1000` |
| LoRA | Enabled (`use_lora=True`) |
| Target modules | `all-linear` |
| Attention | LLM `flex_attention`, vision `flash_attention_2`, action `sdpa` |
| Gradient checkpointing | VLM GC on, AE GC on |

## SO101 Data

The SO101 training target is `so101_pick_cube`, registered in
`opendm/dataset/so101.py`.

| Field | Value |
| --- | --- |
| Dataset name | `so101_pick_cube` |
| Data root | `./data/so101_pick_cube` |
| JSONL root | `./data/so101_pick_cube/jsonl` |
| Image root | `./data/so101_pick_cube/image` |
| Norm stats | Auto-generated under `./norm_stats/` |
| Image keys | `images_1`, `images_2` |
| Action mode | `relative` |
| State | Included (`add_state=True`) |
| Action dim | 6 |
| Chunk size | 50 |

Download and organize the SO101 data with `script/so101_runner.sh`:

```bash
pip install -U huggingface_hub
script/so101_runner.sh dataset
```

The script downloads `Dexmal/so101_pick_cube` from Hugging Face into
`./data/.hf_downloads/so101_pick_cube`, extracts archive parts if needed, and
organizes the final dataset under `./data/so101_pick_cube`.

To use a different data location:

```bash
script/so101_runner.sh dataset --data-root /path/to/data/so101_pick_cube
```

After download, `so101_pick_cube` should have this layout:

```text
data/so101_pick_cube/
  jsonl/
    episode_00000.jsonl
    episode_00001.jsonl
    ...
  videos/
    so101_YYYYMMDD_HHMMSS_filtered/
      file-000.mp4_top.mp4
      file-000.mp4_wrist.mp4
      ...
```

Each JSONL row represents one frame:

```json
{
  "prompt": "Pick the cube and place it in the plate.",
  "state": [-2.29, -102.81, 95.82, 54.02, 2.68, 0.68],
  "action": [-0.84, -104.22, 99.16, 54.24, 2.24, 0.16],
  "is_robot": true,
  "extra": {"subtask": "Pick the cube and place it in the plate.", "timestamp": 0.70, "episode_index": 0, "cube_color": "orange"},
  "images_1": {"type": "video", "url": "episode_00000/camera_top.mp4", "frame_idx": 21, "_camera_name": "top"},
  "images_2": {"type": "video", "url": "episode_00000/camera_wrist.mp4", "frame_idx": 21, "_camera_name": "wrist"}
}
```

Norm stats are computed automatically during training when the matching file is
missing. Recompute them when the action/state distribution, action mode, or
action chunk length changes.

## Prepare Inputs

1. Install the repo:

```bash
conda create -n opendm python=3.10
conda activate opendm
pip install -e .
```

2. Download or mount DM05 checkpoints:
   - For training: `./checkpoints/DM05` - Base DM05 model
   - For inference: Both `./checkpoints/DM05` and `./checkpoints/DM05-SO101-Pick-Cube` are needed

3. Ensure your SO101 dataset is located at `./data/so101_pick_cube`, or update
   `opendm/dataset/so101.py` to point at your data location.

4. Confirm that the dataset name, action mode, and chunk size match the
   training recipe. If the matching norm stats file does not exist, training
   will compute it automatically under `./norm_stats/`.

## Local Command

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_so101_lora.py \
  --task train \
  --nproc_per_node 8 \
  --model-config.model-name-or-path ./checkpoints/DM05
```

`playground/dm05_so101_lora.py` already provides the SO101 LoRA defaults,
including dataset name, action mode, chunk size, attention settings, learning
rate, warmup steps, batch size, save interval, and total training steps.
Override these options only when you need to change the reference recipe.

## What Gets Trained

The LoRA configuration wraps every supported linear layer and saves selected
DM05 action modules densely. The `dm05_time_modulators` alias expands to all
action-expert input, MLP, and final time modulators.

The trainable summary is written to
`user_checkpoints/<output_dir>/trainable_summaries/dm05_lora_so101_pick_cube.json`
by default. Check it before trusting a run:

- `target_modules` should resolve from `all-linear`.
- `unexpected_trainable_parameters` should be empty.
- The dense saved modules should include action projections, time MLPs, and
  time modulators.

## Checkpoints and Inference

Step checkpoints are the canonical artifacts for multi-GPU LoRA/FSDP training.
For inference, pass the LoRA checkpoint path as `--model-config.model-name-or-path`;
the loader reads `adapter_config.json`, loads the recorded base model, and merges
the adapter for inference.

Use the checkpoint path produced by your training run. For example, if the
training output directory is `${TRAINING_OUTPUT_DIR}`, use a checkpoint such as
`${TRAINING_OUTPUT_DIR}/checkpoint-4000`.

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_so101_lora.py \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-SO101-Pick-Cube \
  --inference-config.output-action-dim 6
```
