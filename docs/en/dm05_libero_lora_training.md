# DM05 LIBERO LoRA Training

This page is the developer reference for running DM05 LoRA SFT on LIBERO data
from `playground/dm05_libero_lora.py`.

## When to Use LoRA

Use LoRA when you want to adapt DM05 to LIBERO without updating the full model.
The current reference recipe follows a successful internal validation run:

| Item | Reference value |
| --- | --- |
| Hardware | 8x NVIDIA RTX 4090D |
| Global batch | 32 |
| Per-device batch | 4 |
| Gradient accumulation | 1 |
| Train steps | 50,000 |
| Save interval | 10,000 steps |
| Optimizer | AdamW over LoRA and `modules_to_save` trainable weights |
| LR / warmup | `5e-4` / `500` |
| LoRA rank / alpha / dropout | `32` / `16` / `0.0` |
| Target modules | `all-linear` |
| Dense modules to save | `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`, `dm05_time_modulators` |
| Attention | LLM `eager`, vision `sdpa`, action `sdpa` |
| Gradient checkpointing | VLM GC off, AE GC off |

The validation run trained through the full schedule, with a best completed
checkpoint of `98.30%` overall LIBERO success at 49k.
Its pretrained checkpoint was experiment-specific; use the checkpoint that
matches your dataset and target robot.

## LIBERO Data

The built-in LIBERO training target is `libero_pi0_all`, registered in
`opendm/dataset/libero.py`.

| Field | Value |
| --- | --- |
| Dataset name | `libero_pi0_all` |
| JSONL root | `./data/libero/libero_pi0_all` |
| Image root | `./data/libero/libero_pi0_all/image` |
| Norm stats | Auto-generated under `./norm_stats/` |
| Image keys | `images_1`, `images_2` |
| Action mode | `absolute` |
| State desc | 6 joints + 2 gripper dims |
| Action dim | 7 |

Download and organize the LIBERO data with `script/libero_runner.sh`:

```bash
pip install -U huggingface_hub
script/libero_runner.sh dataset
```

The script downloads `Dexmal/libero` from Hugging Face into
`./data/.hf_downloads/libero`, extracts archive parts if needed, and organizes
the final dataset under `./data/libero`.

To use a different data location:

```bash
script/libero_runner.sh dataset --data-root /path/to/data/libero
```

The `train` and `all` commands forward `--data-root` to the default
`libero_pi0_all` training configuration. For direct launcher commands, pass:

```bash
--data-config.jsonl-dir /path/to/data/libero/libero_pi0_all \
--data-config.image-dir /path/to/data/libero/libero_pi0_all/image
```

After download, `libero_pi0_all` should have this layout:

```text
data/libero/libero_pi0_all/
  jsonl/
    <episode>.jsonl
  image/
    ...
```

Each JSONL row is one frame. For `action_mode="absolute"`, every row must carry
the current action because the collator stacks `[t, t+1, ..., t+chunk-1]` into
the training target.

```json
{
  "images_1": {"type": "image", "url": "./episode_000/camera_0/000000.jpg"},
  "images_2": {"type": "image", "url": "./episode_000/camera_1/000000.jpg"},
  "state": [0.12, -0.04, 0.31, 1.22, -0.18, 0.44, 1.0, 0.0],
  "action": [0.13, -0.03, 0.30, 1.20, -0.17, 0.45, 1.0],
  "prompt": "put the object into the bowl"
}
```

Norm stats are computed automatically during training when the matching file is
missing. Recompute them when the action/state distribution, action mode, or
action chunk length changes.

## Reference Milestones

The reference validation run used 8x NVIDIA RTX 4090D with the recipe above and
evaluated LIBERO with 2,000 episodes per checkpoint across the standard suites.
These numbers are useful for sanity checking convergence and runtime:

| Checkpoint | Approx. elapsed training time | Overall success |
| ---: | ---: | ---: |
| 20k | ~13h | 84.55% |
| 30k | ~19.5h | 93.95% |
| 40k | ~26h | 98.00% |
| 49k | ~32h | 98.30% |

Treat these as reference numbers, not hard guarantees. Runtime depends on GPU
type, storage throughput, and dataloader health.

## Prepare Inputs

1. Install the repo:

```bash
conda create -n opendm python=3.10
conda activate opendm
pip install -e .
```

2. Download or mount a DM05 checkpoint and pass it through
`--model-config.model-name-or-path`.

3. Download LIBERO with `script/libero_runner.sh dataset` or update
`opendm/dataset/libero.py` to point at your mounted path.

4. Confirm that the dataset name, action mode, and chunk size match the
   training recipe. If the matching norm stats file does not exist, training
   will compute it automatically under `./norm_stats/`.

## Local Command

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero_lora.py \
  --task train \
  --nproc_per_node 8 \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 50000
```

`playground/dm05_libero_lora.py` already provides the LIBERO LoRA defaults,
including dataset name, action mode, chunk size, attention settings, learning
rate, warmup steps, batch size, save interval, and total training steps.
Override these options only when you need to change the reference recipe.

For a 1-GPU smoke test, reduce `--nproc_per_node`,
`--trainer-config.num-train-steps`, and `--trainer-config.save-steps`. Do not
use smoke-test quality to judge the final recipe.

## What Gets Trained

`DM05LoraConfig` in `opendm/model/dm05/dm05_lora.py` wraps every supported
linear layer with LoRA and saves selected DM05 action modules densely. The
`dm05_time_modulators` alias expands to all action-expert input, MLP, and final
time modulators.

The trainable summary is written to
`user_checkpoints/dm05_sft/trainable_summaries/dm05_lora_libero_pi0_all.json`
by default. Check it before trusting a run:

- `r` should be `32`.
- `lora_alpha` should be `16`.
- `target_modules` should resolve from `all-linear`.
- `unexpected_trainable_parameters` should be empty.
- The dense saved modules should include action projections, time MLPs, and
  expanded time modulators.

## Checkpoints

Step checkpoints are the canonical artifacts for multi-GPU LoRA/FSDP training. See the [DM05 Inference Guide](dm05_inference.md) for adapter loading behavior and the complete LIBERO LoRA service command.
