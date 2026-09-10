# DM05-MEM SFT and Validation Guide

This guide shows how to run DM05-MEM SFT with `playground/dm05_mem_sft_demo.py`. Verify training, saving, and inference on the bundled demo data. This entry initializes from `DM05-MEM` and enables memory history. Inference defaults are the demo / DOS-W1 layout; swapping only the dataset name does not make this a generic robot entry. See section 5 for the supported scope.

## 1. Prepare the Base Model

Run from the OpenDM repository root:

```bash
huggingface-cli download Dexmal/DM05-MEM --local-dir ./checkpoints/DM05-MEM
```

## 2. Understand the MEM Demo SFT Entry

OpenDM provides `playground/dm05_mem_sft_demo.py` as a ready-to-run MEM SFT entry. It is launched through `script/dm05_launcher.sh`. Training defaults match demo SFT, with history enabled:

- `dataset_name`: `demo_mem`
- `is_history`: `true` (the playground requests 32 main-view history slots at 1 FPS)
- `image_keys`: `images_1`, `images_2`, `images_3`
- `output_action_dim`: `14`
- `base_lr`: `2.5e-5`
- `per_device_train_batch_size`: `8`
- `num_train_steps`: `50000`
- `chunk_size`: `50`
- `model_max_length`: `2048`

`demo_mem` uses the same `assets/demo` JSONL and images as `demo`, and adds `fps: 25` in `opendm/dataset/demo.py`. Mem SFT needs a registered `fps` to sample history.

The bundled `episode0.jsonl` has 300 frames (15 image files reused). At 25 FPS, a 1 FPS history slot looks back 25 frames, so later demo samples get **10 non-empty history images**. Filling all 32 slots would need about 800 frames. The 32-slot setting is for real MEM data.

```text
assets/demo/
├── episode0.jsonl
├── index_cache.json
└── images/episode0/...
```

## 3. Check the Data Format

The frame format matches the [DM05 SFT and Validation Guide](dm05_finetuning.md). See the [data guide](data.md) for fields. MEM additionally requires:

- The dataset registration must include `fps`.
- The JSONL does not need `history_images`. Training samples past frames from the main-view `image_keys[0]` at 1 FPS.

## 4. Run DM05-MEM SFT on Demo Data

Use `script/dm05_launcher.sh` with `--exp playground/dm05_mem_sft_demo.py`. Adjust the GPU count and output directory as needed:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_demo.py \
  --nproc_per_node 8 \
  --task train \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.output-dir ./user_checkpoints/dm05_mem_sft_demo
```

If matching normalization statistics are missing, OpenDM computes them from the current experiment data and saves them under `./norm_stats/`. Saved checkpoints copy the file to `norm_stats.json`. Inference must use the same statistics.

## 5. Use Your Own Data

This playground's inference defaults remain the demo / DOS-W1 layout: `robot_type=DOS W1`, `speed=0.1`, `control_mode=joint`, 14-d actions, chunk 50, 32 history slots, plus `compose_eef_rot`, `full_action_mask`, and `use_transformed_state`. Overriding `--data-config.dataset-name` does not change those request defaults or action transforms. For RoboDojo-Sim `cover_blocks`, use [`playground/dm05_mem_sft_robodojo_cover_blocks.py`](dm05_robodojo.md). For other robots, copy this playground and change the defaults.

If the data layout matches the demo (three cameras, 14-d state, registered `fps`), you can override the dataset name:

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_demo.py \
  --nproc_per_node 8 \
  --task train \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.output-dir ./user_checkpoints/dm05_mem_sft_my_robot
```

Still check `dataset_name`, `fps`, `image_keys`, `state_desc`, `output_action_dim`, `chunk_size`, learning rate, and step count. Training and inference must use the same `chunk_size` and history settings.

## 6. Inference

Start the service from a checkpoint this entry just saved. Replace the path with the step directory you actually wrote. See the [DM05 Inference Guide](dm05_inference.md) for the HTTP API and fast backend. The released `Dexmal/DM05-MEM` checkpoint uses the same entry; set `model-name-or-path` to `./checkpoints/DM05-MEM`.

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_demo.py \
  --task inference \
  --model-config.model-name-or-path ./user_checkpoints/dm05_mem_sft_demo/checkpoint-10000 \
  --inference-config.port 7891
```

This entry defaults `observation.robot_type` to `DOS W1`, `speed` to `"0.1"`, and `control_mode` to `joint`. The playground already enables `is_history`. Example request with `history_images`:

```bash
bash tests/curl_history.sh http://127.0.0.1:7891/v1/infer
```

## Checklist

- Launch with `script/dm05_launcher.sh --exp playground/dm05_mem_sft_demo.py`.
- The dataset is registered with `fps`, and is passed through `--data-config.dataset-name`.
- JSONL image fields match `image_keys`.
- Training and inference use the same `chunk_size` and history length.
- Inference uses a step directory this entry saved (or released `Dexmal/DM05-MEM`), and the checkpoint contains `norm_stats.json`.
