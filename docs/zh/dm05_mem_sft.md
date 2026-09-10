# DM05-MEM SFT 与验证指南

本文档说明如何用 `playground/dm05_mem_sft_demo.py` 跑通 DM05-MEM 的 SFT。先用内置 demo 数据验证训练、保存、推理。本入口从 `DM05-MEM` 初始化，并打开 mem 历史。推理默认绑定 demo / DOS-W1 形态，不能只换数据集名称就当成任意机型入口；范围见第 5 节。

## 1. 准备基础模型

在 OpenDM 仓库根目录运行：

```bash
huggingface-cli download Dexmal/DM05-MEM --local-dir ./checkpoints/DM05-MEM
```

## 2. 理解 MEM Demo SFT 入口

OpenDM 提供了 `playground/dm05_mem_sft_demo.py` 作为可直接运行的 MEM SFT 入口。它通过 `script/dm05_launcher.sh` 启动，并预设了与 demo SFT 接近的训练配置，同时打开历史：

- `dataset_name`：`demo_mem`
- `is_history`：`true`（playground 按 1 FPS 请求 32 个主视角历史槽）
- `image_keys`：`images_1`、`images_2`、`images_3`
- `output_action_dim`：`14`
- `base_lr`：`2.5e-5`
- `per_device_train_batch_size`：`8`
- `num_train_steps`：`50000`
- `chunk_size`：`50`
- `model_max_length`：`2048`

`demo_mem` 与 `demo` 使用同一套 `assets/demo` JSONL 和图片，并在 `opendm/dataset/demo.py` 中额外注册了 `fps: 25`。mem SFT 需要数据集 `fps` 才能采样历史帧。

内置 `episode0.jsonl` 有 300 帧（15 张图片重复引用）。数据集 `fps` 为 25 时，1 FPS 的一个历史槽会回看 25 帧，因此后段 demo 样本会有 **10 张非空历史**。填满 32 槽大约需要 800 帧。32 槽是给真实 MEM 数据用的配置。

```text
assets/demo/
├── episode0.jsonl
├── index_cache.json
└── images/episode0/...
```

## 3. 确认数据格式

帧格式与 [DM05 SFT 与验证指南](dm05_finetuning.md) 相同，详见 [数据指南](data.md)。MEM 额外要求：

- 数据集注册必须提供 `fps`。
- JSONL 不需要 `history_images`。训练时从主视角 `image_keys[0]` 按 1 FPS 回看过去帧。

## 4. 使用 Demo 数据运行 DM05-MEM SFT

使用 `script/dm05_launcher.sh` 启动，并通过 `--exp playground/dm05_mem_sft_demo.py` 指定入口。可按需调整 GPU 数量和输出目录：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_demo.py \
  --nproc_per_node 8 \
  --task train \
  --model-config.model-name-or-path ./checkpoints/DM05-MEM \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.output-dir ./user_checkpoints/dm05_mem_sft_demo
```

训练时如果匹配的归一化统计不存在，OpenDM 会从当前实验数据计算并保存到 `./norm_stats/`。保存 checkpoint 时会复制为 checkpoint 目录下的 `norm_stats.json`。推理必须使用同一份统计。

## 5. 替换为自己的数据

本 playground 的推理默认仍是 demo / DOS-W1 形态：`robot_type=DOS W1`、`speed=0.1`、`control_mode=joint`、14 维、chunk 50、32 个历史槽，以及 `compose_eef_rot`、`full_action_mask`、`use_transformed_state`。只改 `--data-config.dataset-name` 不会改这些动作转换和请求默认值。RoboDojo-Sim `cover_blocks` 请用 [`playground/dm05_mem_sft_robodojo_cover_blocks.py`](dm05_robodojo.md)。其它机型应复制本 playground 再改默认值。

若数据布局与 demo 相同（三相机、14 维、注册了 `fps`），可以覆盖数据集名称：

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

仍需核对：`dataset_name`、`fps`、`image_keys`、`state_desc`、`output_action_dim`、`chunk_size`、学习率和步数。训练和推理必须使用相同的 `chunk_size` 和历史设置。

## 6. 推理

用本入口刚保存的 checkpoint 起服务，将路径换成实际 step 目录。HTTP API 和 fast backend 见 [DM05 推理指南](dm05_inference.md)。发布版 `Dexmal/DM05-MEM` 也用本入口，把 `model-name-or-path` 换成 `./checkpoints/DM05-MEM`。

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_mem_sft_demo.py \
  --task inference \
  --model-config.model-name-or-path ./user_checkpoints/dm05_mem_sft_demo/checkpoint-10000 \
  --inference-config.port 7891
```

本入口默认 `observation.robot_type` 为 `DOS W1`、`speed` 为 `"0.1"`、`control_mode` 为 `joint`。playground 已打开 `is_history`。带 `history_images` 的请求示例：

```bash
bash tests/curl_history.sh http://127.0.0.1:7891/v1/infer
```

## 检查清单

- 使用 `script/dm05_launcher.sh --exp playground/dm05_mem_sft_demo.py` 启动。
- 数据集已注册且包含 `fps`，并通过 `--data-config.dataset-name` 传入。
- JSONL 中的图像字段和 `image_keys` 一致。
- 训练和推理使用相同的 `chunk_size` 和历史长度。
- 推理使用本入口保存的 step 目录（或发布版 `Dexmal/DM05-MEM`），checkpoint 中存在 `norm_stats.json`。
