# DM05 SO101 LoRA 训练

本文档是使用 `playground/dm05_so101_lora.py` 在 SO101 数据上运行 DM05 LoRA SFT 的开发者参考。

## 何时使用 LoRA

当你希望在不更新完整模型参数的情况下将 DM05 适配到 SO101 时，可以使用 LoRA。该方法可以高效地针对 SO101 拾取立方体任务进行微调，同时保留基础模型的通用能力。

| 项目 | 配置值 |
| --- | --- |
| 硬件 | 8x GPU (可配置) |
| 单卡 batch | 8 |
| 梯度累积 | 1 |
| 训练步数 | 10,000 (默认) |
| 保存间隔 | 1,000 steps |
| 优化器 | AdamW，优化 LoRA 和可训练权重 |
| LR / warmup | `1e-4` / `1000` |
| LoRA | 启用 (`use_lora=True`) |
| 目标模块 | `all-linear` |
| Attention | LLM `flex_attention`，vision `flash_attention_2`，action `sdpa` |
| Gradient checkpointing | VLM GC 开启，AE GC 开启 |

## SO101 数据

SO101 训练目标是 `so101_pick_cube`，注册位置为 `opendm/dataset/so101.py`。

| 字段 | 值 |
| --- | --- |
| 数据集名称 | `so101_pick_cube` |
| 数据根目录 | `./data/so101_pick_cube` |
| JSONL 根目录 | `./data/so101_pick_cube/jsonl` |
| 图像根目录 | `./data/so101_pick_cube/image` |
| Norm stats | 自动生成到 `./norm_stats/` |
| 图像键 | `images_1`, `images_2` |
| Action mode | `relative` |
| State | 包含 (`add_state=True`) |
| Action dim | 6 |
| Chunk size | 50 |

使用 `script/so101_runner.sh` 下载并组织 SO101 数据：

```bash
pip install -U huggingface_hub
script/so101_runner.sh dataset
```

该脚本从 Hugging Face 下载 `Dexmal/so101_pick_cube` 到 `./data/.hf_downloads/so101_pick_cube`，根据需要解压归档文件，并在 `./data/so101_pick_cube` 下组织最终数据集。

如需使用不同的数据位置：

```bash
script/so101_runner.sh dataset --data-root /path/to/data/so101_pick_cube
```

下载后，`so101_pick_cube` 应具有如下结构：

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

每一行 JSONL 表示一帧：

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

当匹配的 norm stats 文件不存在时，训练会自动计算归一化参数。当 action/state 分布、action mode 或 action chunk 长度发生变化时，需要重新计算。

## 准备输入

1. 安装仓库：

```bash
conda create -n opendm python=3.10
conda activate opendm
pip install -e .
```

2. 下载或挂载 DM05 checkpoints：
   - 训练时：`./checkpoints/DM05` - 基础 DM05 模型
   - 推理时：需要 `./checkpoints/DM05` 和 `./checkpoints/DM05-SO101-Pick-Cube` 两个 checkpoint

3. 确保 SO101 数据集位于 `./data/so101_pick_cube`，或更新 `opendm/dataset/so101.py` 指向你的数据位置。

4. 确认数据集名称、action mode 和 chunk size 与训练配置一致。如果匹配的 norm stats 文件不存在，训练会自动在 `./norm_stats/` 下计算生成。

## 本地命令

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_so101_lora.py \
  --task train \
  --nproc_per_node 8 \
  --model-config.model-name-or-path ./checkpoints/DM05
```

`playground/dm05_so101_lora.py` 已经提供 SO101 LoRA 默认值，包括数据集名称、action mode、chunk size、attention 设置、学习率、warmup steps、batch size、保存间隔和总训练步数。只有在需要修改参考配置时才覆盖这些选项。

## 训练哪些参数

LoRA 配置会用 LoRA 包装每个受支持的 linear layer，并密集保存选定的 DM05 action 模块。`dm05_time_modulators` 别名会展开为所有 action-expert 输入、MLP 和 final time modulators。

默认情况下，可训练参数摘要会写入 `user_checkpoints/<output_dir>/trainable_summaries/dm05_lora_so101_pick_cube.json`。信任一次训练前，请先检查该文件：

- `target_modules` 应由 `all-linear` 解析得到。
- `unexpected_trainable_parameters` 应为空。
- 密集保存模块应包含 action projections、time MLPs 和 time modulators。

## Checkpoints 和推理

Step checkpoints 是多卡 LoRA/FSDP 训练的标准产物。推理时，将 LoRA checkpoint 路径作为 `--model-config.model-name-or-path` 传入；loader 会读取 `adapter_config.json`，加载记录的 base model，并合并 adapter 用于推理。

使用你的训练运行产出的 checkpoint 路径。例如，如果训练输出目录是 `${TRAINING_OUTPUT_DIR}`，可以使用 `${TRAINING_OUTPUT_DIR}/checkpoint-4000` 这样的 checkpoint。

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_so101_lora.py \
  --task inference \
  --nproc_per_node 1 \
  --model-config.model-name-or-path ./checkpoints/DM05-SO101-Pick-Cube \
  --inference-config.output-action-dim 6
```
