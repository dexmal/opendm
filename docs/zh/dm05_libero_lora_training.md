# DM05 LIBERO LoRA 训练

本文档是使用 `playground/dm05_libero_lora.py` 在 LIBERO 数据上运行 DM05 LoRA SFT 的开发者参考。

## 何时使用 LoRA

当你希望在不更新完整模型参数的情况下将 DM05 适配到 LIBERO 时，可以使用 LoRA。当前参考配置来自一次通过内部验证的训练流程：

| 项目 | 参考值 |
| --- | --- |
| 硬件 | 8x NVIDIA RTX 4090D |
| 全局 batch | 32 |
| 单卡 batch | 4 |
| 梯度累积 | 1 |
| 训练步数 | 50,000 |
| 保存间隔 | 10,000 steps |
| 优化器 | AdamW，优化 LoRA 和 `modules_to_save` 中的可训练权重 |
| LR / warmup | `5e-4` / `500` |
| LoRA rank / alpha / dropout | `32` / `16` / `0.0` |
| 目标模块 | `all-linear` |
| 需要密集保存的模块 | `action_in_proj`, `action_out_proj`, `time_mlp_in`, `time_mlp_out`, `dm05_time_modulators` |
| Attention | LLM `eager`，vision `sdpa`，action `sdpa` |
| Gradient checkpointing | VLM GC 关闭，AE GC 关闭 |

该验证训练完成了完整训练计划，49k checkpoint 的 LIBERO 总体成功率最好，为 `98.30%`。其预训练 checkpoint 与具体实验相关；请使用与你的数据集和目标机器人匹配的 checkpoint。

## LIBERO 数据

内置的 LIBERO 训练目标是 `libero_pi0_all`，注册位置为 `opendm/dataset/libero.py`。

| 字段 | 值 |
| --- | --- |
| 数据集名称 | `libero_pi0_all` |
| JSONL 根目录 | `./data/libero/libero_pi0_all` |
| 图像根目录 | `./data/libero/libero_pi0_all/image` |
| Norm stats | 自动生成到 `./norm_stats/` |
| 图像键 | `images_1`, `images_2` |
| Action mode | `absolute` |
| State desc | 6 个关节维度 + 2 个夹爪维度 |
| Action dim | 7 |

使用 `script/libero_runner.sh` 下载并整理 LIBERO 数据：

```bash
pip install -U huggingface_hub
script/libero_runner.sh dataset
```

该脚本会从 Hugging Face 下载 `Dexmal/libero` 到 `./data/.hf_downloads/libero`，按需解压分片归档，并将最终数据整理到 `./data/libero`。

如需使用其他数据位置：

```bash
script/libero_runner.sh dataset --data-root /path/to/data/libero
```

`train` 和 `all` 命令会将 `--data-root` 传递给默认的 `libero_pi0_all` 训练配置。
直接使用启动脚本时传入：

```bash
--data-config.jsonl-dir /path/to/data/libero/libero_pi0_all \
--data-config.image-dir /path/to/data/libero/libero_pi0_all/image
```

下载完成后，`libero_pi0_all` 应该具有如下结构：

```text
data/libero/libero_pi0_all/
  jsonl/
    <episode>.jsonl
  image/
    ...
```

每一行 JSONL 表示一帧。对于 `action_mode="absolute"`，每一行都必须包含当前 action，因为 collator 会将 `[t, t+1, ..., t+chunk-1]` 堆叠成训练目标。

```json
{
  "images_1": {"type": "image", "url": "./episode_000/camera_0/000000.jpg"},
  "images_2": {"type": "image", "url": "./episode_000/camera_1/000000.jpg"},
  "state": [0.12, -0.04, 0.31, 1.22, -0.18, 0.44, 1.0, 0.0],
  "action": [0.13, -0.03, 0.30, 1.20, -0.17, 0.45, 1.0],
  "prompt": "put the object into the bowl"
}
```

当匹配的 norm stats 文件不存在时，训练会自动计算归一化参数。当 action/state 分布、action mode 或 action chunk 长度发生变化时，需要重新计算。

## 参考里程碑

参考验证训练使用上面的配置在 8x NVIDIA RTX 4090D 上运行，并对标准任务集中的每个 checkpoint 执行 2,000 个 LIBERO episodes 评测。这些数字可用于 sanity check 收敛和运行耗时：

| Checkpoint | 近似训练耗时 | 总体成功率 |
| ---: | ---: | ---: |
| 20k | ~13h | 84.55% |
| 30k | ~19.5h | 93.95% |
| 40k | ~26h | 98.00% |
| 49k | ~32h | 98.30% |

这些数值是参考结果，不是强保证。实际耗时取决于 GPU 类型、存储吞吐和 dataloader 状态。

## 准备输入

1. 安装仓库：

```bash
conda create -n opendm python=3.10
conda activate opendm
pip install -e .
```

2. 下载或挂载一个 DM05 checkpoint，并通过 `--model-config.model-name-or-path` 传入。

3. 使用 `script/libero_runner.sh dataset` 下载 LIBERO，或更新 `opendm/dataset/libero.py` 指向你的挂载路径。

4. 确认数据集名称、action mode 和 chunk size 与训练配置一致。如果匹配的 norm stats 文件不存在，训练会自动在 `./norm_stats/` 下计算生成。

## 本地命令

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero_lora.py \
  --task train \
  --nproc_per_node 8 \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 50000
```

`playground/dm05_libero_lora.py` 已经提供 LIBERO LoRA 默认值，包括数据集名称、action mode、chunk size、attention 设置、学习率、warmup steps、batch size、保存间隔和总训练步数。只有在需要修改参考配置时才覆盖这些选项。

如需进行 1 卡 smoke test，请降低 `--nproc_per_node`、`--trainer-config.num-train-steps` 和 `--trainer-config.save-steps`。不要用 smoke-test 质量判断最终训练配置。

## 训练哪些参数

`opendm/model/dm05/dm05_lora.py` 中的 `DM05LoraConfig` 会用 LoRA 包装每个受支持的 linear layer，并密集保存选定的 DM05 action 模块。`dm05_time_modulators` 别名会展开为所有 action-expert 输入、MLP 和 final time modulators。

默认情况下，可训练参数摘要会写入 `user_checkpoints/dm05_sft/trainable_summaries/dm05_lora_libero_pi0_all.json`。信任一次训练前，请先检查该文件：

- `r` 应为 `32`。
- `lora_alpha` 应为 `16`。
- `target_modules` 应由 `all-linear` 解析得到。
- `unexpected_trainable_parameters` 应为空。
- 密集保存模块应包含 action projections、time MLPs 和展开后的 time modulators。

## Checkpoints

Step checkpoints 是多卡 LoRA/FSDP 训练的标准产物。Adapter 加载方式和完整的 LIBERO LoRA 服务命令统一参考 [DM05 推理指南](dm05_inference.md)。
