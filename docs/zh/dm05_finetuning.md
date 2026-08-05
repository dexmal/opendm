# DM05 SFT 与验证指南

本文档说明如何在 OpenDM 中跑通完整的 DM05 SFT 流程。建议先使用内置 `assets/demo` 数据和 `playground/dm05_sft_demo.py` 验证训练、保存、推理链路，再替换成自己的机器人数据。

## 1. 准备基础模型

在 OpenDM 仓库根目录运行：

```bash
huggingface-cli download Dexmal/DM05 --local-dir ./checkpoints/DM05
```

## 2. 理解 Demo SFT 入口

OpenDM 提供了 `playground/dm05_sft_demo.py` 作为可直接运行的 SFT 入口。它通过 `script/dm05_launcher.sh` 启动，并预设了 demo 训练配置：

- `dataset_name`：`demo`
- `image_keys`：`images_1`、`images_2`、`images_3`
- `output_action_dim`：`14`
- `base_lr`：`2.5e-5`
- `per_device_train_batch_size`：`8`
- `num_train_steps`：`50000`
- `chunk_size`：继承 DM05 默认值 `50`

内置 demo 数据集已在 `opendm/dataset/demo.py` 中注册，结构如下：

```text
assets/demo/
├── episode0.jsonl
├── index_cache.json
└── images/episode0/...
```

## 3. 确认数据格式

OpenDM 使用 JSONL 读取机器人演示数据，每一行表示一帧：

```json
{
  "images_1": {"type": "image", "url": "./images/episode0/cam_high/0.jpg"},
  "images_2": {"type": "image", "url": "./images/episode0/cam_left_wrist/0.jpg"},
  "images_3": {"type": "image", "url": "./images/episode0/cam_right_wrist/0.jpg"},
  "state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
  "prompt": "Pick up the object",
  "is_robot": true
}
```

字段说明：

- `images_1`、`images_2`、`images_3` 需要和数据集注册文件中的 `image_keys` 一致。
- `url` 是相对于注册文件中 `image_dir` 的路径。
- `state` 是当前机器人状态，维度顺序需要和 `state_desc` 一致。
- `action` 是可选字段。如果存在，OpenDM 会用 `action` 构造训练目标；如果不存在，则使用未来帧的 `state` 构造目标。
- `prompt` 是任务指令。
- `index_cache.json` 是可选文件。如果不存在，OpenDM 会扫描 JSONL 并自动生成。

默认 `action_mode` 是 `relative`，因此 action 和 state 的维度必须一致。gripper 维度会根据 `state_desc` 保持绝对值。每个 episode 至少需要两帧。

## 4. 使用 Demo 数据运行 DM05 SFT

使用 `script/dm05_launcher.sh` 启动，并通过 `--exp playground/dm05_sft_demo.py` 指定 SFT 入口。

建议先跑一个短训练，确认流程正常：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --nproc_per_node 1 \
  --task train \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --data-config.compute-norm-stats-max-batches 1 \
  --trainer-config.num-train-steps 10 \
  --trainer-config.save-steps 10 \
  --trainer-config.output-dir ./user_checkpoints/dm05_sft_demo_smoke
```

这个 smoke run 只缩短训练步数和保存间隔，不修改 `chunk_size`，因此会沿用 DM05 默认的 `50`。只有当你的数据和控制策略需要不同的 action horizon 时，才建议显式覆盖 `--model-config.chunk-size`，并确保训练和推理使用同一个值。

正常 SFT 训练可以保留 demo 入口默认值，只按需调整 GPU 数量和输出目录：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --nproc_per_node 8 \
  --task train \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.output-dir ./user_checkpoints/dm05_sft_demo
```

训练时，OpenDM 需要使用 `state` 和 `action` 的归一化统计。如果匹配的文件不存在，OpenDM 会从当前实验数据中自动计算，并保存到 `./norm_stats/`。同一实验中相同 `robot_type` 的数据共享一组统计值，不同机型写入同一文件的 `norm_stats_by_robot`。保存 checkpoint 时会把完整文件复制为 checkpoint 目录下的 `norm_stats.json`。

这很重要，因为推理时也要用同一份 norm stats 把输入 `state` 归一化，并把模型输出的 `action` 反归一化。

## 5. 替换为自己的数据

demo SFT 流程跑通后，可以注册自己的数据集。例如创建 `opendm/dataset/my_robot.py`：

```python
from opendm.constants.robot import RobotStateDesc
from opendm.dataset.register import register_dataset

MY_ROBOT_STATE_DESC = (
    [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
    + [RobotStateDesc.JOINT] * 6
    + [RobotStateDesc.GRIPPER]
)

register_dataset(
    {
        "my_robot": {
            "jsonl_dir": "./assets/my_robot/",
            "image_dir": "./assets/my_robot/",
            "image_keys": ["images_1", "images_2", "images_3"],
            "image_prompts": ["Head", "Left wrist", "Right wrist"],
            "state_desc": MY_ROBOT_STATE_DESC,
        },
    }
)
```

然后继续使用同一个 SFT 入口，只覆盖数据集名称：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --nproc_per_node 8 \
  --task train \
  --data-config.dataset-name my_robot \
  --model-config.model-name-or-path ./checkpoints/DM05 \
  --trainer-config.num-train-steps 50000 \
  --trainer-config.output-dir ./user_checkpoints/dm05_my_robot
```

`playground/dm05_sft_demo.py` 可以作为 SFT 模板，但用户数据通常需要检查并调整以下配置：

- `dataset_name`：改成自己注册的数据集名称。
- `image_keys`：和 JSONL 中的图像字段一致。
- `image_prompts`：与加载图像顺序对应的 prompt 标签。
- `state_desc`：和 `state` / `action` 的维度顺序一致。
- `output_action_dim`：和模型输出 action 的最后一维一致。
- `chunk_size`：根据实际控制 horizon 设置；如果不确定，先保持默认 `50`。
- `base_lr`、batch size、训练步数：根据数据规模和 GPU 显存调整。

需要确保 `image_keys`、`state_desc`、action 维度和 `chunk_size` 在训练和推理阶段保持一致。

## 6. 推理

自定义 SFT checkpoint 的服务命令、服务验证、HTTP API 和 fast backend 配置统一参考 [DM05 推理指南](dm05_inference.md)。

## 检查清单

- 使用 `script/dm05_launcher.sh --exp playground/dm05_sft_demo.py` 启动 SFT。
- 数据集名称已注册，并通过 `--data-config.dataset-name` 传入。
- JSONL 中的图像字段和 `image_keys` 一致。
- `state_desc` 长度和 state/action 维度一致。
- 训练和推理使用相同的 `chunk_size`。
- 推理 checkpoint 中存在 `norm_stats.json`，或者 `./norm_stats/` 下存在同一数据集和 `chunk_size` 对应的统计文件。
