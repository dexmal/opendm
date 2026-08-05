# OpenDM 数据格式与数据集注册

OpenDM 使用 JSON Lines（JSONL）格式的机器人 episode 训练 DM05。本文档说明
当前 OpenDM 数据流水线支持的帧格式，以及如何注册数据集并通过
`--data-config.dataset-name` 使用。

## 1. 使用项目提供的数据（Using Provided Data）

OpenDM 已注册以下两个常用数据集。表中的路径均相对于 OpenDM 仓库根目录。

| 数据集 | 下载地址 | 注册名称 | 注册的 `jsonl_dir` | 注册的 `image_dir` |
| --- | --- | --- | --- | --- |
| LIBERO | [Dexmal/libero](https://huggingface.co/datasets/Dexmal/libero) | `libero_pi0_all` | `./data/libero/libero_pi0_all` | `./data/libero/libero_pi0_all/image` |
| RoboTwin 2.0 | [Dexmal/robotwin2-full](https://huggingface.co/datasets/Dexmal/robotwin2-full) | `robotwin2_generalist` | `./data/robotwin2.0` | `./data/robotwin2.0/video` |

在仓库根目录运行以下命令，即可下载并整理 LIBERO 数据：

```bash
script/libero_runner.sh dataset
```

LIBERO 的 episode 文件位于其注册 `jsonl_dir` 下的 `jsonl/` 子目录。RoboTwin
2.0 同样将 episode 文件放在 `jsonl/` 子目录中，下载和解压方法参考
[RoboTwin 2.0 指南](dm05_robotwin2.md)；完整 LIBERO 流程参考
[LIBERO 指南](dm05_libero.md)。

这些位置与 `opendm/dataset/libero.py` 和 `opendm/dataset/robotwin2.py` 中的
注册配置一致。如果数据位于其他位置，训练时可传入 `--data-config.jsonl-dir`
和 `--data-config.image-dir`。

## 2. 数据集目录结构

每个 episode 使用一个独立的 `.jsonl` 文件。媒体文件可以位于同一数据根目录
下，也可以放在通过 `image_dir` 配置的其他目录。

```text
assets/my_robot/
├── episode0.jsonl
├── episode1.jsonl
├── images/
│   ├── episode0/
│   └── episode1/
└── index_cache.json  # 文件不存在时由 OpenDM 生成
```

- OpenDM 会在 `jsonl_dir` 下递归扫描 `.jsonl` 文件。
- 每个 `.jsonl` 文件表示一个 episode。每行必须是表示一帧的完整 JSON 对象，
  不要插入空行。
- 每个 episode 至少需要两帧，因为 OpenDM 会根据当前帧和未来帧构造动作
  目标。
- `index_cache.json` 不存在时会在 `jsonl_dir` 下自动生成。如果生成缓存后
  增加、删除或修改了 JSONL 文件，需要删除旧缓存，让 OpenDM 重新生成。

## 3. 帧数据格式

使用图片的帧示例：

```json
{"images_1":{"type":"image","url":"./images/episode0/cam_high/0.jpg"},"images_2":{"type":"image","url":"./images/episode0/cam_left_wrist/0.jpg"},"images_3":{"type":"image","url":"./images/episode0/cam_right_wrist/0.jpg"},"state":[0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0],"prompt":"Pick up the object","is_robot":true}
```

使用视频的图像字段示例：

```json
{"type":"video","url":"./videos/episode0/cam_high.mp4","frame_idx":21}
```

第一个 JSON 对象是完整的帧记录；第二个只表示一个使用视频的图像字段值。

### 字段说明

| 字段 | 是否必需 | 说明 |
| --- | --- | --- |
| `images_*` | 是，注册到 `image_keys` 的每个字段都必需 | 相机输入元数据。`type` 只能是 `image` 或 `video`；`url` 相对于 `image_dir` 解析；视频还必须提供 `frame_idx`。 |
| `state` | 是 | 当前帧的机器人状态向量，维度和顺序必须与 `state_desc` 一致。 |
| `prompt` | 是 | 自然语言任务指令。 |
| `action` | 否 | 显式目标动作向量。如果省略，OpenDM 会使用未来帧的 `state` 构造目标。如果提供，该字段必须出现在同一 episode 的每一帧中。 |
| `is_robot` | 否 | Dexdata 风格机器人记录的兼容标记。OpenDM demo 数据将其设为 `true`；当前 DM05 训练流程不会用它构造动作目标。 |

OpenDM 只会按照注册配置中 `image_keys` 的顺序加载指定媒体字段。`state` 是每帧
必需字段，因为动作构造和归一化阶段都要求提供它。`add_state` 只控制是否将当前
`state` 经过归一化和离散化后，以 `States: ...` 文本加入模型的用户提示词；
设置 `add_state=False` 并不表示 JSONL 数据可以省略 `state`。

Dexdata 中的 `answer`、`conversations` 等对话字段不是必需项。

## 4. 动作目标与 Episode 边界

- 如果帧中包含 `action`，action chunk 从当前帧开始读取当前及未来的
  `action`。
- 如果没有 `action`，chunk 从下一帧开始，使用未来帧的 `state` 作为动作
  目标。
- 当 action chunk 超过 episode 末尾时，OpenDM 会重复最后一个可用值。
- 在 relative action 模式下，OpenDM 会从目标动作中减去当前 `state`，但
  `RobotStateDesc.GRIPPER` 标记的维度保持绝对值。因此 `state`、`action` 和
  `state_desc` 的维度必须一致。
- 在 absolute action 模式下，`action` 的维度可以与 `state` 不同，但必须与
  实验使用的 action schema、归一化参数和下游控制器一致。
- 通过实验入口中的 `data_config.action_mode` 选择 relative 或 absolute
  action（命令行参数为 `--data-config.action-mode`），并确保它与数据集及下游
  控制器一致。

归一化参数按实验计算。同一实验中，属于同一 `robot_type` 的多个数据源共同
计算一组 state/action 统计值；不同机型分别保存在同一个 `norm_stats.json` 的
`norm_stats_by_robot` 中。训练和推理都会按照样本或请求中的 `robot_type` 选择
对应统计值。历史上只包含顶层 `norm_stats` 的文件仍可直接使用；不需要归一化
state 的实验可以省略 profile 中的 `state`。

扩展文件通过 `default_robot_type` 声明默认机型，并在顶层 `norm_stats` 保留该
机型统计值供旧代码读取。多机型实验需通过 `norm_stats_default_robot_type` 指定
默认机型，单机型实验会自动确定。

## 5. 注册数据集

创建一个 Python 注册文件。OpenDM 会自动导入 `opendm/dataset/` 下文件名不以
`_` 开头的 `.py` 文件，但 `register.py` 除外。

```python
# opendm/dataset/my_robot.py

from opendm.constants.robot import RobotStateDesc, RobotType
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
            "robot_type": RobotType.ALOHA,
            "state_desc": MY_ROBOT_STATE_DESC,
        },
    },
)
```

### 注册字段

| 字段 | 说明 |
| --- | --- |
| 数据集键（`my_robot`） | 传给 `--data-config.dataset-name` 的名称。 |
| `jsonl_dir` | OpenDM 会在其中递归扫描 episode JSONL 文件的目录。 |
| `image_dir` | 解析媒体 `url` 时使用的基础目录。 |
| `image_keys` | 按顺序从每帧加载的相机字段。 |
| `image_prompts` | 必填。与 `image_keys` 一一对应的相机标签，写入 chat template（如 `Head`、`Left wrist`）。 |
| `state_desc` | state 每个维度的语义类型；在 relative action 模式下，它也用于标识保持绝对值的维度。支持 `RobotStateDesc.JOINT`、`RobotStateDesc.EEF` 和 `RobotStateDesc.GRIPPER`。 |
| `robot_type` | 机器人标签，通常使用 `RobotType` 中的值；它用于选择 state 描述和对应机型的归一化统计。历史无标签数据仍受支持。 |

路径可以是绝对路径，也可以相对于启动训练时的工作目录。仓库中的示例命令默认
从 OpenDM 仓库根目录运行。

在训练命令中传入注册的数据集名称：

```text
--data-config.dataset-name my_robot
```

注册一组数据集时可以使用前缀：

```python
register_dataset({"pick": {...}}, prefix="my_robot")
```

最终的数据集名称为 `my_robot_pick`；训练时应传入这个带前缀的名称。

## 6. 使用外部注册目录

如果不希望把自定义注册文件放进仓库，可以将它们放在同一个目录下，并在启动
训练前设置 `OPENDM_DATA_PATH`：

```bash
export OPENDM_DATA_PATH=/absolute/path/to/my_dataset_registry
```

`OPENDM_DATA_PATH` 指向数据集注册 Python 模块，而不是原始数据目录。请使用唯一的
模块名（例如 `custom_libero_paths.py`），避免与内置模块冲突。

该目录使用相同的文件名规则，每个文件都应按上面的方式调用
`register_dataset`。

## 检查清单

- 每个 `.jsonl` 文件表示一个 episode，每行表示一帧，不包含空行，并且每个
  episode 至少有两帧。
- 每一帧都包含注册到 `image_keys` 的全部字段；每个媒体 `url` 都能基于
  `image_dir` 解析，视频字段包含 `frame_idx`。
- `state_desc` 与 `state` 逐维对应且顺序一致。relative action 模式下，
  `action` 与 `state` 的维度和顺序也必须一致；absolute action 模式下，
  `action` 与实验使用的 action schema 一致。
- 显式 `action` 要么存在于一个 episode 的所有帧中，要么全部省略。
- 修改 JSONL 文件后删除过期的 `index_cache.json`。
- `--data-config.dataset-name` 与最终注册名称一致。
