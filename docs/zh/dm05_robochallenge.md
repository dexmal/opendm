# DM05 RoboChallenge Table 30 v2 推理指南

本文档说明如何使用 `third_party/robochallenge_inference` 中提供的 DM05
RoboChallenge Table 30 v2 推理客户端。

## 概览

RoboChallenge 推理客户端会连接 RoboChallenge 平台，从 submission 中选择
当前可执行的 job，拉取机器人观测，调用 DM05 policy 推理，并把动作提交回平台。

当前支持四种 Table 30 v2 机器人配置：

- `arx5`
- `ur5`
- `aloha`
- `w1`

## 文件结构

```text
third_party/robochallenge_inference/
├── configs/
│   ├── default.yaml
│   └── generalist/      # arx5 / ur5 / aloha / w1 配置
├── policies/            # 基于 OpenDM 的 DM05 policy 和输出处理
├── robot/               # RoboChallenge HTTP 客户端和 job 循环
├── runner/              # policy 调用和 debug 保存
├── utils/               # 任务元信息、坐标变换、日志和工具函数
├── execute.py
└── requirements.txt
```

## 环境准备

先安装 OpenDM。fast 推理需要安装 `fast-infer` 可选依赖：

```bash
# 在 OpenDM 仓库根目录运行。
pip install -e ".[fast-infer]"

cd third_party/robochallenge_inference
pip install -r requirements.txt
```

推理前需要设置模型路径：

```bash
export OPENDM_ROOT=/path/to/opendm

export ARX5_CHECKPOINT=/path/to/arx5/checkpoint
export ARX5_NORM_STATS=/path/to/arx5/norm_stats.json

export UR5_CHECKPOINT=/path/to/ur5/checkpoint
export UR5_NORM_STATS=/path/to/ur5/norm_stats.json

export ALOHA_CHECKPOINT=/path/to/aloha/checkpoint
export ALOHA_NORM_STATS=/path/to/aloha/norm_stats.json

export W1_CHECKPOINT=/path/to/w1/checkpoint
export W1_NORM_STATS=/path/to/w1/norm_stats.json
```

如果没有设置 `*_NORM_STATS`，客户端会默认读取
`CHECKPOINT/norm_stats.json`。

## 启动推理

在 `third_party/robochallenge_inference` 目录下运行：

```bash
cd third_party/robochallenge_inference

python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID
```

根据 run 对应的机型选择配置：

```bash
python execute.py --config-name generalist/arx5  user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
python execute.py --config-name generalist/ur5   user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
python execute.py --config-name generalist/aloha user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
python execute.py --config-name generalist/w1    user_id=YOUR_USER_ID submission_id=YOUR_SUBMISSION_ID run_id=YOUR_RUN_ID
```

`run_id` 是可选参数。不传时，worker 会轮询整个 submission，并选择和当前
机器人类型匹配的 active job。

## 启动参数

启动入口使用 Hydra override。

| 参数 | 是否必需 | 作用 |
| --- | --- | --- |
| `--config-name generalist/<robot>` | 是 | 选择 `arx5`、`ur5`、`aloha` 或 `w1` |
| `user_id=...` | 是 | RoboChallenge 平台请求使用的用户 id |
| `submission_id=...` | 是 | 包含 run collection 的 submission id |
| `run_id=...` | 否 | 限定只执行 submission 中的某一个 run |
| `checkpoint=...` | 否 | 覆盖环境变量中的 checkpoint 路径 |
| `norm_stats=...` | 否 | 覆盖归一化参数路径 |
| `action_horizon=...` | 否 | 覆盖 action horizon |
| `action_playback_target_steps=...` | 否 | 将模型输出动作均匀采样到指定步数 |
| `debug=true` | 否 | 启用逐步 debug 数据保存 |
| `debug_image_limit=...` | 否 | debug 开启时保存的平台图片快照数量；负数表示保存全部 |
| `log_dir=...` | 否 | runtime 日志和可选 debug 数据保存目录 |
| `hydra.run.dir=...` | 否 | Hydra 输出目录 |

默认关闭 debug 数据保存，避免长任务产生过大的日志目录。`runtime.log`
仍会写入 `log_dir`。如果需要保存逐步复现数据和有限数量的平台图片快照，可以传：

```bash
python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID \
  debug=true \
  debug_image_limit=20
```

只有需要完整图片采集时才建议使用 `debug_image_limit=-1`。

## Fast Backend 默认配置

fast 推理默认值配置在 `configs/default.yaml` 的
`robot_profiles.<robot>.runtime_args` 下。

默认 TensorRT engine 路径：

| 机型 | 默认 engine 路径 |
| --- | --- |
| ARX5 | `checkpoints/trt_engines/dm05_arx5_h8.engine` |
| UR5 | `checkpoints/trt_engines/dm05_ur5_h2.engine` |
| ALOHA | `checkpoints/trt_engines/dm05_aloha_h3.engine` |
| W1 | `checkpoints/trt_engines/dm05_w1_h3.engine` |

后缀表示 TensorRT vision engine 的图片数量：

- `h8`：ARX5 使用 3 张当前图像和 5 个历史图像 slot。
- `h2`：UR5 使用 2 张当前图像。
- `h3`：ALOHA 和 W1 使用 3 张当前图像。

如果 engine 文件不存在，OpenDM 会在第一次 fast backend 启动时自动构建。
如果需要强制重新构建：

```bash
python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID \
  +runtime_args.force_rebuild_trt=true
```

如果希望单次运行关闭 fast backend：

```bash
python execute.py \
  --config-name generalist/arx5 \
  user_id=YOUR_USER_ID \
  submission_id=YOUR_SUBMISSION_ID \
  run_id=YOUR_RUN_ID \
  runtime_args.backend=default
```

## Runtime 默认值

机型级默认值在 `configs/default.yaml` 中配置。

- ARX5 使用 logical-step history，`action_horizon=50`，
  `action_playback_target_steps=25`。
- UR5 使用 `action_horizon=25`，不启用 playback sampling。
- ALOHA 和 W1 默认使用 `action_horizon=25`。
- 单任务覆盖配置在 `configs/generalist/*.yaml` 中。
