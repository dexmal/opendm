# DM05 推理指南

本文档是 DM05 推理服务启动和 HTTP API 使用的统一指南。Benchmark 与训练文档只链接
到这里，不再重复维护推理配置。

[English](../en/dm05_inference.md)

## 1. 使用前准备

所有命令都在 OpenDM 仓库根目录运行。请准备：

- 按照主 README 安装完成的 OpenDM 环境。
- 与所选 playground 入口匹配的 DM05 checkpoint。
- checkpoint 目录下与训练配置匹配的 `norm_stats.json`。如果不存在，OpenDM 会回退到
  `./norm_stats/` 下的匹配文件。
- 一张 NVIDIA GPU。fast backend 还需要 TensorRT 和 Triton。

Checkpoint、playground 入口、`chunk_size`、image keys、state/action 维度和归一化统计
必须来自同一套训练配置。

## 2. 选择推理入口

| 使用场景 | 入口 | 常用 checkpoint | Chunk size | 图片数 | Action 维度 |
| --- | --- | --- | ---: | ---: | ---: |
| LIBERO | `playground/dm05_libero.py` | `Dexmal/DM05-libero` | 10 | 2 | 7 |
| RoboTwin 2.0 | `playground/dm05_robotwin2.py` | `Dexmal/DM05-robotwin2` | 50 | 3 | 14 |
| Demo 或自定义 SFT | `playground/dm05_sft_demo.py` 或自定义入口 | SFT checkpoint | 训练值 | 训练值 | 训练值 |
| LIBERO LoRA | `playground/dm05_libero_lora.py` | LIBERO LoRA step checkpoint | 10 | 2 | 7 |

不要混用不同 benchmark 的 checkpoint、入口或推理维度。

## 3. 启动 Default Backend

Default backend 使用标准 PyTorch 推理路径。推理模式下 launcher 会直接启动一个 Python
进程，不需要传 `--nproc_per_node`。

### LIBERO

下载发布的 checkpoint：

```bash
hf download Dexmal/DM05-libero \
  --local-dir ./checkpoints/DM05-libero
```

启动服务：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --model-config.chunk-size 10 \
  --inference-config.output-action-dim 7 \
  --inference-config.image-keys images_1 images_2 \
  --inference-config.port 7891
```

LIBERO 入口使用两张图片、8 维 state、7 维 action 和 `Franka` robot type。

### RoboTwin 2.0

下载发布的 checkpoint：

```bash
hf download Dexmal/DM05-robotwin2 \
  --local-dir ./checkpoints/DM05-robotwin2-bf16
```

启动服务：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_robotwin2.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-robotwin2-bf16 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-keys images_1 images_2 images_3 \
  --inference-config.port 7891
```

RoboTwin 2.0 入口使用三张图片、14 维 state/action 和 `Aloha RoboTwin2` robot type。

### Demo 或自定义 SFT

使用训练产出的 checkpoint，并保持所有数据相关配置与训练一致。内置 demo 的命令为：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_sft_demo.py \
  --task inference \
  --data-config.dataset-name demo \
  --model-config.model-name-or-path ./user_checkpoints/dm05_sft_demo_smoke/checkpoint-10 \
  --model-config.chunk-size 50 \
  --inference-config.output-action-dim 14 \
  --inference-config.image-keys images_1 images_2 images_3 \
  --inference-config.port 7891
```

使用自定义数据时，将入口、dataset name、checkpoint、chunk size、image keys 和 action
维度替换为训练时使用的值。

### LIBERO LoRA

将 LoRA step checkpoint 作为 `model-name-or-path`。Loader 会读取其中的
`adapter_config.json`，加载记录的 base model，并合并 adapter 用于推理。

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero_lora.py \
  --task inference \
  --model-config.model-name-or-path ${TRAINING_OUTPUT_DIR}/checkpoint-50000 \
  --model-config.chunk-size 10 \
  --inference-config.output-action-dim 7 \
  --inference-config.image-keys images_1 images_2 \
  --inference-config.port 7891
```

## 4. 启动 Fast Backend

Fast backend 使用 TensorRT vision encoder、优化后的 attention/MLP kernels 和启动时
预捕获的 CUDA Graph profiles 来降低推理延迟。

安装可选依赖：

```bash
pip install -e ".[fast-infer]"
```

在对应的 default backend 命令中加入 `--inference-config.backend fast`。如果不传
`--inference-config.vision-trt-engine-path`，默认 engine 路径是
`checkpoints/trt_engines/dm05_vision.engine`。LIBERO 完整示例：

```bash
script/dm05_launcher.sh \
  --exp playground/dm05_libero.py \
  --task inference \
  --model-config.model-name-or-path ./checkpoints/DM05-libero \
  --model-config.chunk-size 10 \
  --inference-config.backend fast \
  --inference-config.output-action-dim 7 \
  --inference-config.image-keys images_1 images_2 \
  --inference-config.port 7891
```

如果 engine 不存在，launcher 会在 HTTP 服务启动前自动构建。每个 checkpoint 和图片
布局应使用独立的 engine 路径。现有 engine 目前只按图片数量决定是否复用，因此更换
checkpoint 时必须换新路径，或者传 `--inference-config.force-rebuild`。

手动构建 engine：

```bash
python -m opendm.infer.build_vision_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --onnx-path checkpoints/trt_engines/dm05_vision.onnx \
  --engine-path checkpoints/trt_engines/dm05_vision.engine \
  --num-images 2
```

`--num-images` 必须等于 `--inference-config.image-keys` 的数量。

### Fast Backend 约束

- 请求使用 batch size 1，服务串行处理请求。
- `diffusion_steps` 在服务启动并 capture profiles 后固定。
- 预处理后的 multimodal prefix 最长为 1024 tokens。
- 默认 prefix buckets 为 `576 704 768 896 1024`；无法容纳当前图片数量的过小默认值
  会被自动跳过。
- 自定义 bucket 列表必须非空、严格递增，并且不能超过 1024。
- Bucket 越多，服务启动时间和显存占用越高。
- 请求超过最大自定义 bucket、但仍不超过 1024 时，会使用更慢的 eager fallback。

只有 workload 需要不同 prefix shape 时才覆盖默认值：

```bash
--inference-config.prefix-seq-len-buckets 576 704 768 896 1024
```

所有配置的 profiles 都会在 HTTP 服务就绪前完成 capture。

## 5. 调用 HTTP API

服务提供 `POST /process_frame`，请求格式为 `multipart/form-data`。一个包含两张图片的
LIBERO 请求如下：

```bash
curl -X POST http://127.0.0.1:7891/process_frame \
  -F 'text=pick up the black bowl and place it on the plate' \
  -F 'states=[0,0,0,0,0,0,0,0]' \
  -F image=@/path/to/agentview.jpg \
  -F image=@/path/to/wrist.jpg
```

请求字段：

- `text`：任务指令，默认是空字符串。
- `states`：必填的一维 JSON array，长度和顺序必须与 checkpoint 的归一化统计一致。
- `image`：可重复的图片字段，数量和顺序必须与 `image_keys` 一致。
- `robot_type`：用于说明 state/action 语义的机器人类型。LIBERO 和 RoboTwin 入口分别
  默认使用 `Franka` 和 `Aloha RoboTwin2`；自定义 relative-action 入口可能要求显式传入。
- `control_mode` 和 `speed`：可选文本条件；checkpoint 训练时使用了这些字段才需要传入。

正常响应包含 action chunk 和仅模型调用延迟：

```json
{
  "response": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "model_latency_ms": 71.884
}
```

`model_latency_ms` 统计同步后的模型调用，不包含请求解析、图片解码、tokenization、
归一化和响应序列化。

使用内置 demo checkpoint 及其三图请求格式时，也可以运行：

```bash
bash tests/curl_demo.sh http://127.0.0.1:7891/process_frame
```

## 6. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--exp` | 与 checkpoint 和任务匹配的 playground 入口。 |
| `--model-config.model-name-or-path` | Full model 或 LoRA checkpoint 目录。 |
| `--model-config.chunk-size` | Action horizon，必须与训练和客户端一致。 |
| `--trainer-config.model-max-length` | Tokenized multimodal prefix 最大长度。 |
| `--inference-config.diffusion-steps` | Action diffusion steps，默认值为 `10`。 |
| `--inference-config.output-action-dim` | 返回的 action 维度，必须与归一化统计一致。 |
| `--inference-config.image-keys` | 服务期望的有序图片输入。 |
| `--inference-config.backend` | `default` 或 `fast`。 |
| `--inference-config.vision-trt-engine-path` | 当前 checkpoint 专用的 TensorRT vision engine 路径；默认值为 `checkpoints/trt_engines/dm05_vision.engine`。 |
| `--inference-config.force-rebuild` | Fast 推理前重新构建 vision engine。 |
| `--inference-config.prefix-seq-len-buckets` | 可选的非空 fast-backend 自定义 buckets。 |
| `--inference-config.port` | HTTP 服务端口，默认值为 `7891`。 |

## 7. 常见问题

| 错误或现象 | 检查项 |
| --- | --- |
| 找不到归一化统计或统计不匹配 | 使用 checkpoint 的 `norm_stats.json`，并保持 dataset/action/chunk 配置与训练一致。 |
| State 或 action 维度错误 | 让 `states` 和 `output_action_dim` 与归一化向量维度一致。 |
| 上传图片数量错误 | 按 `image_keys` 的数量和顺序上传图片。 |
| TensorRT 图片数量不匹配 | 使用与 image keys 数量相同的 `--num-images` 重建 engine。 |
| 更换 checkpoint 后复用同一 engine 导致结果异常 | 使用 checkpoint 专用 engine 路径，或者传 `--inference-config.force-rebuild`。 |
| Prefix buckets 为空、乱序或过大 | 传入非空递增列表，且所有值不超过 1024。 |
| Prefix 超过 1024 tokens | 缩短 instruction 或降低 `model_max_length`。 |
| Fast 服务长时间未就绪 | 等待 engine 准备和所有 CUDA Graph profiles 在启动阶段完成 capture。 |

## 8. 相关指南

- [DM05 LIBERO 训练与评测](dm05_libero.md)
- [DM05 RoboTwin 2.0 训练与评测](dm05_robotwin2.md)
- [DM05 SFT 与验证](dm05_finetuning.md)
- [DM05 LIBERO LoRA 训练](dm05_libero_lora_training.md)
