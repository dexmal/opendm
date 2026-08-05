# DM05 Inference Guide

This is the canonical guide for starting the DM05 inference service and calling
its HTTP API. Benchmark and training guides link here instead of duplicating
inference setup.

[中文](../zh/dm05_inference.md)

## 1. Before You Start

Run all commands from the OpenDM repository root. Prepare:

- An OpenDM environment installed according to the main README.
- A DM05 checkpoint compatible with the selected playground entry point.
- The matching `norm_stats.json` in the checkpoint directory. If it is absent,
  OpenDM falls back to the matching file under `./norm_stats/`.
- One NVIDIA GPU. The fast backend additionally requires TensorRT and Triton.

The checkpoint, playground entry point, `chunk_size`, image keys, state/action
dimensions, and normalization statistics must come from the same training
configuration.

## 2. Choose an Entry Point

| Use case | Entry point | Typical checkpoint | Chunk size | Images | Action dimension |
| --- | --- | --- | ---: | ---: | ---: |
| LIBERO | `playground/dm05_libero.py` | `Dexmal/DM05-libero` | 10 | 2 | 7 |
| RoboTwin 2.0 | `playground/dm05_robotwin2.py` | `Dexmal/DM05-robotwin2` | 50 | 3 | 14 |
| Demo or custom SFT | `playground/dm05_sft_demo.py` or your own entry | Your SFT checkpoint | Training value | Training value | Training value |
| LIBERO LoRA | `playground/dm05_libero_lora.py` | A LIBERO LoRA step checkpoint | 10 | 2 | 7 |

Do not mix a benchmark checkpoint with another benchmark's entry point or
inference dimensions.

## 3. Start the Default Backend

The default backend uses the standard PyTorch inference path. The inference
launcher starts one Python process directly, so `--nproc_per_node` is not needed.

### LIBERO

Download the released checkpoint:

```bash
hf download Dexmal/DM05-libero \
  --local-dir ./checkpoints/DM05-libero
```

Start the service:

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

The LIBERO entry point uses two images, an 8-dimensional state, a 7-dimensional
action, and the `Franka` robot type.

### RoboTwin 2.0

Download the released checkpoint:

```bash
hf download Dexmal/DM05-robotwin2 \
  --local-dir ./checkpoints/DM05-robotwin2-bf16
```

Start the service:

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

The RoboTwin 2.0 entry point uses three images, a 14-dimensional state/action,
and the `Aloha RoboTwin2` robot type.

### Demo or Custom SFT

Use the checkpoint produced by training and keep all data-dependent settings
consistent with that run. For the built-in demo configuration:

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

For custom data, replace the entry point, dataset name, checkpoint, chunk size,
image keys, and action dimension with the values used during training.

### LIBERO LoRA

Pass a LoRA step checkpoint as `model-name-or-path`. The loader reads its
`adapter_config.json`, loads the recorded base model, and merges the adapter for
inference.

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

## 4. Start the Fast Backend

The fast backend uses a TensorRT vision encoder, optimized attention and MLP
kernels, and startup-captured CUDA Graph profiles to reduce latency.

Install the optional dependencies:

```bash
pip install -e ".[fast-infer]"
```

Add `--inference-config.backend fast` to the matching default backend command.
If `--inference-config.vision-trt-engine-path` is not provided, the default
engine path is `checkpoints/trt_engines/dm05_vision.engine`. For LIBERO:

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

If the engine does not exist, the launcher builds it before starting the HTTP
service. Use a separate engine path for each checkpoint and image layout. An
existing engine is currently reused based on its image count, so changing the
checkpoint requires a new path or `--inference-config.force-rebuild`.

To build an engine manually:

```bash
python -m opendm.infer.build_vision_trt \
  --checkpoint ./checkpoints/DM05-libero \
  --onnx-path checkpoints/trt_engines/dm05_vision.onnx \
  --engine-path checkpoints/trt_engines/dm05_vision.engine \
  --num-images 2
```

`--num-images` must equal the number of values passed through
`--inference-config.image-keys`.

### Fast Backend Constraints

- Requests use batch size 1 and are processed serially by the service.
- `diffusion_steps` is fixed when the service captures its profiles at startup.
- The processed multimodal prefix is limited to 1024 tokens.
- The default prefix buckets are `576 704 768 896 1024`. Defaults too small for
  the configured image count are skipped automatically.
- A custom bucket list must be non-empty, strictly increasing, and no larger
  than 1024.
- More buckets increase service startup time and GPU memory use.
- A request longer than the largest custom bucket uses a slower eager fallback,
  provided it is still within the 1024-token limit.

Override the defaults only when the workload requires different prefix shapes:

```bash
--inference-config.prefix-seq-len-buckets 576 704 768 896 1024
```

All configured profiles are captured before the HTTP service becomes ready.

## 5. Call the HTTP API

The service exposes `POST /process_frame` with `multipart/form-data`. A LIBERO
request with two images looks like:

```bash
curl -X POST http://127.0.0.1:7891/process_frame \
  -F 'text=pick up the black bowl and place it on the plate' \
  -F 'states=[0,0,0,0,0,0,0,0]' \
  -F image=@/path/to/agentview.jpg \
  -F image=@/path/to/wrist.jpg
```

Request fields:

- `text`: task instruction. It defaults to an empty string.
- `states`: required one-dimensional JSON array. Its length and ordering must
  match the checkpoint's normalization statistics.
- `image`: repeated image file field. The count and order must match
  `image_keys`.
- `robot_type`: robot embodiment used for state/action semantics. LIBERO and
  RoboTwin entry points default to `Franka` and `Aloha RoboTwin2`, respectively.
  Custom relative-action entries may require this field explicitly.
- `control_mode` and `speed`: optional text-conditioning fields. Provide them
  when the checkpoint was trained with these fields.

A successful response contains the action chunk and model-only latency:

```json
{
  "response": [
    [0.012, -0.034, 0.18, 0.0, 0.0, 0.0, -1.0],
    [0.015, -0.031, 0.17, 0.0, 0.0, 0.0, -1.0]
  ],
  "model_latency_ms": 71.884
}
```

`model_latency_ms` measures the synchronized model call. It does not include
request parsing, image decoding, tokenization, normalization, or response
serialization.

For the built-in demo checkpoint and its three-image request format, you can
also run:

```bash
bash tests/curl_demo.sh http://127.0.0.1:7891/process_frame
```

## 6. Common Parameters

| Parameter | Meaning |
| --- | --- |
| `--exp` | Playground entry point matching the checkpoint and task. |
| `--model-config.model-name-or-path` | Full model or LoRA checkpoint directory. |
| `--model-config.chunk-size` | Action horizon; must match training and the client. |
| `--trainer-config.model-max-length` | Maximum tokenized multimodal prefix length. |
| `--inference-config.diffusion-steps` | Number of action diffusion steps; default `10`. |
| `--inference-config.output-action-dim` | Returned action dimension; must match normalization statistics. |
| `--inference-config.image-keys` | Ordered image inputs expected by the service. |
| `--inference-config.backend` | `default` or `fast`. |
| `--inference-config.vision-trt-engine-path` | Checkpoint-specific TensorRT vision engine path; default `checkpoints/trt_engines/dm05_vision.engine`. |
| `--inference-config.force-rebuild` | Rebuild the vision engine before fast inference. |
| `--inference-config.prefix-seq-len-buckets` | Optional non-empty custom fast-backend buckets. |
| `--inference-config.port` | HTTP service port; default `7891`. |

## 7. Troubleshooting

| Error or symptom | Check |
| --- | --- |
| Missing or mismatched normalization statistics | Use the checkpoint's `norm_stats.json` and the same dataset/action/chunk configuration as training. |
| State or action dimension error | Match `states` and `output_action_dim` to the normalization vectors. |
| Wrong number of uploaded images | Send exactly one image per configured `image_key`, in the same order. |
| TensorRT image-count mismatch | Rebuild the engine with `--num-images` equal to the number of image keys. |
| Results change after switching checkpoints with the same engine | Use a checkpoint-specific engine path or pass `--inference-config.force-rebuild`. |
| Empty, unsorted, or oversized prefix buckets | Pass a non-empty increasing list whose values are at most 1024. |
| Prefix exceeds 1024 tokens | Shorten the instruction or reduce `model_max_length`. |
| Fast service takes time to become ready | Wait for engine preparation and all configured CUDA Graph profiles to finish at startup. |

## 8. Related Guides

- [DM05 LIBERO Training and Evaluation](dm05_libero.md)
- [DM05 RoboTwin 2.0 Training and Evaluation](dm05_robotwin2.md)
- [DM05 SFT and Validation](dm05_finetuning.md)
- [DM05 LIBERO LoRA Training](dm05_libero_lora_training.md)
