# Pins

Everything the image and the download depend on is fixed to an exact version, so a build today and a build next year give the
same result. This model can be built on three vLLM releases (`VLLM=0.33`, the default, `VLLM=0.32`, or `VLLM=0.31`); each
has its own Dockerfile.

| What | vLLM 0.31 (`Dockerfile.vllm-0.31`) | vLLM 0.32 (`Dockerfile.vllm-0.32`) | vLLM 0.33 (`Dockerfile.vllm-0.33`) |
|---|---|---|---|
| vLLM | `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` | `vllm/vllm-openai:v0.32.0@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc` | `vllm/vllm-openai:v0.33.0@sha256:e3481489818d3e563cae517b84ab959cc8b69f85a9fbb771c189140131ffc65c` |
| blazux/qwen3.8-Flash-DGX (set in `pre-build.sh`) | `5108d90dedf15a31aa20432319f151b2fdda17a3` | same | same |
| jschmied/qwen38-flash-next-gb10 (files checksum-pinned) | `e0ef69d4f5575dad00d34e05479eaf4c6547bace` | same | same |
| nvidia/Qwen3.8-Flash-Next-NVFP4 (set in `model.env`) | `fc694b54fb0174e0913e6adf86691ef85a4ead47` | same | same |

The rest is part of this repository:

| | vLLM 0.31 | vLLM 0.32 | vLLM 0.33 |
|---|---|---|---|
| Overlay files in `overlay/` | the two `v031` files (steps 12+13 and 14) | the same two, plus the two `v032` files (steps 1 and 4) | as 0.32, steps 12+13 from `patches/qwen-tool-12-13-v033.diff` |
| `gb10/think-eos-guard` | `serving_think_guard-v031.diff` | `serving_think_guard-v032.diff` | `serving_think_guard-v033.diff` |
| `gb10/yarn-mrope-58879` | `yarn_mrope_58879-v031.diff` | `yarn_mrope_58879-v032.diff` | `yarn_mrope_58879-v032.diff` |
| `gb10/gb10-decode` | `apply_gb10.py` (finds the anchors of each release) | same | same |
| `gb10/cutedsl-warmup`, `gb10/mtp-draft-rope` | not used | used | used (`apply_mtp_rope_v033.py`) |

How an image is put together: [docs/image.md](../../docs/image.md); the layers of this one:
[architecture](architecture.md#image-layers).
