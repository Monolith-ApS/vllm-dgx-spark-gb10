# Building

```bash
scripts/build.sh qwen3.8-flash-next
TAG=myregistry/vllm-dgx-spark-gb10:qwen3.8-flash-next scripts/build.sh qwen3.8-flash-next
```

Build on the DGX Spark itself (aarch64, GB10 GPU architecture). `scripts/build.sh` clones the blazux overlay into `.build/` at
its pinned commit and runs:

```bash
docker build -f models/qwen3.8-flash-next/Dockerfile --build-context blazux=.build/blazux -t vllm-dgx-spark-gb10:qwen3.8-flash-next .
```

The deterministic top-k kernel is compiled during the build with `MAX_JOBS=2`. Image size: ~24 GB. The layers are shown in
[architecture.md](architecture.md#image-layers).

## Repository layout

```
gb10/                      shared GB10 layer (model independent)
  gb10-decode/             FP8 LM head, fused shared-expert gate, router tier, FP8/FP4 weight-only linears
  think-eos-guard/         reasoning-channel guard for the qwen3 reasoning parser
  yarn-mrope-58879/        YaRN correction range with mRoPE (vLLM PR #58879)
models/qwen3.8-flash-next/
  Dockerfile               the image (layers above)
  overlay/                 two overlay files rebased for vLLM v0.31
  serve.py, profiles.json  entrypoint and profiles
scripts/                   setup-host, get-model, build, run
tools/                     test tools
```

## Pins

| What | Pin |
|---|---|
| vLLM | `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3` |
| blazux/qwen3.8-Flash-DGX | `5108d90dedf15a31aa20432319f151b2fdda17a3` |
| jschmied/qwen38-flash-next-gb10 | `e0ef69d4f5575dad00d34e05479eaf4c6547bace` (files checksum-pinned) |
| nvidia/Qwen3.8-Flash-Next-NVFP4 | `fc694b54fb0174e0913e6adf86691ef85a4ead47` |
