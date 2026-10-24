# Qwen3.8 Flash Next: vLLM 0.31.0 or 0.32.0

This model can be built and run on two vLLM releases. 0.31 is the default; 0.32 is chosen with `VLLM=0.32`:

```bash
VLLM=0.32 scripts/install.sh qwen3.8-flash-next          # or step by step:
VLLM=0.32 scripts/build.sh qwen3.8-flash-next
VLLM=0.32 PROFILE=1024k-c1 scripts/run.sh qwen3.8-flash-next
```

Both images can be on the machine at the same time (`vllm-dgx-spark-gb10:qwen3.8-flash-next-vllm0.31` and `-vllm0.32`), each
with its own compile cache; `scripts/run.sh` starts the one that matches `VLLM`. The profiles are the same. What differs is in
[pins](pins.md) and [architecture](architecture.md#image-layers); on 0.32 the server uses the Humming MoE kernels instead of
Marlin, and mamba cache mode `align` instead of `all` (0.32 no longer has `all`).

## Measured

vLLM 0.32.0 was compared on all eight profiles, on one DGX Spark, 2026-10-23/24, with the same GB10 layers plus the two fixes
described in [docs/vllm-versions.md](../../docs/vllm-versions.md#problems-we-found-in-vllm-0320), Humming MoE kernels.

| | 0.31.0 | 0.32.0 + two fixes |
|---|---|---|
| Profiles that fill to full context, 0 preemptions | 8 of 8 | 8 of 8 |
| Tool calling / control prompts, every profile | 18/18, 4/4 | 18/18, 4/4 |
| Decode, 1 request (mean over the profiles) | as in [measurements](measurements.md#decode-speed) | about 99 % of that (97-103 % per profile) |
| Decode, all at once (mean over the profiles) | as in [measurements](measurements.md#decode-speed) | about 98 % of that (94-103 % per profile) |
| Decode with thinking on | as in [measurements](measurements.md#thinking-on-vs-off) | the same within +/- 3 % |
| Ready after, `512k-c3`, 3 restarts | 126.7 / 126.8 / 126.7 s | 117.7 / 117.7 / 120.7 s |
| KV cache size | as in [measurements](measurements.md#capacity) | 1-5 % larger |

So 0.32.0 starts about 7-9 s faster, has a slightly larger KV cache and decodes at the same speed.

These numbers come from an image with the same layers built outside this repository. The 0.32 image built by
`Dockerfile.vllm-0.32` has not been run through the full test yet; until it has, treat 0.32 here as not verified.

Without the two fixes, 0.32.0 does not run this model's profiles above 256K: every one of them crashed at ~265K tokens in the
fill test (8 of 8 runs), and engine initialisation took 50-57 s instead of about 30 s.
