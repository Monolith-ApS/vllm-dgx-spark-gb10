# Qwen3.8 Flash Next: vLLM 0.31.0, 0.32.0 or 0.33.0

This model can be built and run on three vLLM releases. 0.33 is the default; 0.32 is chosen with `VLLM=0.32` and 0.31 with `VLLM=0.31`:

```bash
VLLM=0.31 scripts/install.sh qwen3.8-flash-next          # or step by step:
VLLM=0.31 scripts/build.sh qwen3.8-flash-next
VLLM=0.31 PROFILE=1024k-c1 scripts/run.sh qwen3.8-flash-next
```

The images can be on the machine at the same time (`vllm-dgx-spark-gb10:qwen3.8-flash-next-vllm0.31`, `-vllm0.32`, `-vllm0.33`), each
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
| KV cache size | as in [measurements](measurements.md#capacity) | 1-5 % larger (2-6 % for the build from this repository) |

So 0.32.0 starts about 7-9 s faster, has a slightly larger KV cache and decodes at the same speed.

These numbers were first measured with an image of the same layers built outside this repository.

### The build from this repository

The image built by `VLLM=0.32 scripts/build.sh` went through the same test on 2026-10-25: every profile filled to the full
context with 0 preemptions, tool calling 18/18 and the control prompts 4/4 on all eight. All its numbers, next to those of
0.31.0, are in [measurements](measurements.md).

Against 0.31.0 that is 97-100 % of the decode speed for one request, 97-103 % with all conversations at once, a 2-6 % larger
KV cache and 0.1-0.6 GiB more free memory after the fill test. Ready after (container start to API ready, warm caches):
121-133 s.

Without the two fixes, 0.32.0 does not run this model's profiles above 256K: every one of them crashed at ~265K tokens in the
fill test (8 of 8 runs), and engine initialisation took 50-57 s instead of about 30 s.

## vLLM 0.33.0 (default)

Built from this repository (`scripts/build.sh`, `VLLM=0.33`). The first comparison below was measured on all eight profiles on
2026-11-07, same machine, same tests, with the attention block size still at 1680; all numbers are in
[measurements](measurements.md). The block size was changed to 3456 the day after (see below), so the KV pool figures in this
table are the old ones.

| | 0.32.0 | 0.33.0 |
|---|---|---|
| Decode, 1 request | as in [measurements](measurements.md#decode-speed) | 102-105 % of 0.32.0 per profile |
| Decode, all at once | as in [measurements](measurements.md#decode-speed) | 99-109 % of 0.32.0 per profile |
| Tool calling / control prompts, every profile | 18/18, 4/4 | 18/18, 4/4 |
| KV cache size (block size 1680, before the fix) | as in [measurements](measurements.md#capacity) | 12-36 % smaller |
| Ready after (container start to API ready, warm caches) | 121-133 s | 127-136 s |

What changes on 0.33 in this repository: the MoE kernels are Marlin (as on 0.31), `--mamba-cache-mode` is no longer a
command-line option of vLLM, so a profile's mamba cache mode is not passed, and the attention block size is 3456 instead of
1680 (0.33 rounds the block up to a multiple of lcm(`--block-size`, 64) in mamba align mode, so 1680 became 6720 and padded the
mamba pages 105 %; 3456 pads them 5.6 %). The MTP draft rope fix and the CuTeDSL warm-up fix are still needed, the MTP fix
adapted to 0.33's code (`apply_mtp_rope_v033.py`).

**Profiles on 0.33.** With the block size at 3456 the KV pool of `128k-c8` and `384k-c4` holds one conversation less, so on 0.33
these two are `128k-c7` and `384k-c3` (`PROFILE=list` shows what a release offers; asking for the other name says which to
use). The others are unchanged, with `gpu_memory_utilization` 0.88. On the machine this repository was developed on, every one
of these profiles booted on 0.33 and passed the control prompts, tool calling and a capacity check through the switcher; the
full fill test of `tools/fillc.py` has not been re-run on the image built from this repository since the block size changed.

**Why 0.33 is the default.** Together with the content guard (see [configuration](configuration.md#very-long-conversations-content-guard))
it is the setup that runs on the maintainers' own Spark: thinking off by default, MTP 7, FP8 KV cache with layer 48 kept in
bf16, and long conversations (above about 175K tokens) answering to the end instead of stopping in the middle of a sentence.
If your conversations need every slot of `128k-c8` or `384k-c4`, use `VLLM=0.32`.
