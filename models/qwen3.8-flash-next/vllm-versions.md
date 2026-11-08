# Qwen3.8 Flash Next: vLLM 0.31.0, 0.32.0 or 0.33.0

This model can be built and run on three vLLM releases. 0.32 is the default; 0.31 is chosen with `VLLM=0.31`, and 0.33
(experimental, [see below](#vllm-0330-experimental)) with `VLLM=0.33`:

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

## vLLM 0.33.0 (experimental)

Built from this repository (`VLLM=0.33 scripts/build.sh`) and measured on all eight profiles on 2026-11-07, same machine, same
tests. All numbers are in [measurements](measurements.md).

| | 0.32.0 (default) | 0.33.0 |
|---|---|---|
| Decode, 1 request | as in [measurements](measurements.md#decode-speed) | 102-105 % of 0.32.0 per profile |
| Decode, all at once | as in [measurements](measurements.md#decode-speed) | 99-109 % of 0.32.0 per profile |
| Tool calling / control prompts, every profile | 18/18, 4/4 | 18/18, 4/4 |
| KV cache size | as in [measurements](measurements.md#capacity) | 12-36 % smaller (most with the most slots) |
| Profiles whose KV cache holds every slot at full length | 8 of 8 | 2 of 8 (`640k-c2`, `1024k-c1`; both filled with 0 preemptions) |
| MemAvailable after start | 6.1-7.7 GiB | 5.0-7.2 GiB |
| Ready after (container start to API ready, warm caches) | 121-133 s | 127-136 s |

What changes on 0.33 in this repository: the MoE kernels are Marlin (as on 0.31), and `--mamba-cache-mode` is no longer a
command-line option of vLLM, so a profile's mamba cache mode is not passed. The MTP draft rope fix and the CuTeDSL warm-up fix
are still needed, the MTP fix adapted to 0.33's code (`apply_mtp_rope_v033.py`).

So 0.33.0 is a little faster but holds fewer tokens: with the profiles as they are, six of the eight cannot keep all their
conversations at full length at once, and the server would preempt (throw a conversation's cache away and recompute it later).
That is why 0.32.0 stays the default. Use 0.33.0 for `640k-c2` or `1024k-c1`, or where the conversations stay well below the
profile's context window.
