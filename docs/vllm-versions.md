# vLLM versions on the GB10

This repository is about getting the most performance, quality and stability out of vLLM on the GB10 chip. A new vLLM release
is therefore not adopted because it is newer, but when it has been measured against the one in use: same profiles, same
tests, same machine. This page is about vLLM itself: which combinations of release and patches we have run, how they compare
in speed, and the problems with long context windows and start-up that we ran into.

## Policy

- A model can be offered on more than one vLLM release: one Dockerfile per release (`models/<model>/Dockerfile.vllm-<release>`),
  each pinned by digest, chosen with `VLLM=<release>` in the scripts. The releases a model offers and its default are in its
  `model.env`; the exact pins in its `pins.md`.
- A release is offered only after the model has been measured on it, and becomes the default only when it is at least as
  good on the points that matter: the same context windows work (filled to the full length with several conversations at
  once, without the server throwing any away), decode speed is not lower, answers and tool calling are unchanged, and
  start-up is not slower.
- A patch we need on top of a release is kept as small as possible, documented here, and reported to vLLM.

## Combinations measured

Each row is a vLLM release with the patches on top of it. The example is Qwen3.8 Flash Next on its eight profiles (128K x 8 to
1M x 1); the full numbers are in [its comparison](../models/qwen3.8-flash-next/vllm-versions.md).

| Combination | Contexts above 256K | Decode speed | Start-up (engine initialisation / ready after) | In this repository |
|---|---|---|---|---|
| 0.31.0 + GB10 layers | work | baseline | ~30 s / ~127 s | `VLLM=0.31` (default) |
| 0.32.0 + GB10 layers | crash at ~265K tokens | - | 50-57 s / - | not offered |
| 0.32.0 + GB10 layers + `mtp-draft-rope` + `cutedsl-warmup` | work | same as 0.31.0 (97-103 %) | 33-43 s / ~118 s | `VLLM=0.32` |

0.32.0 with both patches is 7-9 s faster to start and has a 1-5 % larger KV cache; it decodes at the same speed and needs
two patches that 0.31.0 does not. The 0.32 row was measured with an image of the same layers built outside this repository;
the build from `Dockerfile.vllm-0.32` here still has to go through the same test.

## Problems we found in vLLM 0.32.0

- **Contexts above 262,144 tokens crash with speculative decoding.** With YaRN scaling (a model that is used beyond its native
  context window) and a speculative draft layer (MTP), stock 0.32.0 stops with `CUDA error: an illegal memory access` as soon as
  a conversation passes ~265K tokens. The cause we found: the draft layer builds its rope table for only 262,144 positions,
  because it does not get the YaRN settings of the target model. The crash is reported to vLLM as
  [vllm-project/vllm#62383](https://github.com/vllm-project/vllm/issues/62383). Our patch, `gb10/mtp-draft-rope`, gives the
  draft the target's settings.
- **Start-up warm-up.** Unpatched, 0.32.0 runs 53 warm-up steps at start instead of 9-11, so engine initialisation takes 50-57 s
  (0.31: about 30 s); with our patch, `gb10/cutedsl-warmup`, 33-43 s.

## Things to know when testing several vLLM builds

- **Compile cache.** The cache of compiled graphs (`/root/.cache/vllm`) is keyed by the configuration, not by patched source
  code. Two images with the same configuration but different patches can pick up each other's compiled graph and fail at
  start with a size mismatch (`expected size 1048576==4194304`) or run the wrong code. Give every image variant its own cache
  directory when you test more than one.
- **Fill test.** A context window is only shown to work when several conversations are filled to the full length at the same
  time and nothing is preempted. A short benchmark does not show the long-context crash above; it appears only past ~265K
  tokens ([tools.md](tools.md) has the fill test).
