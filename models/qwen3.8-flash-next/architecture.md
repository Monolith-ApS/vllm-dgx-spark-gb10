# Qwen3.8 Flash Next: architecture

How images are built in general: [docs/image.md](../../docs/image.md).

## Image layers

```mermaid
flowchart TB
    A["vllm/vllm-openai:v0.31.0<br/>(official, pinned by digest)"] --> B
    B["Qwen3.8 Flash Next overlay<br/>blazux/qwen3.8-Flash-DGX @ 5108d90<br/>PLE mmap, FLA GB10 fixes, mamba block size,<br/>QSA exact/deterministic top-k, NVFP4+FP8 hybrid loader,<br/>MTP draft vocabulary 65,536, tool-parser fixes,<br/>weight-loading fixes"] --> C
    C["think-eos-guard<br/>EOS inside open reasoning is resumed;<br/>quoted closing think tag stays text"] --> D
    D["YaRN / mRoPE correction range<br/>vLLM PR #58879"] --> E
    E["GB10 decode layer<br/>FP8 LM head, fused shared-expert gate,<br/>GB10 router GEMM tier,<br/>FP8 / NVFP4 weight-only linears"] --> F
    F["serve.py + profiles.json<br/>(entrypoint)"]
    K["jschmied/qwen38-flash-next-gb10 @ e0ef69d<br/>deterministic top-k kernel sources<br/>(checksum-pinned, compiled in the build)"] -.-> B
```

This is the image for vLLM 0.31 (`Dockerfile.vllm-0.31`, `VLLM=0.31`). The image for vLLM 0.32 (`Dockerfile.vllm-0.32`, the
default) has the same layers on vLLM 0.32.0, with steps 1 and 4 of the overlay and the think-eos-guard and YaRN diffs
adapted to 0.32, and two more layers before the entrypoint: the CuTeDSL warm-up fix (`gb10/cutedsl-warmup`) and the MTP draft
rope fix (`gb10/mtp-draft-rope`). On 0.32 the server uses the Humming MoE kernels instead of Marlin ([pins](pins.md),
[vLLM 0.31 against 0.32](vllm-versions.md)).

## Model components and precision

Qwen3.8 Flash Next: 48 decoder layers (36 Gated DeltaNet linear-attention layers + 12 full-attention layers, every 4th:
3, 7, ..., 47), MoE in every layer (512 routed experts, 10 active, + 1 shared expert), hidden size 2560, vocabulary 248,320,
one MTP layer, native context 262,144.

"Checkpoint" = as stored in `nvidia/Qwen3.8-Flash-Next-NVFP4`. "Image" = converted at load time by the GB10 layer of this image.

```mermaid
flowchart TB
    IN["Token ids"] --> EMB["embed_tokens<br/>BF16 (checkpoint)"]
    IN --> PLE["PLE n-gram embedding (layer 1)<br/>FP8 (checkpoint), read from disk by mmap"]
    EMB --> L
    PLE --> L
    subgraph L["48 decoder layers"]
        direction TB
        HC["Hyper-connections<br/>input_mix_weight_up/down: FP8 (image)<br/>block_inject, hc_norm: BF16"]
        subgraph MIX["Token mixer"]
            direction LR
            subgraph GDN["36 x Gated DeltaNet"]
                G1["in_proj_qkv, in_proj_z, out_proj<br/>NVFP4 weight-only (image)"]
                G2["in_proj_a, in_proj_b, conv1d, A_log, dt_bias, norm<br/>BF16 (checkpoint)"]
                G3["recurrent state: FP32<br/>prefix-state cache: mamba cache mode per profile"]
            end
            subgraph ATT["12 x full attention (QSA)"]
                A1["q/k/v/o projections<br/>FP8 weight-only (image)"]
                A2["indexer index_qk_proj<br/>FP8 weight-only (image)"]
                A3["KV cache: FP8 e4m3"]
            end
        end
        subgraph MOE["MoE"]
            direction LR
            M1["router gate<br/>BF16, GB10 router GEMM tier"]
            M2["512 routed experts, top-10<br/>NVFP4 (checkpoint), Marlin W4A16"]
            M3["shared expert gate/up/down<br/>FP8 weight-only (image)"]
            M4["shared expert gate<br/>BF16, fused gemv+sigmoid+mul"]
        end
        HC --> MIX --> MOE
    end
    L --> LMH["lm_head<br/>FP8 rows + FP32 row scale (image)"]
    L --> MTP
    subgraph MTP["MTP drafter (1 layer, 7 speculative tokens, 3 with thinking)"]
        T1["routed experts: FP8 block-scaled (checkpoint)"]
        T2["attention, shared expert, fc_hidden: FP8 weight-only (image)"]
        T3["draft head: reduced vocabulary 65,536, FP8 rows (image)"]
        T4["KV cache: BF16 (layer 48 skipped from FP8 KV)"]
    end
    VIS["Vision encoder (image input): BF16 (checkpoint), idle for text-only requests"]
```

| Component | Checkpoint | Served as | Switch |
|---|---|---|---|
| Routed experts (512 x 48 layers) | NVFP4 | NVFP4, Marlin MoE backend | `--moe-backend marlin` |
| Gated DeltaNet projections `in_proj_qkv`, `in_proj_z`, `out_proj` | BF16 | NVFP4 weight-only (e2m1 + FP8 scale per 16 + FP32 row scale) | `VLLM_GB10_FP4_WEIGHTS=1` |
| Attention `q/k/v/o`, indexer `index_qk_proj` | BF16 | FP8 weight-only + FP32 row scale | `VLLM_GB10_FP8_WEIGHTS=1` |
| Shared expert `gate/up/down` | BF16 | FP8 weight-only | `VLLM_GB10_FP8_WEIGHTS=1` |
| Hyper-connection `input_mix_weight_up/down` | BF16 | FP8 weight-only | `VLLM_GB10_FP8_WEIGHTS=1` |
| `lm_head` | BF16 | FP8 rows + FP32 row scale | `VLLM_GB10_FP8_LMHEAD=1` (default) |
| MTP experts | FP8 block-scaled | FP8 block-scaled | - |
| MTP attention, shared expert, `fc_hidden` | BF16 | FP8 weight-only (FP4 skipped for `mtp`) | `VLLM_GB10_FP8_WEIGHTS=1`, `VLLM_GB10_FP4_SKIP=mtp` |
| MTP draft head | - | 65,536-token vocabulary, FP8 rows | `VLLM_MTP_DRAFT_VOCAB`, `VLLM_GB10_FP8_DRAFT_HEAD=1` (default) |
| PLE n-gram embedding | FP8 | FP8, read from disk (mmap) | `VLLM_PLE_MMAP=1` |
| `embed_tokens`, norms, router gate, GDN `in_proj_a/b`, `conv1d`, vision encoder | BF16 | BF16 | - |
| KV cache, 12 attention layers | - | FP8 e4m3 | `--kv-cache-dtype fp8_e4m3` |
| KV cache, MTP layer (48) | - | BF16 | `--kv-cache-dtype-skip-layers 48` |
| Gated DeltaNet recurrent state | FP32 | FP32 | - |

Weight-only means activations stay BF16; weights are dequantized inside the GEMM kernels.

The vision encoder (0.84 GiB) is loaded at start and runs only for requests that contain images. An image costs one
token per 32x32 pixels (16-pixel patches, 2x2 merge): at least 64 tokens, 768 for 1024x768, at most 16,384 (larger
images are scaled down). The MTP drafter is text-only.
