# gb10-decode

Faster decoding on GB10 by reading fewer bytes per token. `apply_gb10.py <site-packages>/vllm` patches vLLM; it finds the
code of vLLM 0.31 and 0.32 itself, so the same files are used on both releases.

| Switch | Default | What it does |
|---|---|---|
| `VLLM_GB10_FP8_LMHEAD` | `1` | LM head as FP8 rows with one fp32 scale per row: decode reads half the bytes of BF16 |
| `VLLM_GB10_SHARED_GATE` | `1` | MoE shared-expert gate: gemv + sigmoid + multiply in one kernel |
| `VLLM_GB10_ROUTER_LL` | `1` | router GEMM may use the CuTeDSL low-latency BF16 tier on GB10 |
| `VLLM_GB10_FP8_WEIGHTS` | `0` | BF16 linear layers become weight-only FP8 rows after loading |
| `VLLM_GB10_FP4_WEIGHTS` | `0` | selected layers (default the GDN projections) become weight-only NVFP4 rows instead |

The full list and the details are in the docstring of `apply_gb10.py`. Each model's `serve.py` sets the switches it uses
(Qwen3.8 Flash Next turns on FP8 and FP4 weights). Every patch refuses an anchor that does not match exactly once.

Files: `apply_gb10.py` (the patches), `gb10_kernels.py` (Triton kernels), `bench_kernels.py`, `bench_fp4.py` (micro
benchmarks). Tested with Qwen3.8 Flash Next on vLLM 0.31 and 0.32.
