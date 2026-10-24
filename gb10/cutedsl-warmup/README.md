# cutedsl-warmup

vLLM 0.32 only. Shortens start-up on GB10 by compiling only the CuTeDSL skinny-GEMM kernels that are actually used and not
yet built.

vLLM 0.32 compiles every skinny-GEMM configuration of the SM121 plans in its warm-up on every start (cute.compile results are
not cached across processes). On an image with the GB10 decode layer most of that work is wasted: layers that `gb10-decode`
turns into FP8/FP4 rows, and the FP8 LM head, never use the skinny GEMM, and the warm-up also recompiles what the profiling
run already compiled. Unpatched, 0.32.0 runs 53 warm-up steps instead of 9-11 and engine initialisation takes 50-57 s; with
this patch 33-43 s ([docs/vllm-versions.md](../../docs/vllm-versions.md)).

`apply_warmup.py <site-packages>/vllm` patches the warm-up. `VLLM_GB10_WARMUP_FIX=0` at run time restores the stock warm-up;
both log what they compiled. The script refuses an anchor that does not match exactly once.

Tested with Qwen3.8 Flash Next on vLLM 0.32.0.
