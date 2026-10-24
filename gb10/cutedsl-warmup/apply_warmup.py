#!/usr/bin/env python3
"""vLLM 0.32 CuTeDSL warmup fix for GB10: stop compiling skinny-GEMM kernels that are already built or never used.

  python3 apply_warmup.py <site-packages>/vllm

vLLM 0.32 compiles every "shape-dynamic skinny GEMM" config of the SM121 Qwen4Exp plans in the CuTeDSL warmup on every
boot (47 units at TP=1, ~17 s; cute.compile results are not cached across processes). Two things make most of that
work wasted on our image:
  1. enable_qwen4_exp_low_latency_gemm requests the configs before process_weights_after_loading, so layers that the
     GB10 decode layer then turns into FP8/FP4 rows (gb10_w_scale / gb10_fp4_scales, which bypass the skinny GEMM) and
     the LM head (FP8 rows for M <= VLLM_GB10_FP8_LMHEAD_MAX_M) still get their configs compiled;
  2. the warmup compiles configs that the profiling CUDA graph capture already compiled lazily (it never checks
     _compiled).
With VLLM_GB10_WARMUP_FIX=1 (default) the warmup compiles only configs of layers that still use the skinny GEMM and
that are not compiled yet. VLLM_GB10_WARMUP_FIX=0 keeps the stock behaviour; both log what they found. Every skinny
GEMM shape actually called and every compile after the warmup (a kernel built while serving) is logged once.
Refuses an anchor that does not match exactly once, so a drifted base fails the build.
"""
import os
import sys

pkg = sys.argv[1]


def patch(rel, pairs):
    p = os.path.join(pkg, rel)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            sys.exit("%s: anchor found %d times: %r" % (rel, n, old[:80]))
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("patched", rel)


patch("model_executor/kernels/linear/cute_dsl/skinny_gemm.py", [(
    """class ShapeDynamicSkinnyGemm:
""",
    """def _gb10_warmup_configs(gemm):
    # GB10: only configs of layers still on the skinny GEMM after weight processing, and not compiled yet.
    import os

    requested = set(gemm._warmup_configs)
    layers = getattr(gemm, "_gb10_layers", None) or []
    lmhead_fp8 = os.environ.get("VLLM_GB10_FP8_LMHEAD", "1") == "1"
    keep, kept, converted = set(), {}, {}
    for ref, plan, is_head, shape in layers:
        layer = ref()
        if layer is None:
            continue
        conv = (
            getattr(layer, "gb10_w_scale", None) is not None
            or getattr(layer, "gb10_fp4_scales", None) is not None
            or (is_head and lmhead_fp8)
        )
        d = converted if conv else kept
        d[shape] = d.get(shape, 0) + 1
        if not conv:
            keep.update(plan)
    wanted = {k for k in requested if k[1] in keep} if layers else requested
    done = {k for k in wanted if k in gemm._compiled}
    fix = os.environ.get("VLLM_GB10_WARMUP_FIX", "1") == "1"
    logger.warning(
        "GB10 warmup: skinny-GEMM layers kept %s, converted (FP8/FP4 rows, FP8 head) %s; configs requested %d, "
        "used by kept layers %d, already compiled %d, to compile %d (fix=%s; stock compiles %d)",
        sorted(kept.items()), sorted(converted.items()), len(requested), len(wanted), len(done),
        len(wanted - done), fix, len(requested),
    )
    gemm._gb10_warmup_done = True
    return wanted - done if fix else requested


class ShapeDynamicSkinnyGemm:
"""), (
    """            for dtype, config, has_residual in sorted(
                self._warmup_configs,
""",
    """            for dtype, config, has_residual in sorted(
                _gb10_warmup_configs(self),
"""), (
    """        if cache_key not in self._compiled:
            self._compile(a.dtype, config, has_residual)
""",
    """        if cache_key not in self._compiled:
            if getattr(self, "_gb10_warmup_done", False):
                logger.warning("GB10 warmup: skinny GEMM compiled after the warmup: %s M=%d", config, a.shape[0])
            self._compile(a.dtype, config, has_residual)
""")])

patch("models/qwen4_exp/nvidia/low_latency_gemm.py", [(
    """    config = None if plan is None else plan.get(x.shape[0])
""",
    """    config = None if plan is None else plan.get(x.shape[0])
    _seen_key = (tuple(weight.shape), x.shape[0], config is not None)
    if _seen_key not in _GB10_SEEN:
        _GB10_SEEN.add(_seen_key)
        logger.warning("GB10 warmup: skinny-GEMM op called N,K=%s M=%d plan=%s", _seen_key[0], _seen_key[1], _seen_key[2])
"""), (
    """def _qwen4_exp_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
""",
    """_GB10_SEEN: set = set()


def _qwen4_exp_low_latency_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
"""), (
    """        warmup_configs.update(plan.values())
""",
    """        warmup_configs.update(plan.values())
        if not hasattr(shape_dynamic_skinny_gemm, "_gb10_layers"):
            shape_dynamic_skinny_gemm._gb10_layers = []
        import weakref

        shape_dynamic_skinny_gemm._gb10_layers.append(
            (weakref.ref(child), tuple(plan.values()), is_head, tuple(weight.shape))
        )
"""), (
    """import torch
from torch import nn
""",
    """import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)
""")])
