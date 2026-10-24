#!/usr/bin/env python3
"""Apply the GB10 decode patches to a vLLM source tree (the package directory `vllm/`).

  apply_gb10.py VLLM_PKG_DIR

Switches (all default on, set to 0 to fall back to stock vLLM):
  VLLM_GB10_FP8_LMHEAD    LM head as FP8 rows + fp32 row scale (decode reads 311 MB instead of 622 MB on Qwen3-Next)
  VLLM_GB10_SHARED_GATE   shared expert gate: gemv + sigmoid + mul in one kernel
  VLLM_GB10_ROUTER_LL     router GEMM: allow the cuteDSL ll_bf16 tier on family-120 Blackwell (GB10)
  VLLM_GB10_FP8_DRAFT_HEAD  qwen38-flash-dgx MTP reduced draft vocabulary (VLLM_MTP_DRAFT_VOCAB): the draft head as FP8
                          rows (only present in that image; skipped elsewhere)
Off by default:
  VLLM_GB10_FP8_WEIGHTS   unquantized (bf16) Linear layers whose name ends in one of VLLM_GB10_FP8_WEIGHTS_SUFFIXES
                          become FP8 rows + fp32 row scale after loading (weight-only, activations stay bf16): half the
                          bytes per decode token and half the memory. For checkpoints that leave attention/GDN in bf16.
  VLLM_GB10_FP4_WEIGHTS   the layers in VLLM_GB10_FP4_WEIGHTS_SUFFIXES (default: the GDN projections) become weight-only
                          NVFP4 rows instead (e2m1 + fp8 scale per 16 + fp32 row scale, 0.5625 bytes per weight)
                          Layers whose prefix contains one of VLLM_GB10_FP4_SKIP (default: mtp) stay FP8; VLLM_GB10_FP4_ONLY (default
                          empty) limits FP4 to prefixes that contain one of its entries.
"""
import os, shutil, sys

pkg = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, "gb10_kernels.py"), os.path.join(pkg, "model_executor/layers/gb10_kernels.py"))


def patch(rel, pairs):
    p = os.path.join(pkg, rel)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"{rel}: anchor found {s.count(old)} times:\n{old}")
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)
    print("patched", rel)


patch("model_executor/layers/logits_processor.py", [(
    '''        """Project hidden states through the lm_head, honoring head_dtype."""
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
''',
    '''        """Project hidden states through the lm_head, honoring head_dtype."""
        if (
            _GB10_FP8_LMHEAD
            and embedding_bias is None
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.dim() == 2
            and hidden_states.shape[0] <= _GB10_FP8_LMHEAD_MAX_M
            and lm_head.tp_size == 1
            and isinstance(getattr(lm_head, "weight", None), torch.Tensor)
            and lm_head.weight.dtype == torch.bfloat16
        ):
            w8 = getattr(lm_head, "_gb10_w8", None)
            if w8 is None and not torch.cuda.is_current_stream_capturing():
                from vllm.model_executor.layers.gb10_kernels import fp8_rows

                w = lm_head.weight.data
                parts = [fp8_rows(w[i : i + 16384]) for i in range(0, w.shape[0], 16384)]
                lm_head._gb10_w8 = w8 = torch.cat([p[0] for p in parts])
                lm_head._gb10_s = torch.cat([p[1] for p in parts])
            if w8 is not None:
                from vllm.model_executor.layers.gb10_kernels import fp8_rows_gemm

                return fp8_rows_gemm(hidden_states, w8, lm_head._gb10_s)
        if self.head_dtype is None or self.head_dtype == hidden_states.dtype:
''')])
# module-level switch next to the imports
p = os.path.join(pkg, "model_executor/layers/logits_processor.py")
s = open(p, encoding="utf-8").read()
s = s.replace("\nimport torch\n", "\nimport os\n\nimport torch\n\n_GB10_FP8_LMHEAD = os.environ.get(\"VLLM_GB10_FP8_LMHEAD\", \"1\") == \"1\"\n"
              "_GB10_FP8_LMHEAD_MAX_M = int(os.environ.get(\"VLLM_GB10_FP8_LMHEAD_MAX_M\", \"256\"))\n", 1)
assert "_GB10_FP8_LMHEAD_MAX_M = " in s
open(p, "w", encoding="utf-8").write(s)

patch("model_executor/models/qwen2_moe.py", [(
    '''        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out
''',
    '''        if self.expert_gate is not None:
            if (
                _GB10_SHARED_GATE
                and out.is_cuda
                and x.dim() == 2
                and out.dim() == 2
                and out.is_contiguous()
                and x.stride(-1) == 1
                and getattr(self.expert_gate, "bias", None) is None
                and self.expert_gate.weight.dtype == x.dtype
                and self.expert_gate.weight.shape[0] == 1
            ):
                from vllm.model_executor.layers.gb10_kernels import shared_gate_scale_

                return shared_gate_scale_(x, self.expert_gate.weight, out)
            out = F.sigmoid(self.expert_gate(x)[0]) * out
''')])
p = os.path.join(pkg, "model_executor/models/qwen2_moe.py")
s = open(p, encoding="utf-8").read()
s = s.replace("\nimport torch\n", "\nimport os\n\nimport torch\n\n_GB10_SHARED_GATE = os.environ.get(\"VLLM_GB10_SHARED_GATE\", \"1\") == \"1\"\n", 1)
assert "_GB10_SHARED_GATE = " in s
open(p, "w", encoding="utf-8").write(s)

patch("model_executor/layers/fused_moe/router/gate_linear.py", [(
    '''        can_use_specialized_kernels = (
            current_platform.is_cuda() and (is_hopper or is_blackwell) and not bias
        )''',
    '''        # GB10 / RTX Blackwell (family 120) has PDL and thread block clusters, so the cuteDSL ll_bf16 tier works
        is_sm12x = (
            os.environ.get("VLLM_GB10_ROUTER_LL", "1") == "1"
            and current_platform.is_device_capability_family(120)
        )
        can_use_specialized_kernels = (
            current_platform.is_cuda() and (is_hopper or is_blackwell or is_sm12x) and not bias
        )''')])
p = os.path.join(pkg, "model_executor/layers/fused_moe/router/gate_linear.py")
s = open(p, encoding="utf-8").read()
if "\nimport os\n" not in s:
    s = s.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)
assert "\nimport os\n" in s
open(p, "w", encoding="utf-8").write(s)

patch("model_executor/layers/linear.py", [(
    """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if current_platform.is_cpu():""",
    """    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if _gb10_fp4_weight_wanted(layer):
            from vllm.model_executor.layers.gb10_kernels import fp4_rows_chunked

            packed, gscale, rscale = fp4_rows_chunked(layer.weight.data)
            layer.weight = torch.nn.Parameter(packed, requires_grad=False)
            layer.gb10_fp4_scales = (gscale, rscale)
            torch.cuda.empty_cache()
            return
        if _gb10_fp8_weight_wanted(layer):
            from vllm.model_executor.layers.gb10_kernels import fp8_rows_chunked

            w8, scale = fp8_rows_chunked(layer.weight.data)
            layer.weight = torch.nn.Parameter(w8, requires_grad=False)
            layer.gb10_w_scale = scale
            torch.cuda.empty_cache()
            return
        if current_platform.is_cpu():"""), (
    """        return self._gemm_impl(layer, x, layer.weight, bias)""",
    """        if getattr(layer, "gb10_fp4_scales", None) is not None:
            from vllm.model_executor.layers.gb10_kernels import fp4_linear

            out = fp4_linear(x, layer.weight, *layer.gb10_fp4_scales)
            return out if bias is None else out + bias
        if getattr(layer, "gb10_w_scale", None) is not None:
            from vllm.model_executor.layers.gb10_kernels import fp8_linear

            out = fp8_linear(x, layer.weight, layer.gb10_w_scale)
            return out if bias is None else out + bias
        return self._gemm_impl(layer, x, layer.weight, bias)""")])
p = os.path.join(pkg, "model_executor/layers/linear.py")
s = open(p, encoding="utf-8").read()
s = s.replace("\nimport torch\n", """\nimport torch\n

_GB10_FP8_WEIGHTS = os.environ.get("VLLM_GB10_FP8_WEIGHTS", "0") == "1"
_GB10_FP8_WEIGHTS_SUFFIXES = tuple(
    x for x in os.environ.get(
        "VLLM_GB10_FP8_WEIGHTS_SUFFIXES",
        "in_proj_qkvz,in_proj_qkv,in_proj_z,out_proj,qkv_proj,o_proj,q_proj,gate_up_proj,down_proj,"
        "input_mix_weight_up,input_mix_weight_down,kv_proj,index_qk_proj,fc_hidden",
    ).split(",") if x
)


_GB10_FP4_WEIGHTS = os.environ.get("VLLM_GB10_FP4_WEIGHTS", "0") == "1"
_GB10_FP4_WEIGHTS_SUFFIXES = tuple(
    x for x in os.environ.get("VLLM_GB10_FP4_WEIGHTS_SUFFIXES", "in_proj_qkvz,in_proj_qkv,in_proj_z,out_proj").split(",") if x
)
# layers whose prefix contains one of these stay FP8 (the MTP draft layers: FP4 there dropped acceptance to 0)
_GB10_FP4_SKIP = tuple(x for x in os.environ.get("VLLM_GB10_FP4_SKIP", "mtp").split(",") if x)
# when set, ONLY layers whose prefix contains one of these become FP4 (e.g. VLLM_GB10_FP4_ONLY=mtp with VLLM_GB10_FP4_SKIP=none: the MTP layer alone)
_GB10_FP4_ONLY = tuple(x for x in os.environ.get("VLLM_GB10_FP4_ONLY", "").split(",") if x)


def _gb10_fp4_weight_wanted(layer) -> bool:
    # weight-only NVFP4 rows (VLLM_GB10_FP4_WEIGHTS=1): same eligibility as FP8, own suffix list, K a multiple of 16
    if not _GB10_FP4_WEIGHTS or not torch.cuda.is_available():
        return False
    prefix = getattr(layer, "prefix", "") or ""
    w = getattr(layer, "weight", None)
    return (
        _gb10_eligible(layer)
        and w.shape[1] % 16 == 0
        and prefix.split(".")[-1] in _GB10_FP4_WEIGHTS_SUFFIXES
        and not any(x in prefix for x in _GB10_FP4_SKIP)
        and (not _GB10_FP4_ONLY or any(x in prefix for x in _GB10_FP4_ONLY))
    )


def _gb10_eligible(layer) -> bool:
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    w = getattr(layer, "weight", None)
    return (
        isinstance(w, torch.Tensor)
        and w.dim() == 2
        and w.dtype == torch.bfloat16
        and w.is_cuda
        and w.numel() >= (1 << 20)
        and getattr(layer, "bias", None) is None
    )


def _gb10_fp8_weight_wanted(layer) -> bool:
    if not _GB10_FP8_WEIGHTS or not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] != 12:
        return False
    w = getattr(layer, "weight", None)
    prefix = getattr(layer, "prefix", "") or ""
    return (
        isinstance(w, torch.Tensor)
        and w.dim() == 2
        and w.dtype == torch.bfloat16
        and w.is_cuda
        and w.numel() >= (1 << 20)
        and getattr(layer, "bias", None) is None
        and prefix.split(".")[-1] in _GB10_FP8_WEIGHTS_SUFFIXES
    )
""", 1)
if "\nimport os\n" not in s:
    s = s.replace("\nimport torch\n", "\nimport os\n\nimport torch\n", 1)
assert "_gb10_fp8_weight_wanted(layer) -> bool" in s and "\nimport os\n" in s
open(p, "w", encoding="utf-8").write(s)

# vLLM 0.32 (#59753): Qwen4Exp swaps UnquantizedLinearMethod for Qwen4ExpLowLatencyLinearMethod (SM121 bf16 skinny-GEMM
# plans) on layers with a plan, and its apply calls the bf16 GEMM directly. A layer this patch already turned into FP8/FP4
# rows goes back to UnquantizedLinearMethod.apply (the gb10 branch above); bf16 layers keep the skinny GEMM.
rel = "models/qwen4_exp/nvidia/low_latency_gemm.py"
if os.path.exists(os.path.join(pkg, rel)):
    patch(rel, [(
        """        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen4_exp_low_latency_gemm(x, layer.weight)
""",
        """        if (
            bias is None
            and not envs.VLLM_BATCH_INVARIANT
            and getattr(layer, "gb10_w_scale", None) is None
            and getattr(layer, "gb10_fp4_scales", None) is None
        ):
            return torch.ops.vllm.qwen4_exp_low_latency_gemm(x, layer.weight)
""")])

# qwen38-flash-dgx only: the MTP reduced draft vocabulary head (65536 x 2560 bf16 = 335 MB read per draft step).
rel = "models/qwen4_exp/nvidia/mtp.py"
if os.path.exists(os.path.join(pkg, rel)):
    patch(rel, [(
        """        st = self._dv_state = (ids, wk, vocab)
""",
        """        head_w = wk
        if (
            _dv_os.environ.get("VLLM_GB10_FP8_DRAFT_HEAD", "1") == "1"
            and wk.is_cuda
            and wk.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(wk.device)[0] == 12
        ):
            from vllm.model_executor.layers.gb10_kernels import fp8_rows_chunked

            head_w = fp8_rows_chunked(wk)
            _dv_logger.info("MTP draft head as FP8 rows (gb10)")
        st = self._dv_state = (ids, head_w, vocab)
"""), (
        """    ids, wk, vocab = st
    red = torch.nn.functional.linear(hidden_states.to(wk.dtype), wk)
""",
        """    ids, wk, vocab = st
    if isinstance(wk, tuple):
        from vllm.model_executor.layers.gb10_kernels import fp8_linear

        red = fp8_linear(hidden_states.to(torch.bfloat16), wk[0], wk[1])
    else:
        red = torch.nn.functional.linear(hidden_states.to(wk.dtype), wk)
""")])

# qwen38-flash-dgx only: QSA took the global KV dtype and ignored --kv-cache-dtype-skip-layers (upstream vLLM #60023 fixes
# the same). With it, FP8 KV can keep the MTP layer's cache (layer 48) in bf16: FP8 MTP-layer KV cost ~20 % decode through
# lower draft acceptance (76 -> 57 % on Flash Next). Plus opt-in K/V magnitude stats and per-tensor FP8 KV scales
# (VLLM_QSA_KV_STATS / VLLM_QSA_KV_SCALE): the checkpoint ships no k/v scales, so e4m3 KV runs at scale 1.0.
# vLLM 0.32 has #60023 (qsa_kv_cache_dtype honours the skip list) and a fused K/V prepare inside the indexer launch: there
# only the layer id and the hook on the unfused path are added (the opt-in stats/scales do not run on the fused path).
rel = "models/qwen4_exp/nvidia/qsa.py"
_qsa = os.path.join(pkg, rel)
_qsa_v032 = os.path.exists(_qsa) and "qsa_kv_cache_dtype(cache_config, prefix)" in open(_qsa, encoding="utf-8").read()
if _qsa_v032:
    _qsa_dtype = (
        """        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
""",
        """        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
        self._gb10_layer_id = extract_layer_index(prefix)
""")
    _qsa_update = (
        """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            impl.do_kv_cache_update(
""",
        """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            if _GB10_KV_STATS or _GB10_KV_SCALE:
                _gb10_kv_hook(self, key, value, num_tokens)
            impl.do_kv_cache_update(
""")
else:
    _qsa_dtype = (
        """        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(""",
        """        self.kv_cache_dtype = cache_config.cache_dtype
        _skip = getattr(cache_config, "kv_cache_dtype_skip_layers", None) or []
        if _skip and str(layer_id) in [str(x) for x in _skip]:
            self.kv_cache_dtype = "auto"  # listed layer: bf16 cache (--kv-cache-dtype-skip-layers)
        self._gb10_layer_id = layer_id
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(""")
    _qsa_update = (
        """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
""",
        """        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if _GB10_KV_STATS or _GB10_KV_SCALE:
            _gb10_kv_hook(self, key, value, num_tokens)
        impl.do_kv_cache_update(
""")
if os.path.exists(_qsa):
    patch(rel, [_qsa_dtype, (
        """class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
""",
        """_GB10_KV_STATS = __import__("os").environ.get("VLLM_QSA_KV_STATS", "0") == "1"
_GB10_KV_SCALE = __import__("os").environ.get("VLLM_QSA_KV_SCALE", "").strip()


def _gb10_kv_scale_for(layer_id):
    if _GB10_KV_SCALE.startswith("{"):
        import json

        v = json.loads(_GB10_KV_SCALE).get(str(layer_id))
        return None if v is None else (float(v[0]), float(v[1]))
    k, v = _GB10_KV_SCALE.split(",")
    return float(k), float(v)


def _gb10_kv_hook(layer, key, value, num_tokens):
    \"\"\"Opt-in FP8-KV diagnostics and scales for the QSA layers (gb10). The checkpoint has no k/v scales, so the e4m3
    cache runs at scale 1.0. VLLM_QSA_KV_STATS=1 logs K/V magnitudes per prefill chunk; VLLM_QSA_KV_SCALE="k,v" or
    '{"3": [k, v], ...}' sets the per-tensor scales (stored = x / scale, dequantized in the QSA kernel).\"\"\"
    from vllm.logger import init_logger

    log = init_logger(__name__)
    if _GB10_KV_SCALE and not getattr(layer, "_gb10_kv_scaled", False):
        layer._gb10_kv_scaled = True
        sc = _gb10_kv_scale_for(layer._gb10_layer_id)
        if sc is not None:
            layer._k_scale.fill_(sc[0])
            layer._v_scale.fill_(sc[1])
            layer._k_scale_float, layer._v_scale_float = sc
            log.info("QSA layer %s fp8 KV scales k=%g v=%g (gb10)", layer._gb10_layer_id, sc[0], sc[1])
    if _GB10_KV_STATS and num_tokens >= 1024:
        def q(x):
            f = x[:num_tokens].float().abs().flatten()
            if f.numel() > 4_000_000:
                f = f[:: f.numel() // 4_000_000 + 1]
            return (f.max().item(), torch.quantile(f, 0.999).item(), (f < 2**-6).float().mean().item(),
                    (f < 2**-9).float().mean().item())
        log.info("QSA-KVSTATS layer %s n %d K max %.4g p999 %.4g sub %.4f tiny %.4f V max %.4g p999 %.4g sub %.4f tiny %.4f",
                 layer._gb10_layer_id, num_tokens, *q(key), *q(value))


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
"""), _qsa_update])
