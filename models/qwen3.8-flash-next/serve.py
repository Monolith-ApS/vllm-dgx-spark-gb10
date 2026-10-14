#!/usr/bin/env python3
"""Entrypoint: start vLLM for Qwen3.8 Flash Next with one of the profiles in profiles.json.

Environment:
  PROFILE        profile name (default 512k-c3); a -think suffix (512k-c3-think) also sets THINKING=1;
                 PROFILE=list prints the profiles and exits
  MODEL          model directory inside the container (default /models/Qwen3.8-Flash-Next-NVFP4)
  SERVED_NAME    model name in the API (default qwen3.8-flash-next)
  THINKING       0 (default) or 1: default value of enable_thinking for requests that do not set it
  PORT           listen port (default 8000)
  API_KEY        optional; passed to vLLM as --api-key
  GMU            override the profile's gpu_memory_utilization
  DRY_RUN=1      print the vLLM command line and environment instead of starting it
  LOAD_FORMAT    weight loader (default instanttensor; safetensors also works)
  MTP            speculative draft tokens (default 7, with thinking 3)

Extra arguments to the container are appended to the vLLM command line (later flags win).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE_CONTEXT = 262144

SPLITTING_OPS = [
    "vllm::unified_attention_with_output", "vllm::unified_mla_attention_with_output", "vllm::mamba_mixer2", "vllm::mamba_mixer",
    "vllm::short_conv", "vllm::qwen4_exp_ple_short_conv", "vllm::qwen4_exp_qsa_with_output", "vllm::linear_attention",
    "vllm::qwen_gdn_attention_core", "vllm::qwen_gdn_attention_core_fused_norm_packed", "vllm::gdn_attention_core_xpu",
    "vllm::olmo_hybrid_gdn_full_forward", "vllm::sparse_attn_indexer", "vllm::rocm_aiter_sparse_attn_indexer",
    "vllm::deepseek_v4_attention", "vllm::hpc_rope_norm_forward", "vllm::unified_kv_cache_update",
    "vllm::unified_mla_kv_cache_update", "vllm::ple_mmap_lookup_ids",
]

BASE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    # no telemetry
    "DO_NOT_TRACK": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "1",
    # overlay switches
    "VLLM_PLE_MMAP": "1", "VLLM_PLE_MMAP_WORKERS": "32", "VLLM_PLE_MMAP_MADVISE": "random", "VLLM_PLE_MMAP_PREWARM": "0",
    "VLLM_PLE_MMAP_FAST_ROWS": "0",
    "VLLM_QSA_DET_TOPK": "1", "VLLM_QSA_DET_LIB": "/opt/llm/kernel-det/_C_det.so", "VLLM_QSA_EXACT_TOPK": "0",
    "VLLM_MTP_DRAFT_VOCAB": "/opt/llm/draft_vocab_65536.npy",
    "VLLM_FP8_PAD_M4": "0",
    # GB10 decode layer: bf16 linears -> FP8 rows, GDN projections -> NVFP4 rows (LM head FP8, shared gate, router tier are on by default)
    "VLLM_GB10_FP8_WEIGHTS": "1",
    "VLLM_GB10_FP4_WEIGHTS": "1",
}

SAMPLING = {
    False: {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    True: {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.0},
}


def compact(obj):
    return json.dumps(obj, separators=(",", ":"))


def main():
    profiles = json.load(open(os.path.join(HERE, "profiles.json"), encoding="utf-8"))["profiles"]
    name = os.environ.get("PROFILE", "512k-c3")
    if name == "list":
        for k, p in profiles.items():
            print("%-10s context %7d  concurrency %d  gmu %.2f  %s" % (k, p["context"], p["concurrency"], p["gpu_memory_utilization"], p.get("status", "")))
        return 0
    thinking = os.environ.get("THINKING", "0") == "1"
    if name.endswith("-think"):
        name, thinking = name[:-len("-think")], True
    if name not in profiles:
        sys.exit("unknown PROFILE %r; choose one of: %s (each also as <name>-think)" % (name, ", ".join(profiles)))
    p = profiles[name]
    ctx, seqs = int(p["context"]), int(p["concurrency"])
    gmu = float(os.environ.get("GMU") or p["gpu_memory_utilization"])
    model = os.environ.get("MODEL", "/models/Qwen3.8-Flash-Next-NVFP4")
    served = os.environ.get("SERVED_NAME", "qwen3.8-flash-next")
    if not os.path.isfile(os.path.join(model, "config.json")):
        sys.exit("no model at %s (mount the model directory, see scripts/get-model.sh)" % model)

    # MTP draft depth: 7 without thinking, 3 with thinking (reasoning text accepts fewer draft tokens); MTP overrides both
    spec = {"method": "mtp", "num_speculative_tokens": int(os.environ.get("MTP") or (3 if thinking else 7))}
    argv = ["vllm", "serve", model, "--served-model-name", served, "--host", "0.0.0.0", "--port", os.environ.get("PORT", "8000"),
            "--max-model-len", str(ctx), "--max-num-seqs", str(seqs), "--gpu-memory-utilization", "%.2f" % gmu,
            "--max-num-batched-tokens", str(p.get("max_num_batched_tokens", 4096)),
            "--enable-prefix-caching", "--enable-chunked-prefill",
            "-cc.cudagraph_mode=PIECEWISE", "-cc.splitting_ops=" + compact(SPLITTING_OPS), "--no-enable-flashinfer-autotune",
            "--kv-cache-dtype", "fp8_e4m3", "--kv-cache-dtype-skip-layers", "48",
            "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3",
            "--moe-backend", "marlin", "--block-size", "1680", "--load-format", os.environ.get("LOAD_FORMAT", "instanttensor"),
            "--override-generation-config", compact(SAMPLING[thinking]),
            "--default-chat-template-kwargs", compact({"enable_thinking": thinking})]
    if p.get("mamba_cache_mode"):
        argv += ["--mamba-cache-mode", p["mamba_cache_mode"]]
    env = dict(BASE_ENV)
    if ctx > NATIVE_CONTEXT:  # YaRN above the native context
        factor = ctx / NATIVE_CONTEXT
        factor = int(factor) if factor == int(factor) else round(factor, 3)
        argv += ["--hf-overrides", compact({"text_config": {"rope_parameters": {
            "mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000,
            "partial_rotary_factor": 0.25, "factor": float(factor), "original_max_position_embeddings": NATIVE_CONTEXT}}})]
        spec["max_model_len"] = ctx
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    argv += ["--speculative-config", compact(spec)]
    if os.environ.get("API_KEY"):
        argv += ["--api-key", os.environ["API_KEY"]]
    argv += sys.argv[1:]
    for k, v in env.items():
        os.environ.setdefault(k, v)
    if os.environ.get("DRY_RUN") == "1":
        print(" ".join(argv))
        print(json.dumps({k: os.environ[k] for k in env}, indent=1))
        return 0
    print("profile %s: context %d, concurrency %d, gmu %.2f, thinking %s" % (name, ctx, seqs, gmu, "on" if thinking else "off"), flush=True)
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    sys.exit(main())
