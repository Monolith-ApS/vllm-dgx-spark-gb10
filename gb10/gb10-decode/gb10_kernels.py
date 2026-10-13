"""GB10 decode kernels for vLLM (vllm-dgx-spark-gb10).

fp8_lm_head_*      the LM head as FP8 e4m3 rows with one fp32 scale per row: decode reads half the bytes of bf16
shared_gate_scale_ out *= sigmoid(x @ w_gate.T) in one kernel (Qwen2/3 MoE shared expert gate: gemv to 1 + sigmoid + mul)
"""
import torch
import triton
import triton.language as tl

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0


def fp8_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16/fp32 [N, K] -> (fp8 [N, K], fp32 scale [N]) with w ~= w8 * scale[:, None]."""
    amax = w.float().abs().amax(dim=1).clamp(min=1e-12)
    scale = amax / FP8_MAX
    w8 = (w.float() / scale[:, None]).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    return w8.contiguous(), scale.contiguous()


@triton.jit
def _fp8_rows_gemm(x_ptr, w_ptr, s_ptr, o_ptr, M, N, K, sxm, swn, som,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(w_ptr + rn[:, None] * swn + kk[None, :], mask=(rn[:, None] < N) & (kk[None, :] < K), other=0.0,
                    eviction_policy="evict_first")
        acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc)
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * s[None, :]
    tl.store(o_ptr + rm[:, None] * som + rn[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _dequant_rows(w_ptr, s_ptr, o_ptr, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    m = (rn[:, None] < N) & (rk[None, :] < K)
    w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=m, other=0.0).to(tl.float32)
    s = tl.load(s_ptr + rn, mask=rn < N, other=0.0)
    tl.store(o_ptr + rn[:, None] * K + rk[None, :], (w * s[:, None]).to(tl.bfloat16), mask=m)


def dequant_rows(w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    N, K = w8.shape
    out = torch.empty((N, K), dtype=torch.bfloat16, device=w8.device)
    _dequant_rows[(triton.cdiv(N, 64), triton.cdiv(K, 256))](w8, scale, out, N, K, BLOCK_N=64, BLOCK_K=256, num_warps=4)
    return out


# Above this many rows the GEMM is compute bound: dequantize the weight to bf16 once and let cuBLAS run it
# (Triton's mma.sync path reaches ~48 TFLOPS on GB10, cuBLAS ~95).
DEQUANT_MIN_M = 768


def fp8_rows_gemm(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor, out_dtype=None) -> torch.Tensor:
    """x [M, K] bf16 @ (w8 * scale)[N, K].T -> [M, N]."""
    x = x.contiguous()
    M, K = x.shape
    N = w8.shape[0]
    if M >= DEQUANT_MIN_M:
        out = torch.nn.functional.linear(x, dequant_rows(w8, scale))
        return out if out_dtype in (None, out.dtype) else out.to(out_dtype)
    out = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    if M <= 16:
        cfg = dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=256, num_warps=4, num_stages=4)
    elif M <= 64:
        cfg = dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=256, num_warps=4, num_stages=2)
    else:
        cfg = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3)
    grid = (triton.cdiv(N, cfg["BLOCK_N"]), triton.cdiv(M, cfg["BLOCK_M"]))
    _fp8_rows_gemm[grid](x, w8, scale, out, M, N, K, x.stride(0), w8.stride(0), out.stride(0), **cfg)
    return out


@triton.jit
def _shared_gate_scale(x_ptr, w_ptr, o_ptr, H, D, sxm, som, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for h0 in range(0, H, BLOCK):
        r = h0 + tl.arange(0, BLOCK)
        xv = tl.load(x_ptr + m * sxm + r, mask=r < H, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + r, mask=r < H, other=0.0).to(tl.float32)
        acc += xv * wv
    g = tl.sigmoid(tl.sum(acc, axis=0))
    for d0 in range(0, D, BLOCK):
        r = d0 + tl.arange(0, BLOCK)
        ov = tl.load(o_ptr + m * som + r, mask=r < D, other=0.0).to(tl.float32)
        tl.store(o_ptr + m * som + r, (ov * g).to(o_ptr.dtype.element_ty), mask=r < D)


def shared_gate_scale_(x: torch.Tensor, w_gate: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """In place: out[m] *= sigmoid(x[m] . w_gate) for x [M, H], w_gate [1, H], out [M, D]."""
    M, H = x.shape
    if M == 0:
        return out
    _shared_gate_scale[(M,)](x, w_gate, out, H, out.shape[1], x.stride(0), out.stride(0), BLOCK=1024, num_warps=4)
    return out


def fp8_rows_chunked(w: torch.Tensor, rows: int = 16384) -> tuple[torch.Tensor, torch.Tensor]:
    """fp8_rows in row chunks, so the fp32 temporaries stay small for a large matrix."""
    parts = [fp8_rows(w[i : i + rows]) for i in range(0, w.shape[0], rows)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def _fp8_linear(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # the kernel multiplies in bf16; an fp32 caller (hyperconnection low-rank path) gets its dtype back
    x2 = x.reshape(-1, x.shape[-1])
    out = fp8_rows_gemm(x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16), w8, scale, out_dtype=x.dtype)
    return out.reshape(*x.shape[:-1], w8.shape[0])


def _fp8_linear_fake(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], w8.shape[0]))


# --- weight-only NVFP4 rows (VLLM_GB10_FP4_WEIGHTS) ------------------------------------------------------------------
# bf16 weights re-quantized at load to e2m1 nibbles + an fp8 e4m3 scale per 16 weights + an fp32 scale per row:
# 0.5625 bytes per weight (FP8 rows: 1.0). Nibble layout: low nibble = even k; code = sign << 3 | exponent << 1 | mantissa.

E2M1_MAX = 6.0
FP8_MAX = 448.0
GROUP = 16
# |v| thresholds between the e2m1 magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6 (round to nearest)
_THRESH = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


# group scale = gmax * c / 6 for each c, per group the c with the least squared error (real Flash Next GDN weights:
# relative error 9.4 -> 8.8 %)
FP4_SCALE_SEARCH = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75)


def fp4_rows(w: torch.Tensor):
    N, K = w.shape
    assert K % GROUP == 0, "K must be a multiple of 16"
    wf = w.float().view(N, K // GROUP, GROUP)
    gmax = wf.abs().amax(dim=2)  # [N, K/16]
    rscale = (gmax.amax(dim=1) / (E2M1_MAX * FP8_MAX)).clamp(min=1e-30)  # [N]
    thresh = torch.tensor(_THRESH, device=w.device)
    lut = torch.tensor(_E2M1, device=w.device)
    best = None
    for c in FP4_SCALE_SEARCH:
        gscale = (gmax * c / (E2M1_MAX * rscale[:, None])).clamp(max=FP8_MAX).to(FP8)  # [N, K/16]
        denom = gscale.float() * rscale[:, None]
        v = wf.abs() / torch.where(denom == 0, torch.ones_like(denom), denom)[:, :, None]
        mag = torch.bucketize(v, thresh)  # 0..7
        err = ((lut[mag] - v) * denom[:, :, None]).square().sum(dim=2)
        if best is None:
            best = [err, gscale, mag]
        else:
            m = err < best[0]
            best = [torch.where(m, err, best[0]), torch.where(m, gscale.view(torch.uint8), best[1].view(torch.uint8)).view(FP8),
                    torch.where(m[:, :, None], mag, best[2])]
    _, gscale, mag = best
    code = (mag | ((wf < 0).to(torch.int64) << 3)).to(torch.uint8).view(N, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()
    return packed, gscale.contiguous(), rscale.float().contiguous()


def fp4_rows_chunked(w: torch.Tensor, rows: int = 4096):
    parts = [fp4_rows(w[i : i + rows]) for i in range(0, w.shape[0], rows)]
    return tuple(torch.cat([p[j] for p in parts]) for j in range(3))


def dequant_fp4_rows_ref(packed, gscale, rscale):
    N = packed.shape[0]
    lut = torch.tensor(_E2M1, device=packed.device)
    lo, hi = packed & 0xF, packed >> 4
    code = torch.stack((lo, hi), dim=2).view(N, -1).long()
    val = lut[code & 7] * torch.where(code >= 8, -1.0, 1.0)
    val = val.view(N, -1, GROUP) * gscale.float()[:, :, None] * rscale[:, None, None]
    return val.view(N, -1).to(torch.bfloat16)


@triton.jit
def _decode_tile(p, s_ptr, rn, k0, N, K, sws, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    lo = (p & 0xF).to(tl.int32)
    hi = (p >> 4).to(tl.int32)
    nib = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))  # [n, k]: even k = low nibble
    e = (nib >> 1) & 3
    m = nib & 1
    bits = tl.where(e == 0, m * (126 << 7), ((e + 126) << 7) | (m << 6)) | ((nib & 8) << 12)
    w = bits.to(tl.int16).to(tl.bfloat16, bitcast=True)
    rg = tl.arange(0, BLOCK_K // 16)
    kg = k0 // 16 + rg  # K need not be a multiple of BLOCK_K: the tail groups must not be read (fp8 NaN * 0 = NaN)
    gs = tl.load(s_ptr + rn[:, None] * sws + kg[None, :], mask=(rn[:, None] < N) & (kg[None, :] < K // 16), other=0.0)
    w = tl.reshape(w, (BLOCK_N, BLOCK_K // 16, 16)) * gs.to(tl.bfloat16)[:, :, None]
    return tl.reshape(w, (BLOCK_N, BLOCK_K))


@triton.jit
def _fp4_rows_gemm(x_ptr, q_ptr, s_ptr, r_ptr, o_ptr, M, N, K, sxm, sqn, sws, som,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)  # M-major: the programs that share a weight tile run next to each other (L2 hit)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    rk2 = tl.arange(0, BLOCK_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + rk
        x = tl.load(x_ptr + rm[:, None] * sxm + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        kb = k0 // 2 + rk2
        p = tl.load(q_ptr + rn[:, None] * sqn + kb[None, :], mask=(rn[:, None] < N) & (kb[None, :] < K // 2), other=0,
                    eviction_policy="evict_first")
        w = _decode_tile(p, s_ptr, rn, k0, N, K, sws, BLOCK_N, BLOCK_K)
        acc = tl.dot(x, tl.trans(w), acc)
    r = tl.load(r_ptr + rn, mask=rn < N, other=0.0)
    acc = acc * r[None, :]
    tl.store(o_ptr + rm[:, None] * som + rn[None, :], acc.to(o_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _dequant_fp4(q_ptr, s_ptr, r_ptr, o_ptr, N, K, sqn, sws, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = tl.program_id(1) * BLOCK_K
    kb = k0 // 2 + tl.arange(0, BLOCK_K // 2)
    p = tl.load(q_ptr + rn[:, None] * sqn + kb[None, :], mask=(rn[:, None] < N) & (kb[None, :] < K // 2), other=0)
    w = _decode_tile(p, s_ptr, rn, k0, N, K, sws, BLOCK_N, BLOCK_K).to(tl.float32)
    r = tl.load(r_ptr + rn, mask=rn < N, other=0.0)
    kk = k0 + tl.arange(0, BLOCK_K)
    tl.store(o_ptr + rn[:, None] * K + kk[None, :], (w * r[:, None]).to(tl.bfloat16),
             mask=(rn[:, None] < N) & (kk[None, :] < K))


def dequant_fp4_rows(packed, gscale, rscale):
    N = packed.shape[0]
    K = packed.shape[1] * 2
    out = torch.empty((N, K), dtype=torch.bfloat16, device=packed.device)
    _dequant_fp4[(triton.cdiv(N, 64), triton.cdiv(K, 256))](packed, gscale, rscale, out, N, K, packed.stride(0),
                                                           gscale.stride(0), BLOCK_N=64, BLOCK_K=256, num_warps=4)
    return out


FP4_DEQUANT_MIN_M = 768
FP4_CONFIGS = {  # M bucket -> launch config; tuned on GB10 (rotating weight copies past L2, M-major grid)
    8: dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=512, num_warps=8, num_stages=3),  # N < 8192 (more programs than BN 32)
    16: dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=512, num_warps=4, num_stages=2),
    24: dict(BLOCK_M=16, BLOCK_N=64, BLOCK_K=256, num_warps=8, num_stages=3),
    32: dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=4),
    64: dict(BLOCK_M=32, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3),
    768: dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=64, num_warps=8, num_stages=3),
}


def _fp4_config(M, N):
    if M <= 16:
        return FP4_CONFIGS[16] if N >= 8192 else FP4_CONFIGS[8]
    for b in (24, 32, 64):
        if M <= b:
            return FP4_CONFIGS[b]
    return FP4_CONFIGS[768]


def fp4_rows_gemm(x, packed, gscale, rscale, out_dtype=None, cfg=None):
    x = x.contiguous()
    M, K = x.shape
    N = packed.shape[0]
    if M >= FP4_DEQUANT_MIN_M:
        out = torch.nn.functional.linear(x, dequant_fp4_rows(packed, gscale, rscale))
        return out if out_dtype in (None, out.dtype) else out.to(out_dtype)
    out = torch.empty((M, N), dtype=out_dtype or x.dtype, device=x.device)
    if cfg is None:
        cfg = _fp4_config(M, N)
    grid = (triton.cdiv(M, cfg["BLOCK_M"]), triton.cdiv(N, cfg["BLOCK_N"]))
    _fp4_rows_gemm[grid](x, packed, gscale, rscale, out, M, N, K, x.stride(0), packed.stride(0), gscale.stride(0),
                         out.stride(0), **cfg)
    return out


def _fp4_linear(x: torch.Tensor, packed: torch.Tensor, gscale: torch.Tensor, rscale: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    out = fp4_rows_gemm(x2 if x2.dtype == torch.bfloat16 else x2.to(torch.bfloat16), packed, gscale, rscale,
                        out_dtype=x.dtype)
    return out.reshape(*x.shape[:-1], packed.shape[0])


def _fp4_linear_fake(x, packed, gscale, rscale):
    return x.new_empty((*x.shape[:-1], packed.shape[0]))

def _register() -> bool:
    """Register the custom op when this module is imported (at weight loading), never lazily inside torch.compile tracing."""
    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:  # standalone use (bench_kernels.py)
        return False
    if not hasattr(torch.ops.vllm, "gb10_fp8_linear"):
        direct_register_custom_op(op_name="gb10_fp8_linear", op_func=_fp8_linear, fake_impl=_fp8_linear_fake)
    if not hasattr(torch.ops.vllm, "gb10_fp4_linear"):
        direct_register_custom_op(op_name="gb10_fp4_linear", op_func=_fp4_linear, fake_impl=_fp4_linear_fake)
    return True


_REGISTERED = _register()


def fp8_linear(x: torch.Tensor, w8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """FP8-row weight linear as a vLLM custom op (opaque to torch.compile, capturable in CUDA graphs)."""
    if _REGISTERED:
        return torch.ops.vllm.gb10_fp8_linear(x, w8, scale)
    return _fp8_linear(x, w8, scale)


def fp4_linear(x: torch.Tensor, packed: torch.Tensor, gscale: torch.Tensor, rscale: torch.Tensor) -> torch.Tensor:
    """Weight-only NVFP4 rows linear as a vLLM custom op."""
    if _REGISTERED:
        return torch.ops.vllm.gb10_fp4_linear(x, packed, gscale, rscale)
    return _fp4_linear(x, packed, gscale, rscale)
