"""Microbenchmark of the GB10 decode kernels against what vLLM runs today (CUDA graphs, so launch cost is excluded)."""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb10_kernels import fp8_rows, fp8_rows_gemm, shared_gate_scale_


def graph_time(fn, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / (5 * iters) * 1000  # us


torch.manual_seed(0)
dev = "cuda"
V, H = 151936, 2048
w = (torch.randn(V, H, device=dev) * 0.02).to(torch.bfloat16)
w8, sc = fp8_rows(w)
for M in (1, 4, 16, 64):
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ref = F.linear(x, w)
    got = fp8_rows_gemm(x, w8, sc)
    err = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
    agree = (got.float().argmax(-1) == ref.float().argmax(-1)).float().mean().item()
    tb = graph_time(lambda: F.linear(x, w))
    tf = graph_time(lambda: fp8_rows_gemm(x, w8, sc))
    print(f"lm_head M={M:3d}: bf16 cuBLAS {tb:7.1f} us ({V*H*2/tb/1e3:5.0f} GB/s)  fp8 triton {tf:7.1f} us "
          f"({V*H/tf/1e3:5.0f} GB/s)  rel.err {err:.4f}  argmax agree {agree:.3f}")

# prefill-sized GEMMs with FP8 weights (Flash Next GDN in_proj: 2560 -> 12288)
Kp, Np = 2560, 12288
wp = (torch.randn(Np, Kp, device=dev) * 0.02).to(torch.bfloat16)
wp8, sp = fp8_rows(wp)
for M in (1, 4, 64, 65, 256, 512, 1023, 1024, 8192):
    x = torch.randn(M, Kp, device=dev, dtype=torch.bfloat16)
    tb = graph_time(lambda: F.linear(x, wp), iters=10)
    tf = graph_time(lambda: fp8_rows_gemm(x, wp8, sp), iters=10)
    fl = 2 * M * Kp * Np
    print(f"gemm {Np}x{Kp} M={M:5d}: bf16 cuBLAS {tb:8.1f} us ({fl/tb/1e6:6.1f} TFLOPS, {Np*Kp*2/tb/1e3:4.0f} GB/s)  "
          f"fp8 triton {tf:8.1f} us ({fl/tf/1e6:6.1f} TFLOPS, {Np*Kp/tf/1e3:4.0f} GB/s)")

# router gate: bf16 x bf16 -> fp32, 2048 -> 512, as GateLinear's cuBLAS tier does it
E = 512
wg = (torch.randn(E, H, device=dev) * 0.02).to(torch.bfloat16)
for M in (1, 4):
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    tr = graph_time(lambda: torch.mm(x, wg.t(), out_dtype=torch.float32))
    tl_ = graph_time(lambda: F.linear(x, wg))
    print(f"router M={M}: mm out fp32 {tr:6.1f} us   F.linear bf16 {tl_:6.1f} us")

# shared expert gate: today gemv(H->1) + sigmoid + mul; fused kernel
wsg = (torch.randn(1, H, device=dev) * 0.02).to(torch.bfloat16)
for M in (1, 4):
    x = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    out = torch.randn(M, H, device=dev, dtype=torch.bfloat16)
    ref = torch.sigmoid(F.linear(x, wsg)) * out
    o2 = out.clone(); shared_gate_scale_(x, wsg, o2)
    err = (o2.float() - ref.float()).abs().max().item()
    t3 = graph_time(lambda: torch.sigmoid(F.linear(x, wsg)) * out)
    o3 = out.clone()
    tf = graph_time(lambda: shared_gate_scale_(x, wsg, o3))
    print(f"shared gate M={M}: 3 ops {t3:6.1f} us   fused {tf:6.1f} us   max err {err:.5f}")
