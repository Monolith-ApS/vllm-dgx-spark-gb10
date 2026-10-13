"""Correctness + speed of the weight-only NVFP4 rows kernel against the FP8 rows kernel (CUDA graphs).

  bench_fp4.py [quick]
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb10_kernels import fp8_rows, fp8_rows_gemm
from gb10_kernels import fp4_rows, fp4_rows_gemm, dequant_fp4_rows, dequant_fp4_rows_ref


def graph_time(fn, iters=48):
    """fn(i) is called with i = 0..iters-1, so a benchmark can cycle through weight copies larger than L2 together."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(iters):
            fn(i)
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / (5 * iters) * 1000


torch.manual_seed(0)
dev = "cuda"
quick = len(sys.argv) > 1
# Flash Next shapes: GDN in_proj_qkv 2560->12288-ish, in_proj_z, out_proj, attention q/o, hyperconnection up (320->10240)
shapes = [(12288, 2560), (8192, 2560), (2560, 8192), (10240, 320), (2560, 4096)]
for N, K in shapes:
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    w[:, :8] *= 8  # some outliers
    q, gs, rs = fp4_rows(w)
    C = max(2, int(160e6 // (N * K)))  # copies, > 6x L2 for fp8
    Q = [(q.clone(), gs.clone(), rs.clone()) for _ in range(C)]
    deq = dequant_fp4_rows(q, gs, rs)
    ref_deq = dequant_fp4_rows_ref(q, gs, rs)
    assert torch.equal(deq, ref_deq), "triton dequant != reference dequant"
    qerr = ((deq.float() - w.float()).norm() / w.float().norm()).item()
    w8, s8 = fp8_rows(w)
    W8 = [(w8.clone(), s8.clone()) for _ in range(C)]
    e8 = ((w8.float() * s8[:, None] - w.float()).norm() / w.float().norm()).item()
    print(f"[{N}x{K}] weight rel. error: fp4 {qerr:.4f}  fp8 {e8:.4f}")
    for M in ((1, 4) if quick else (1, 2, 4, 7, 16, 32, 64, 256)):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        ref = F.linear(x, deq)
        got = fp4_rows_gemm(x, q, gs, rs)
        err = ((got.float() - ref.float()).abs().max() / ref.float().abs().max()).item()
        t4 = graph_time(lambda i: fp4_rows_gemm(x, *Q[i % C]))
        t8 = graph_time(lambda i: fp8_rows_gemm(x, *W8[i % C]))
        b4 = N * K * 0.5625
        print(f"  M={M:3d}: fp4 {t4:7.1f} us ({b4/t4/1e3:4.0f} GB/s)  fp8 {t8:7.1f} us ({N*K/t8/1e3:4.0f} GB/s)  "
              f"speedup {t8/t4:4.2f}x  kernel err vs dequant {err:.5f}")
if not quick:
    # config sweep for M=1/4 on the largest shape
    from gb10_kernels import CONFIGS
    N, K = 12288, 2560
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    q, gs, rs = fp4_rows(w)
    C = 6
    Q = [(q.clone(), gs.clone(), rs.clone()) for _ in range(C)]
    for M in (1, 4, 16):
        x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        res = []
        for bn in (32, 64, 128):
            for bk in (128, 256, 512):
                for nw in (2, 4, 8):
                    for st in (2, 3, 4):
                        cfg = dict(BLOCK_M=16, BLOCK_N=bn, BLOCK_K=bk, num_warps=nw, num_stages=st)
                        try:
                            t = graph_time(lambda i: fp4_rows_gemm(x, *Q[i % C], cfg=cfg), iters=24)
                        except Exception as e:  # shared memory / register limits
                            continue
                        res.append((t, cfg))
        res.sort(key=lambda r: r[0])
        for t, cfg in res[:4]:
            print(f"  sweep M={M}: {t:6.1f} us ({N*K*0.5625/t/1e3:4.0f} GB/s) {cfg}")
