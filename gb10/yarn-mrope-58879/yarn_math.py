"""#58870 check: YaRN inverse frequencies for Qwen3.8 Flash Next, vLLM 0.30 (correction range from 4x max_position) vs the HF/PR formula (original_max_position_embeddings).
Pure math, no vllm/torch needed. Mirrors vllm yarn_find_correction_range / yarn_linear_ramp_mask / YaRNScalingRotaryEmbedding._compute_inv_freq."""
import math
ROT_DIM = int(256 * 0.25); BASE = 1e7; BETA_FAST, BETA_SLOW = 32, 1; ORIG = 262144
def corr_dim(n_rot, dim, base, maxpos): return dim * math.log(maxpos / (n_rot * 2 * math.pi)) / (2 * math.log(base))
def corr_range(maxpos):
    lo = math.floor(corr_dim(BETA_FAST, ROT_DIM, BASE, maxpos)); hi = math.ceil(corr_dim(BETA_SLOW, ROT_DIM, BASE, maxpos))
    return max(lo, 0), min(hi, ROT_DIM - 1)
def inv_freq(factor, maxpos):
    lo, hi = corr_range(maxpos); n = ROT_DIM // 2
    out = []
    for i in range(n):
        pos = BASE ** (2 * i / ROT_DIM); ext = 1 / pos; itp = 1 / (factor * pos)
        ramp = min(max((i - lo) / (hi - lo if hi != lo else 0.001), 0), 1); mask = 1 - ramp
        out.append(itp * (1 - mask) + ext * mask)
    return out
print("rotary_dim=%d -> %d inverse frequencies, base=%g" % (ROT_DIM, ROT_DIM // 2, BASE))
for ctx, f in ((524288, 2), (786432, 3), (1048576, 4)):
    good = inv_freq(f, ORIG); bad = inv_freq(f, ORIG * 4)
    rel = [abs(b - g) / g for g, b in zip(good, bad)]
    diff = [i for i, r in enumerate(rel) if r > 1e-9]
    print("ctx %7d factor %d: correct range %s, 0.30 range %s; %d of 32 freqs differ, max rel err %.1f%% (idx %d), wavelength of those: %s" % (
        ctx, f, corr_range(ORIG), corr_range(ORIG * 4), len(diff), 100 * max(rel), rel.index(max(rel)),
        "%.0f..%.0f tokens" % (2 * math.pi / good[diff[0]], 2 * math.pi / good[diff[-1]]) if diff else "-"))
