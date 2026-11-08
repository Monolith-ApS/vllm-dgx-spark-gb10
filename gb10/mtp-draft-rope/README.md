# mtp-draft-rope

vLLM 0.32 and 0.33 (`apply_mtp_rope_v033.py` for 0.33's code). Gives the speculative MTP draft the same YaRN rope settings as the target model, so contexts above
262,144 tokens work with MTP.

## Symptom

On stock vLLM 0.32.0 a model that is served beyond its native 262,144-token window (YaRN through a dict `--hf-overrides`)
and uses MTP crashes with `CUDA error: an illegal memory access` as soon as a conversation passes ~265K tokens. The engine
core dies; the API server keeps running, so `/health` turns 503 and every request returns HTTP 500. Short benchmarks do not
show it; only a conversation that grows past ~265K tokens does. Reported as
[vllm-project/vllm#62383](https://github.com/vllm-project/vllm/issues/62383). vLLM 0.31 does not have it.

## Cause

Dict overrides reach only the target model. The MTP draft builds plain M-RoPE from `config.json` (262,144 positions), and
0.32 sizes the M-RoPE cos/sin table by that number (0.31 made it 4x larger), so the first draft step past position 262,144
reads outside the table.

## The fix

`apply_mtp_rope.py <site-packages>/vllm` changes `vllm/config/speculative.py`: with method `mtp` and a draft that is the target
checkpoint itself, the draft gets the target's dict overrides before its MTP conversion. Its rope is then the same YaRN as the
target's and its table covers `max_model_len`. Any other draft keeps the stock rule. The script refuses an anchor that does not
match exactly once, so a changed vLLM fails the build. The server logs `GB10: MTP draft gets the target's hf_overrides` at
start.

Upstream [vllm-project/vllm#62409](https://github.com/vllm-project/vllm/pull/62409) fixes the crash the other way: it makes the
table large enough and leaves the draft on plain RoPE. The two do not conflict. Once a vLLM release contains a fix, this layer
can go.

## Tested

Qwen3.8 Flash Next on vLLM 0.32.0, built from this repository (2026-10-25): all eight profiles from `128k-c8` to `1024k-c1`
filled to the full context with 0 preemptions, tool calling 18/18, decode at the speed of 0.31.0
([numbers](../../models/qwen3.8-flash-next/measurements.md)). Without this layer every profile
above 256K crashed at ~265K tokens (8 of 8 runs).

Qwen3.8 Flash Next on vLLM 0.33.0, built from this repository (2026-11-07): `640k-c2` and `1024k-c1` filled to 636K x 2 and
1,028K x 1 with 0 preemptions (the other profiles' KV cache is too small on 0.33.0 for a full fill, see
[vllm-versions](../../docs/vllm-versions.md#what-we-found-in-vllm-0330)).
