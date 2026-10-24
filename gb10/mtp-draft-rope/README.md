# mtp-draft-rope

vLLM 0.32 only. Gives the speculative MTP draft the same YaRN rope settings as the target model, so contexts above
262,144 tokens work with MTP.

On stock vLLM 0.32.0 a model that is served beyond its native 262,144-token window (YaRN through a dict `--hf-overrides`)
and uses MTP crashes with `CUDA error: an illegal memory access` as soon as a conversation passes ~265K tokens. Dict overrides
reach only the target model, so the draft builds its rope table for 262,144 positions and the first decode step past it reads
outside the table. Reported as [vllm-project/vllm#62383](https://github.com/vllm-project/vllm/issues/62383).

`apply_mtp_rope.py <site-packages>/vllm` changes `vllm/config/speculative.py`: with method `mtp` and a draft that is the target
checkpoint itself, the draft gets the target's dict overrides before its MTP conversion. Any other draft keeps the stock rule.
The script refuses an anchor that does not match exactly once, so a changed vLLM fails the build.

Tested with Qwen3.8 Flash Next on vLLM 0.32.0: all eight profiles (128K to 1M) filled to the full window with 0 preemptions
([docs/vllm-versions.md](../../docs/vllm-versions.md)).
