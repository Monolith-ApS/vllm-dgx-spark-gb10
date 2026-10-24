# think-eos-guard

Server-side fix for two reasoning-channel failures of Qwen3.8 (vLLM 0.30 to 0.32, `qwen3` reasoning parser):

1. The model samples EOS (`<|im_end|>`) inside an unfinished reasoning block. vLLM returns `finish_reason=stop` with the whole
   output in `reasoning_content`, empty `content` and no tool call. The guard resumes the generation from prompt + generated
   tokens (EOS removed, `min_tokens` on the continuation), up to `THINK_GUARD_MAX_RESUMES` times. The client sees one stream.
2. The model writes the closing think tag as quoted text. The tag counts as a real end of reasoning only directly after a
   newline; other occurrences are rewritten to `</` + U+200B + `think>` before the parser sees them.

Sampling is not changed, so MTP speculative decoding keeps working. Related upstream: vLLM #55420.

Files: `think_guard.py` (module, the same for every release), `serving_think_guard-v031.diff` and `-v032.diff` (wrap
`engine_client.generate` in `OpenAIServingChat`, one per vLLM release),
`test_think_guard.py` (offline tests: `python test_think_guard.py`).

| Variable | Default | Meaning |
|---|---|---|
| `THINK_GUARD` | `1` | `0` disables the guard |
| `THINK_GUARD_MAX_RESUMES` | `6` | resumes per request |
| `THINK_GUARD_MIN_TOKENS` | `12` | `min_tokens` on a continuation |

Log lines start with `think-guard`.
