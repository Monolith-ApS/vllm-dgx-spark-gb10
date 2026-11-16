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
| `THINK_GUARD_CONTENT` | `0` | `1` also resumes answers that end on EOS in the middle of a sentence (see below) |
| `THINK_GUARD_CONTENT_MIN_PROMPT` | `120000` | prompt tokens needed before the content guard applies |

Log lines start with `think-guard`.

## Content guard (optional)

With thinking off the prompt contains a closed think block, so the guard above stays out of the way. In very long
conversations Qwen3.8 Flash Next can then sample EOS in the middle of a sentence: on a DGX Spark, replaying a real
815-message agent session of about 189K tokens ended after 4 to 120 tokens in 7 to 10 of 10 runs, while the same session cut
to 169K tokens was fine. The end token had 0.37 to 0.51 probability in the middle of sentences at 189K and 0 at 169K and
95K (teacher-forced). It did not depend on the KV cache dtype, MTP (7, 3 or off), sampling, the vLLM version (0.31 and 0.33),
the GDN state dtype or the prefill chunk size; thinking on and a different last user message removed it.

`THINK_GUARD_CONTENT=1` resumes such a turn like the reasoning case: the prompt has at least
`THINK_GUARD_CONTENT_MIN_PROMPT` tokens, no tool call was seen, the engine ends on EOS and the last visible character is a letter,
digit, space or one of `,;:-([`. A trailing newline counts as finished. It also applies after a closed reasoning block when
thinking is on. Sampling is untouched, so MTP stays on. On the replay above all answers then end on a full sentence (2 to 13
resumes per turn; `THINK_GUARD_MAX_RESUMES=30` and `THINK_GUARD_MIN_TOKENS=16` were used). A legitimate answer that ends on a
letter or digit at such a long prompt would be extended by at least `THINK_GUARD_MIN_TOKENS` tokens, so keep it off unless you
see the symptom. 20 offline tests in `test_think_guard.py`.
