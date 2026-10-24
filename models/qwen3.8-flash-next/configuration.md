# Qwen3.8 Flash Next: configuration

How the scripts and environment variables work in general: [docs/configuration.md](../../docs/configuration.md). This page is what
is specific to this model. The model runs on vLLM 0.31 (default) or 0.32 (`VLLM=0.32`, see [vllm-versions.md](vllm-versions.md));
the profiles and variables below are the same on both.

## Profiles

`PROFILE` selects context window and concurrency. Concurrency = conversations that can be at the full context window at the
same time without losing their KV/prefix cache. Profiles are defined in `profiles.json` in this folder.

| Profile | Context (tokens) | Concurrency | gpu_memory_utilization | Mamba cache mode | Fill test |
|---|---|---|---|---|---|
| `128k-c8` | 131,072 | 8 | 0.89 | all | passed |
| `256k-c5` | 262,144 | 5 | 0.89 | all | passed |
| `384k-c4` | 393,216 | 4 | 0.90 | all | passed |
| `480k-c3` | 491,520 | 3 | 0.89 | all | passed |
| `512k-c3` (default) | 524,288 | 3 | 0.89 | all | passed |
| `640k-c2` | 655,360 | 2 | 0.89 | all | passed |
| `768k-c2` | 786,432 | 2 | 0.89 | all | passed |
| `1024k-c1` | 1,048,576 | 1 | 0.88 | all | passed |

Every profile also exists as `<name>-think` (thinking on by default, same memory use).

Fill test: [measurements.md](measurements.md#capacity). Above 262,144 tokens the profile adds YaRN rope scaling
(`factor = context / 262144`) automatically.

## Environment variables

Read by `scripts/run.sh` and passed on to the container (the general variables - `PORT`, `API_KEY`, `DATA_DIR` and so on - are in
[docs/configuration.md](../../docs/configuration.md#environment-variables)):

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE` | `512k-c3` | see the table; `list` prints the profiles |
| `THINKING` | `0` | default `enable_thinking` for requests that do not set it |
| `SERVED_NAME` | `qwen3.8-flash-next` | model name in the API |
| `GMU` | profile value | override gpu_memory_utilization |
| `MTP` | `7`, with thinking `3` | speculative (MTP) draft tokens per step |
| `LOAD_FORMAT` | `instanttensor` | weight loader (`safetensors` also works) |

Arguments after the image name in `docker run` are appended to the vLLM command line.

## Thinking mode

Per request:

```json
{"model": "qwen3.8-flash-next", "messages": [...], "chat_template_kwargs": {"enable_thinking": true}}
```

Server-wide default sampling, chosen by `THINKING` at start; a request can override each value:

| `THINKING` | temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| `0` | 1.0 | 0.95 | 20 | 0.0 | - |
| `1` | 0.8 | 0.95 | 20 | 0.0 | 1.0 |

With thinking on by default (`-think` or `THINKING=1`) the server uses 3 MTP draft tokens instead of 7: reasoning text
accepts fewer draft tokens, so the shorter draft decodes faster ([measurements.md](measurements.md#thinking-on-vs-off)).
Requests that turn thinking on against a non-thinking profile run with 7.

Reasoning text is returned in `reasoning` (streaming: `delta.reasoning`), separate from `content`. Speed and results with thinking on vs off:
[measurements.md](measurements.md#thinking-on-vs-off).

## Tool calling

Enabled (`--enable-auto-tool-choice`, parser `qwen3_coder`). Send `tools` in the OpenAI format.
