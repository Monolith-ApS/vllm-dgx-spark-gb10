# Configuration

## Profiles

`PROFILE` selects context window and concurrency. Concurrency = conversations that can be at the full context window at the
same time without losing their KV/prefix cache. Profiles are defined in `models/qwen3.8-flash-next/profiles.json`.

| Profile | Context (tokens) | Concurrency | gpu_memory_utilization | Mamba cache mode | Fill test |
|---|---|---|---|---|---|
| `128k-c8` | 131,072 | 8 | 0.89 | all | passed |
| `256k-c5` | 262,144 | 5 | 0.90 | all | passed |
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

For `scripts/run.sh` and the container:

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE` | `512k-c3` | see the table; `list` prints the profiles |
| `THINKING` | `0` | default `enable_thinking` for requests that do not set it |
| `SERVED_NAME` | `qwen3.8-flash-next` | model name in the API |
| `PORT` | `8000` | host port |
| `API_KEY` | none | when set, clients must send it as Bearer token |
| `GMU` | profile value | override gpu_memory_utilization |
| `DRY_RUN` | `0` | `1` prints the vLLM command line instead of starting |

`scripts/run.sh` also reads `DATA_DIR`, `MODEL_DIR`, `IMAGE` and `NAME` (container name, default `qwen38-flash-next`).

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

Reasoning text is returned in `reasoning` (streaming: `delta.reasoning`), separate from `content`. Speed and results with thinking on vs off:
[measurements.md](measurements.md#thinking-on-vs-off).

## Tool calling

Enabled (`--enable-auto-tool-choice`, parser `qwen3_coder`). Send `tools` in the OpenAI format.
