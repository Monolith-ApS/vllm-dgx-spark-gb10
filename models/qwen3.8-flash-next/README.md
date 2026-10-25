# Qwen3.8 Flash Next

`nvidia/Qwen3.8-Flash-Next-NVFP4`, served as `qwen3.8-flash-next`: about 78 tokens/s for a single request, from 128K context
with 8 conversations to 1M context with 1, tool calling, optional thinking mode.

```bash
scripts/install.sh qwen3.8-flash-next
```

(With this as the only model, the name can be left out.) What the command does: [installation](../../docs/installation.md).
When it says OK, try it:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Hello"}]}'
```

Any OpenAI-compatible client works. That means the programs that normally talk to the OpenAI API (coding agents, chat front
ends, the Python `openai` package): point them at this server instead, with base URL `http://<spark>:8000/v1` (`<spark>` is the
machine's name or IP address, `localhost` if the client runs on the Spark itself) and model name `qwen3.8-flash-next`.

Anyone who can reach port 8000 can use the server. To require a password, see [security](../../docs/security.md).

## Choosing a profile

A profile is a trade-off. The server has one pool of memory for what the conversations have said so far (the KV cache), and the
profile decides how it is shared: a longer context window means fewer conversations that can use the full length at the same
time. The name says both: `256k-c5` is a 256K-token window for up to 5 conversations.

```bash
PROFILE=256k-c5 scripts/run.sh qwen3.8-flash-next
```

| Profile | Context window | Conversations at full length | Speed, one request | Speed, all conversations at once (total) |
|---|---|---|---|---|
| `128k-c8` | 128K | 8 | 80 (76-82) tok/s | 232 (219-253) tok/s |
| `256k-c5` | 256K | 5 | 78 (76-80) tok/s | 177 (162-186) tok/s |
| `384k-c4` | 384K | 4 | 77 (77-78) tok/s | 162 (159-165) tok/s |
| `480k-c3` | 480K | 3 | 80 (76-84) tok/s | 141 (138-147) tok/s |
| `512k-c3` (default) | 512K | 3 | 79 (77-81) tok/s | 146 (142-151) tok/s |
| `640k-c2` | 640K | 2 | 78 (75-80) tok/s | 117 (114-120) tok/s |
| `768k-c2` | 768K | 2 | 79 (78-82) tok/s | 121 (117-127) tok/s |
| `1024k-c1` | 1M | 1 | 77 (72-80) tok/s | - |

Speeds are tokens per second on vLLM 0.32.0 (the default): mean (lowest-highest) over 3 runs (`384k-c4`: 1 run). The numbers
for vLLM 0.31.0 are in [measurements](measurements.md). "All at once" is the total for all conversations together, so
each one gets roughly the total divided by the number of conversations (`128k-c8`: about 29 tok/s each).

**Which one?**

- Several people or agents, ordinary prompts: `128k-c8` (most conversations, highest total speed).
- Not sure: `512k-c3`, the default.
- One very large codebase or document: `1024k-c1`.

Every profile passed the capacity test: all conversations filled to the full context window at the same time, without
the server throwing any of them away. Any profile also runs with thinking on by default: add `-think`
(`PROFILE=1024k-c1-think`). For thinking, `768k-c2-think` and `1024k-c1-think` are the recommended ones; thinking is slower
and uses many more tokens, see [Thinking on vs off](measurements.md#thinking-on-vs-off).

Shorter conversations use less memory, so more than the listed number can run at once as long as the total fits.
Details per profile, and what the terms mean: [measurements](measurements.md).

## More about this model

| | |
|---|---|
| [configuration.md](configuration.md) | profile table with the memory settings, environment variables, thinking mode, tool calling |
| [measurements.md](measurements.md) | decode speed, capacity, tool calling, thinking on vs off |
| [vllm-versions.md](vllm-versions.md) | vLLM 0.32 (default) or 0.31 (`VLLM=0.31`): how to choose, and how they compare |
| [architecture.md](architecture.md) | image layers and the precision of every model component (diagrams) |
| [pins.md](pins.md) | the exact versions the image is built from |

## Limits specific to this model

- The image stores some model components at lower precision than the checkpoint (FP8 KV cache, FP8 and NVFP4 weights; see
  [architecture](architecture.md)). Quality was checked only with the tests in [measurements](measurements.md) (tool calling,
  control prompts, small coding tasks, capacity fills). That does not show the model is as good as the original on every task,
  nor quality at long context beyond what those tests cover (contexts above 256K use YaRN scaling).
- The profiles use most of the 128 GB unified memory. Other processes on the machine reduce the headroom.

The model weights are governed by the license on the model card of `nvidia/Qwen3.8-Flash-Next-NVFP4`.
