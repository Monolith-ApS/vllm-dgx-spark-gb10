# vllm-dgx-spark-gb10

A working, tested vLLM setup for the NVIDIA DGX Spark: unpack the Spark, clone this repository, run one script, and you
have an OpenAI-compatible API serving a current model with long context and several parallel conversations.

What you get:

- A container image built from the official vLLM release plus the patches the model needs to run correctly and fast on
  the GB10 chip, all pinned to exact versions.
- `scripts/install.sh`: prepares the host, downloads the model, builds the image and starts it.
- Ready-made profiles (context window x parallel conversations) that have been load-tested on the Spark.
- The test tools used for the measurements, so you can verify your own machine.

Models: **Qwen3.8 Flash Next** (`nvidia/Qwen3.8-Flash-Next-NVFP4`) - ~78 tokens/s for a single request, from 128K context with 8
conversations to 1M context with 1, tool calling, optional thinking mode.

Not affiliated with NVIDIA or Qwen. Model weights are not included; they are downloaded from Hugging Face under the model's
own license.

> [!NOTE]
> **Hardware compatibility.** This repository is tested only on the **NVIDIA DGX Spark** (GB10, 128 GB, DGX OS).
>
> Other GB10 systems use the same chip and memory size and may work: ASUS Ascent GX10, Dell Pro Max with GB10, HP ZGX Nano,
> Lenovo ThinkStation PGX, MSI EdgeXpert, Gigabyte AI TOP Atom, Acer Veriton GN100. They are **not tested**. OS image,
> driver/CUDA version, memory available to the GPU and cooling/power limits can differ, which can change the memory headroom
> of the profiles and the speeds.
>
> If you run this on one of them, please report back in a GitHub issue - working or not - with the machine, OS/driver
> version, profile and `MemAvailable` after start (`grep MemAvailable /proc/meminfo`).

## Quick start

On the Spark, with docker and the NVIDIA Container Toolkit (DGX OS ships both):

```bash
git clone https://github.com/Monolith-ApS/vllm-dgx-spark-gb10.git
cd vllm-dgx-spark-gb10
scripts/install.sh
```

`scripts/install.sh` asks for the sudo password once and then does everything: creates `/data/models` and `/data/cache`,
downloads the model (~133 GB, no Hugging Face login), builds the image (~20-30 min), starts it with profile `512k-c3` and
waits until the API answers. It can be run again at any time; finished steps are skipped. The individual steps are
`scripts/setup-host.sh`, `scripts/get-model.sh`, `scripts/build.sh` and `scripts/run.sh` ([docs/installation.md](docs/installation.md)).

Then:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Hello"}]}'
```

Any OpenAI-compatible client works with base URL `http://<spark>:8000/v1` and model `qwen3.8-flash-next`. Set `API_KEY` when
starting to require a key.

## Choosing a profile

A profile sets the context window and how many conversations can be at that full length at the same time:

```bash
PROFILE=256k-c5 scripts/run.sh
```

| Profile | Context | Conversations at full context | Decode, 1 request | Decode, all conversations at once (total) |
|---|---|---|---|---|
| `128k-c8` | 128K | 8 | 75-79 tok/s | 223-256 tok/s |
| `256k-c5` | 256K | 5 | 79-82 tok/s | 178-189 tok/s |
| `384k-c4` | 384K | 4 | 79-83 tok/s | 165-176 tok/s |
| `480k-c3` | 480K | 3 | 79-80 tok/s | 139-149 tok/s |
| `512k-c3` (default) | 512K | 3 | 79-80 tok/s | 141-152 tok/s |
| `640k-c2` | 640K | 2 | 73-79 tok/s | 112-118 tok/s |
| `768k-c2` | 768K | 2 | 79-83 tok/s | 116-118 tok/s |
| `1024k-c1` | 1M | 1 | 78-79 tok/s | - |

Every profile passed the capacity test: all conversations filled to the full context window at the same time, without
evictions. Any profile also runs with thinking on by default: add `-think` (`PROFILE=1024k-c1-think`). For thinking,
`768k-c2-think` and `1024k-c1-think` are the recommended ones.

Shorter conversations use less memory, so more than the listed number can run at once as long as the total fits.
Details per profile: [docs/measurements.md](docs/measurements.md).

## Documentation

| | |
|---|---|
| [docs/installation.md](docs/installation.md) | host directories, scripts, updating, stopping |
| [docs/configuration.md](docs/configuration.md) | profiles, environment variables, thinking mode, API key |
| [docs/architecture.md](docs/architecture.md) | image layers and the precision of every model component (diagrams) |
| [docs/measurements.md](docs/measurements.md) | decode speed, capacity, tool calling, thinking on vs off |
| [docs/tools.md](docs/tools.md) | test tools to verify your own machine |
| [docs/building.md](docs/building.md) | pinned versions and repository layout |

## Limitations and responsibility

- Model output - answers, code, commands, tool calls - can be wrong, incomplete, outdated or made up. Review it before you
  rely on it, and run generated code and agent tool calls only in isolated environments you are allowed to use.
- The measurements come from one DGX Spark with the versions pinned in this repository. Another driver/CUDA version, OS
  image, model revision, sampling settings, speculative decoding settings or agent framework can change speed, memory
  headroom and model behaviour.
- The image stores some model components at lower precision than the checkpoint (FP8 KV cache, FP8 and NVFP4 weights; see
  [docs/architecture.md](docs/architecture.md)). Quality was checked only with the tests in
  [docs/measurements.md](docs/measurements.md) (tool calling, control prompts, small coding tasks, capacity fills). That does
  not show the model is as good as the original on every task, nor quality at long context beyond what those tests cover
  (contexts above 256K use YaRN scaling).
- The profiles use most of the 128 GB unified memory. Other processes on the machine reduce the headroom, and on GB10 running
  out of memory can freeze the whole machine until it is power cycled.
- The server is not a security boundary. By default the API needs no key and port 8000 is published on all interfaces: set
  `API_KEY` and keep the port off networks you do not trust.

Whoever runs it is responsible for access control, network exposure, monitoring, and following the licenses of the model
and the software.

## License and disclaimer

Apache-2.0 (see `LICENSE`). Third-party sources and credits: `NOTICE`. The model weights are not part of this repository;
their use is governed by the license on the model card of `nvidia/Qwen3.8-Flash-Next-NVFP4`.

**Use at your own risk.** This repository is provided as is, without warranty of any kind, including correctness, fitness
for a particular purpose, security or non-infringement. Monolith ApS and the contributors are not liable for any damage or
loss arising from its use, including loss or corruption of data, downtime, damage to hardware or software, and wrong,
incomplete or harmful model output or decisions based on it (see also Apache-2.0, sections 7 and 8).

Report reproducible problems in a GitHub issue. Do not include API keys, tokens, passwords or private data.

Copyright 2026 Monolith ApS (https://www.monolith.dk/en)

Maintainer: Carsten Berggreen (https://berggreens.dk/en/)
