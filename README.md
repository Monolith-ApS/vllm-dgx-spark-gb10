# vllm-dgx-spark-gb10

A working, tested vLLM setup for the NVIDIA DGX Spark: unpack the Spark, clone this repository, run one script, and you
have an OpenAI-compatible API serving a current model with long context and several parallel conversations.

The focus is vLLM: the most performance, quality and stability that vLLM can deliver on the GB10 chip for the models in this
repository. Each model gets the vLLM release, patches and settings that were measured and tested for it, not a one-size setup.

What you get:

- A container image per model, built from an official vLLM release (chosen and pinned per model) plus the patches that
  model needs to run correctly and fast on the GB10 chip, all pinned to exact versions.
- `scripts/install.sh <model>`: prepares the host, downloads the model, builds the image and starts it. The same scripts
  serve every model.
- Ready-made profiles (context window x parallel conversations) that have been load-tested on the Spark.
- The test tools used for the measurements, so you can verify your own machine.

One model runs at a time; the profiles use most of the Spark's memory.

Not affiliated with NVIDIA or Qwen. Model weights are not included; they are downloaded from Hugging Face under each model's
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

## Models

| Model | Name for the scripts | What it is | Speed, one request |
|---|---|---|---|
| [Qwen3.8 Flash Next](models/qwen3.8-flash-next/README.md) | `qwen3.8-flash-next` | `nvidia/Qwen3.8-Flash-Next-NVFP4`: 128K context with 8 conversations up to 1M context with 1, tool calling, optional thinking mode | ~78 tok/s |

Each model's page has its profiles, speeds and limits.

## Quick start

On the Spark, with docker and the NVIDIA Container Toolkit (DGX OS ships both):

```bash
git clone https://github.com/Monolith-ApS/vllm-dgx-spark-gb10.git
cd vllm-dgx-spark-gb10
scripts/install.sh qwen3.8-flash-next
```

That does everything and can be run again at any time. What it does, how long it takes and the individual steps:
[docs/installation.md](docs/installation.md). When it says OK, the server answers on port 8000; how to try it and which
profile to choose: [the model's page](models/qwen3.8-flash-next/README.md).

Any OpenAI-compatible client works: the programs that normally talk to the OpenAI API (coding agents, chat front ends, the
Python `openai` package) can point at this server instead, with base URL `http://<spark>:8000/v1` and the model's name.

Anyone who can reach port 8000 can use the server. To require a password, see [docs/security.md](docs/security.md).

## Documentation

| | |
|---|---|
| [docs/installation.md](docs/installation.md) | host directories, scripts, switching model, updating, stopping |
| [docs/configuration.md](docs/configuration.md) | profiles and environment variables that work for every model |
| [docs/security.md](docs/security.md) | API key, what it does not protect, safe access over a network |
| [docs/tools.md](docs/tools.md) | test tools to verify your own machine |
| [docs/vllm-versions.md](docs/vllm-versions.md) | how vLLM releases compare on the GB10 and the problems we found (long context, start-up) |
| [docs/image.md](docs/image.md) | how an image is built, repository layout, adding a model |
| [models/qwen3.8-flash-next/](models/qwen3.8-flash-next/README.md) | the model: profiles, configuration, [measurements](models/qwen3.8-flash-next/measurements.md), [architecture](models/qwen3.8-flash-next/architecture.md), [pins](models/qwen3.8-flash-next/pins.md) |

## Limitations and responsibility

- Model output - answers, code, commands, tool calls - can be wrong, incomplete, outdated or made up. Review it before you
  rely on it, and run generated code and agent tool calls only in isolated environments you are allowed to use.
- The measurements come from one DGX Spark with the versions pinned for each model. Another driver/CUDA version, OS
  image, model revision, sampling settings, speculative decoding settings or agent framework can change speed, memory
  headroom and model behaviour. What was and was not checked for each model is on its page.
- The profiles use most of the 128 GB unified memory. Other processes on the machine reduce the headroom, and on GB10 running
  out of memory can freeze the whole machine until it is power cycled.
- The server is not a security boundary. By default the API needs no key and port 8000 is published on all interfaces: set
  `API_KEY` and keep the port off networks you do not trust ([docs/security.md](docs/security.md)).

Whoever runs it is responsible for access control, network exposure, monitoring, and following the licenses of the model
and the software.

## License and disclaimer

Apache-2.0 (see `LICENSE`). Third-party sources and credits: `NOTICE`. The model weights are not part of this repository;
their use is governed by the license on each model's model card on Hugging Face.

**Use at your own risk.** This repository is provided as is, without warranty of any kind, including correctness, fitness
for a particular purpose, security or non-infringement. Monolith ApS and the contributors are not liable for any damage or
loss arising from its use, including loss or corruption of data, downtime, damage to hardware or software, and wrong,
incomplete or harmful model output or decisions based on it (see also Apache-2.0, sections 7 and 8).

Report reproducible problems in a GitHub issue. Do not include API keys, tokens, passwords or private data.

Copyright 2026 Monolith ApS (https://www.monolith.dk/en)

Maintainer: Carsten Berggreen (https://berggreens.dk/en/)
