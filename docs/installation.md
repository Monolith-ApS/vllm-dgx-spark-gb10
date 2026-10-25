# Installation

Requirements: a DGX Spark (see the hardware note in the [README](../README.md)) with docker, the NVIDIA Container Toolkit and
enough free disk for the model: the model's README says how much (Qwen3.8 Flash Next: ~160 GB, model ~133 GB and image ~24 GB).
DGX OS ships docker and the toolkit.

The scripts are the same for every model. They take the model's name as the first argument; with only one model in the
repository the name can be left out. The examples use `qwen3.8-flash-next`.

`VLLM=<release>` in front of any script picks the vLLM release, when the model offers more than one (Qwen3.8 Flash Next: `0.32`,
the default, or `0.31`; see [its comparison](../models/qwen3.8-flash-next/vllm-versions.md)). Use the same `VLLM` for build
and run.

## One command

```bash
git clone https://github.com/Monolith-ApS/vllm-dgx-spark-gb10.git
cd vllm-dgx-spark-gb10
scripts/install.sh qwen3.8-flash-next       # or: PROFILE=1024k-c1-think scripts/install.sh qwen3.8-flash-next
                                            # or: VLLM=0.31 scripts/install.sh qwen3.8-flash-next
```

`scripts/install.sh` does these steps in order and skips what is already done, so you can run it again at any time:

| Step | What happens | Takes (Qwen3.8 Flash Next) |
|---|---|---|
| 1. Prepare the host | `scripts/setup-host.sh`, with sudo (the password is asked once): creates `/data/models` and `/data/cache`, checks docker, the NVIDIA runtime and free disk | seconds |
| 2. Download the model | `scripts/get-model.sh`: from Hugging Face, no login (~133 GB) | depends on your connection |
| 3. Build the image | `scripts/build.sh`: builds on the Spark itself | ~20-30 min |
| 4. Start the server | `scripts/run.sh` with `PROFILE` (the model's default if not set) | - |
| 5. Wait | until the API answers; the first start loads the weights and compiles kernels | ~3-5 min |

If setup-host adds a warning about the docker group, log out and in again and run it once more.

## Step by step

The same steps one at a time, for example to download the model first or to rebuild only the image:

```bash
sudo scripts/setup-host.sh qwen3.8-flash-next
scripts/get-model.sh qwen3.8-flash-next
scripts/build.sh qwen3.8-flash-next
scripts/run.sh qwen3.8-flash-next
curl -s http://localhost:8000/v1/models
```

First start: ~3-5 minutes (weights load + kernel compile). The compile caches in `/data/cache` make later starts faster.

## Directory layout on the host

| Path | Content | Mounted in the container as |
|---|---|---|
| `/data/models/<weights folder>` | model weights (for Qwen3.8 Flash Next `Qwen3.8-Flash-Next-NVFP4`, ~133 GB) | `/models/<weights folder>` (read-only) |
| `/data/cache/<model>-vllm<release>/vllm` | vLLM compile cache | `/root/.cache/vllm` |
| `/data/cache/<model>-vllm<release>/flashinfer` | FlashInfer JIT cache | `/root/.cache/flashinfer` |
| `/data/cache/<model>-vllm<release>/humming` | Humming kernel cache (vLLM 0.32) | `/root/.humming` |

Each image (model and vLLM release) has its own cache directory, created by `scripts/run.sh`: a compile cache written by one
vLLM build must not be read by another ([why](vllm-versions.md#things-to-know-when-testing-several-vllm-builds)).

`DATA_DIR=/other/path` changes the root for all scripts (`sudo DATA_DIR=/other/path scripts/setup-host.sh`, then the same
`DATA_DIR` for the other scripts). `MODEL_DIR` points `scripts/run.sh` at an existing model directory.

## Scripts

| Script | What it does |
|---|---|
| `scripts/install.sh [model]` | all of the below in order; safe to run again; `REBUILD=1` builds the image again |
| `scripts/setup-host.sh [model]` | creates the directories, sets owner and permissions, checks docker, NVIDIA runtime, docker group, free disk (run with sudo) |
| `scripts/get-model.sh [model]` | downloads the model at a pinned commit if any file is missing or has the wrong size; resumes when run again; no login needed |
| `scripts/build.sh [model]` | builds the image `vllm-dgx-spark-gb10:<model>-vllm<release>` ([how](image.md)) |
| `scripts/run.sh [model]` | starts the container with a profile and the image of the chosen vLLM release; `PROFILE=list` shows the profiles |

The scripts use no credentials; the models are public on Hugging Face and download without a login. Everything that differs
between models (repository, revision, folder, default profile, vLLM releases) is in `models/<model>/model.env`; see [image.md](image.md).

## Operating

One model runs at a time, in a container called `vllm-gb10`. Starting another model, or the same one with another profile,
replaces it.

| Task | Command |
|---|---|
| Logs | `docker logs -f vllm-gb10` |
| Change profile | `PROFILE=256k-c5 scripts/run.sh qwen3.8-flash-next` |
| Switch model | `scripts/run.sh <other-model>` (after `scripts/install.sh <other-model>` the first time) |
| Switch vLLM release | `VLLM=0.31 scripts/run.sh qwen3.8-flash-next` (after `VLLM=0.31 scripts/build.sh qwen3.8-flash-next` the first time) |
| Stop | `docker rm -f vllm-gb10` |
| Update | `git pull && scripts/build.sh qwen3.8-flash-next && scripts/run.sh qwen3.8-flash-next` (with the same `VLLM` if you use one) |

The container restarts automatically after a reboot (`--restart unless-stopped`).
