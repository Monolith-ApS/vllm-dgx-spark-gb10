# Installation

Requirements: a DGX Spark (see the hardware note in the [README](../README.md)) with docker, the NVIDIA Container Toolkit and
~160 GB free disk (model ~133 GB, image ~24 GB). DGX OS ships docker and the toolkit.

## One command

```bash
git clone https://github.com/Monolith-ApS/vllm-dgx-spark-gb10.git
cd vllm-dgx-spark-gb10
scripts/install.sh                          # or: PROFILE=1024k-c1-think scripts/install.sh
```

`scripts/install.sh` runs the four steps below in order, skips what is already done and waits until the API answers. If
setup-host adds a warning about the docker group, log out and in again and run it once more.

## Step by step

```bash
sudo scripts/setup-host.sh
scripts/get-model.sh
scripts/build.sh qwen3.8-flash-next
scripts/run.sh
curl -s http://localhost:8000/v1/models
```

First start: ~3-5 minutes (weights load + kernel compile). The compile cache in `/data/cache` makes later starts faster.

## Directory layout on the host

| Path | Content | Mounted in the container as |
|---|---|---|
| `/data/models/Qwen3.8-Flash-Next-NVFP4` | model weights (~133 GB) | `/models/Qwen3.8-Flash-Next-NVFP4` (read-only) |
| `/data/cache/vllm` | vLLM compile cache | `/root/.cache/vllm` |
| `/data/cache/flashinfer` | FlashInfer JIT cache | `/root/.cache/flashinfer` |

`DATA_DIR=/other/path` changes the root for all scripts (`sudo DATA_DIR=/other/path scripts/setup-host.sh`, then the same
`DATA_DIR` for the other scripts). `MODEL_DIR` points `scripts/run.sh` at an existing model directory.

## Scripts

| Script | What it does |
|---|---|
| `scripts/install.sh` | all of the below in order; safe to run again; `REBUILD=1` builds the image again |
| `scripts/setup-host.sh` | creates the directories, sets owner and permissions, checks docker, NVIDIA runtime, docker group, free disk (run with sudo) |
| `scripts/get-model.sh` | downloads the model at a pinned commit if any file is missing or has the wrong size; resumes when run again; no login needed |
| `scripts/build.sh <model>` | fetches the blazux overlay at its pinned commit and builds the image `vllm-dgx-spark-gb10:<model>` |
| `scripts/run.sh` | starts the container with a profile; `PROFILE=list` shows the profiles |

The scripts use no credentials; the model is public on Hugging Face and downloads without a login.

## Operating

| Task | Command |
|---|---|
| Logs | `docker logs -f qwen38-flash-next` |
| Change profile | `PROFILE=256k-c5 scripts/run.sh` (replaces the running container) |
| Stop | `docker rm -f qwen38-flash-next` |
| Update | `git pull && scripts/build.sh qwen3.8-flash-next && scripts/run.sh` |

The container restarts automatically after a reboot (`--restart unless-stopped`).
