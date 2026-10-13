#!/usr/bin/env bash
# setup-host.sh - prepare a DGX Spark / GB10 host for vllm-dgx-spark-gb10.
#
# Creates the data directories, gives them to the user who runs the containers, and checks docker, the NVIDIA runtime
# and free disk space. Safe to run again.
#
#   sudo scripts/setup-host.sh            # DATA_DIR=/data, owner = the user who called sudo
#   sudo DATA_DIR=/srv/llm scripts/setup-host.sh
#
# Layout:
#   $DATA_DIR/models/<model>          model weights (mounted read-only at /models)
#   $DATA_DIR/cache/vllm              vLLM compile cache (mounted at /root/.cache/vllm)
#   $DATA_DIR/cache/flashinfer        FlashInfer JIT cache (mounted at /root/.cache/flashinfer)
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OWNER_USER="${OWNER_USER:-${SUDO_USER:-$(id -un)}}"
NEED_GB="${NEED_GB:-150}"

if [ "$(id -u)" -ne 0 ]; then
  echo "run with sudo (it creates $DATA_DIR and sets its owner)" >&2
  exit 1
fi

mkdir -p "$DATA_DIR/models" "$DATA_DIR/cache/vllm" "$DATA_DIR/cache/flashinfer"
chown "$OWNER_USER":"$(id -gn "$OWNER_USER")" "$DATA_DIR/models" "$DATA_DIR/cache" "$DATA_DIR/cache/vllm" "$DATA_DIR/cache/flashinfer"
chmod 755 "$DATA_DIR/models"
chmod 775 "$DATA_DIR/cache" "$DATA_DIR/cache/vllm" "$DATA_DIR/cache/flashinfer"
echo "OK   directories under $DATA_DIR, owner $OWNER_USER"

if ! command -v docker >/dev/null; then
  echo "FAIL docker is not installed" >&2; exit 1
fi
if docker info 2>/dev/null | grep -qi nvidia; then
  echo "OK   docker with the NVIDIA runtime"
else
  echo "WARN docker info shows no NVIDIA runtime; install the NVIDIA Container Toolkit (DGX OS ships it)" >&2
fi
if ! id -nG "$OWNER_USER" | grep -qw docker; then
  echo "WARN $OWNER_USER is not in the docker group: sudo usermod -aG docker $OWNER_USER (then log in again)" >&2
fi
free_gb=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$NEED_GB" ]; then
  echo "WARN $DATA_DIR has ${free_gb} GB free; Qwen3.8 Flash Next needs ~${NEED_GB} GB (weights ~133 GB + caches)" >&2
else
  echo "OK   ${free_gb} GB free in $DATA_DIR"
fi
echo "next: scripts/get-model.sh"
