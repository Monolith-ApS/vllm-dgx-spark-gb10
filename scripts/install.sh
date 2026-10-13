#!/usr/bin/env bash
# install.sh - from a fresh DGX Spark to a running API in one command.
#
#   scripts/install.sh                         # profile 512k-c3, port 8000
#   PROFILE=1024k-c1-think scripts/install.sh  # any profile from scripts/run.sh / PROFILE=list
#
# Runs setup-host (asks for the sudo password once), get-model, build and run, then waits until the API answers.
# Every step skips work that is already done, so the script is safe to run again (for example after an interruption
# or after a git pull). Switching profile later only needs scripts/run.sh, not this script.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="qwen3.8-flash-next"
export DATA_DIR="${DATA_DIR:-/data}"
PORT="${PORT:-8000}"
export PROFILE="${PROFILE:-512k-c3}" PORT

step() { echo; echo "== $*"; }

step "1/5 host directories and checks"
if [ "$(id -u)" -eq 0 ]; then
  OWNER_USER="${SUDO_USER:-root}" "$ROOT/scripts/setup-host.sh"
else
  sudo DATA_DIR="$DATA_DIR" OWNER_USER="$(id -un)" "$ROOT/scripts/setup-host.sh"
fi
if ! docker info >/dev/null 2>&1; then
  echo "FAIL docker is not usable by $(id -un) yet. If setup-host warned about the docker group: log out and in again," >&2
  echo "     then run scripts/install.sh again (finished steps are skipped)." >&2
  exit 1
fi

step "2/5 model weights"
"$ROOT/scripts/get-model.sh"

step "3/5 image"
if docker image inspect "vllm-dgx-spark-gb10:$MODEL" >/dev/null 2>&1 && [ "${REBUILD:-0}" != "1" ]; then
  echo "OK   image vllm-dgx-spark-gb10:$MODEL exists (REBUILD=1 to build again, e.g. after git pull)"
else
  "$ROOT/scripts/build.sh" "$MODEL"
fi

step "4/5 start ($PROFILE)"
"$ROOT/scripts/run.sh"

step "5/5 waiting for the API (first start compiles kernels, ~3-5 min)"
for _ in $(seq 1 180); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "OK   ready: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/v1  model qwen3.8-flash-next  profile $PROFILE"
    exit 0
  fi
  if [ -z "$(docker ps -q -f name="^${NAME:-qwen38-flash-next}\$")" ]; then
    echo "FAIL the container stopped; last log lines:" >&2
    docker logs --tail 30 "${NAME:-qwen38-flash-next}" >&2 || true
    exit 1
  fi
  sleep 10
done
echo "FAIL not ready after 30 min; see: docker logs ${NAME:-qwen38-flash-next}" >&2
exit 1
