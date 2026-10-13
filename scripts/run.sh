#!/usr/bin/env bash
# run.sh - start Qwen3.8 Flash Next with a profile.
#
#   scripts/run.sh                         # PROFILE=512k-c3, port 8000
#   PROFILE=256k-c5 scripts/run.sh
#   THINKING=1 scripts/run.sh              # thinking on by default for requests that do not set it
#   PROFILE=list scripts/run.sh            # show the profiles
#
# Stops and replaces a container with the same name. Logs: docker logs -f qwen38-flash-next
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
IMAGE="${IMAGE:-vllm-dgx-spark-gb10:qwen3.8-flash-next}"
NAME="${NAME:-qwen38-flash-next}"
PORT="${PORT:-8000}"
PROFILE="${PROFILE:-512k-c3}"
MODEL_DIR="${MODEL_DIR:-$DATA_DIR/models/Qwen3.8-Flash-Next-NVFP4}"

if [ "$PROFILE" = "list" ]; then
  exec docker run --rm -e PROFILE=list -v "$MODEL_DIR":/models/Qwen3.8-Flash-Next-NVFP4:ro "$IMAGE"
fi
[ -f "$MODEL_DIR/config.json" ] || { echo "no model in $MODEL_DIR (scripts/get-model.sh)" >&2; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc host --shm-size 16g --restart unless-stopped \
  -p "$PORT":8000 \
  -e PROFILE="$PROFILE" -e THINKING="${THINKING:-0}" ${API_KEY:+-e API_KEY="$API_KEY"} \
  -v "$MODEL_DIR":/models/Qwen3.8-Flash-Next-NVFP4:ro \
  -v "$DATA_DIR/cache/vllm":/root/.cache/vllm \
  -v "$DATA_DIR/cache/flashinfer":/root/.cache/flashinfer \
  "$IMAGE" >/dev/null
echo "started $NAME ($PROFILE) on port $PORT; first start compiles kernels, ready in ~3-5 min"
echo "check: curl -s http://localhost:$PORT/v1/models"
