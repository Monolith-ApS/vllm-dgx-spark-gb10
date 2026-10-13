#!/usr/bin/env bash
# build.sh - build a model image from this repository.
#
#   scripts/build.sh qwen3.8-flash-next                 # -> vllm-dgx-spark-gb10:qwen3.8-flash-next
#   TAG=myregistry/vllm-dgx-spark-gb10:qwen3.8-flash-next scripts/build.sh qwen3.8-flash-next
#
# Build on the DGX Spark itself (aarch64 + the GB10 GPU architecture). The deterministic top-k kernel is compiled during
# the build; MAX_JOBS=2 keeps the compile from taking the memory a running model needs.
set -euo pipefail

MODEL="${1:-qwen3.8-flash-next}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${TAG:-vllm-dgx-spark-gb10:$MODEL}"
BLAZUX_REPO="https://github.com/blazux/qwen3.8-Flash-DGX.git"
BLAZUX_COMMIT="5108d90dedf15a31aa20432319f151b2fdda17a3"
WORK="$ROOT/.build"

case "$MODEL" in
  qwen3.8-flash-next) ;;
  *) echo "unknown model '$MODEL' (available: qwen3.8-flash-next)" >&2; exit 2 ;;
esac

mkdir -p "$WORK"
if [ ! -d "$WORK/blazux/.git" ]; then
  git clone --quiet "$BLAZUX_REPO" "$WORK/blazux"
fi
git -C "$WORK/blazux" fetch --quiet origin "$BLAZUX_COMMIT" 2>/dev/null || git -C "$WORK/blazux" fetch --quiet origin
git -C "$WORK/blazux" checkout --quiet "$BLAZUX_COMMIT"
echo "blazux overlay at $(git -C "$WORK/blazux" rev-parse --short HEAD)"

DOCKER_BUILDKIT=1 docker build \
  -f "$ROOT/models/$MODEL/Dockerfile" \
  --build-context blazux="$WORK/blazux" \
  -t "$TAG" "$ROOT"
echo "built $TAG"
