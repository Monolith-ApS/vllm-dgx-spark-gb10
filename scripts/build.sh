#!/usr/bin/env bash
# build.sh - build a model's image from this repository.
#
#   scripts/build.sh [model]                  # the model's default vLLM release -> vllm-dgx-spark-gb10:<model>-vllm<release>
#   VLLM=0.31 scripts/build.sh [model]        # another release the model supports (VLLM_RELEASES in its model.env)
#   TAG=myregistry/vllm-dgx-spark-gb10:<model>-vllm0.31 scripts/build.sh <model>
#
# Build on the DGX Spark itself (aarch64 + the GB10 GPU architecture). The Dockerfile is
# models/<model>/Dockerfile.vllm-<release>; if models/<model>/pre-build.sh exists it runs first (fetching pinned sources and
# the like, see that file).
set -euo pipefail
. "$(dirname "$0")/lib.sh"
select_model "${1:-}"

TAG="${TAG:-$IMAGE}"
BUILD_ARGS=()
if [ -f "$ROOT/models/$MODEL/pre-build.sh" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/models/$MODEL/pre-build.sh"
fi

DOCKER_BUILDKIT=1 docker build \
  -f "$ROOT/models/$MODEL/Dockerfile.vllm-$VLLM" \
  ${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"} \
  -t "$TAG" "$ROOT"
echo "built $TAG"
