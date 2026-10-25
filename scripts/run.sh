#!/usr/bin/env bash
# run.sh - start a model with a profile.
#
#   scripts/run.sh [model]                 # the model's default profile and vLLM release, port 8000
#   VLLM=0.31 scripts/run.sh [model]       # the image built for another vLLM release (scripts/build.sh with the same VLLM)
#   PROFILE=256k-c5 scripts/run.sh [model]
#   THINKING=1 scripts/run.sh [model]      # thinking on by default for requests that do not set it
#   PROFILE=list scripts/run.sh [model]    # show the model's profiles
#   DRY_RUN=1 scripts/run.sh [model]       # print the server command line the container would run, start nothing
#
# One model runs at a time: this stops and replaces the container named $NAME (default vllm-gb10), whichever model it
# was running. Logs: docker logs -f vllm-gb10
# Compile caches go to $DATA_DIR/cache/<model>-vllm<release>, one directory per image.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
select_model "${1:-}"

PORT="${PORT:-8000}"
PROFILE="${PROFILE:-$DEFAULT_PROFILE}"

if [ "$PROFILE" = "list" ]; then
  exec docker run --rm -e PROFILE=list -v "$MODEL_DIR":/models/"$WEIGHTS_DIR":ro "$IMAGE"
fi
[ -f "$MODEL_DIR/config.json" ] || { echo "no model in $MODEL_DIR (scripts/get-model.sh $MODEL)" >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "no image $IMAGE (VLLM=$VLLM scripts/build.sh $MODEL)" >&2; exit 1; }

# variables the model's entrypoint understands (PASS_ENV in model.env) are passed on when set
PASS=()
for v in ${PASS_ENV:-}; do
  if [ -n "${!v:-}" ]; then PASS+=(-e "$v=${!v}"); fi
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  exec docker run --rm -e DRY_RUN=1 -e PROFILE="$PROFILE" -e THINKING="${THINKING:-0}" -e SERVED_NAME="$SERVED_NAME" \
    -e MODEL=/models/"$WEIGHTS_DIR" ${PASS[@]+"${PASS[@]}"} -v "$MODEL_DIR":/models/"$WEIGHTS_DIR":ro "$IMAGE"
fi

mkdir -p "$CACHE_DIR/vllm" "$CACHE_DIR/flashinfer" "$CACHE_DIR/humming"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc host --shm-size 16g --restart unless-stopped \
  -p "$PORT":8000 \
  -e PROFILE="$PROFILE" -e THINKING="${THINKING:-0}" -e SERVED_NAME="$SERVED_NAME" ${API_KEY:+-e API_KEY="$API_KEY"} \
  -e MODEL=/models/"$WEIGHTS_DIR" ${PASS[@]+"${PASS[@]}"} \
  -v "$MODEL_DIR":/models/"$WEIGHTS_DIR":ro \
  -v "$CACHE_DIR/vllm":/root/.cache/vllm \
  -v "$CACHE_DIR/flashinfer":/root/.cache/flashinfer \
  -v "$CACHE_DIR/humming":/root/.humming \
  "$IMAGE" >/dev/null
echo "started $NAME: $MODEL ($PROFILE, vLLM $VLLM) on port $PORT; first start compiles kernels, ready in ~3-5 min"
echo "check: curl -s http://localhost:$PORT/v1/models"
