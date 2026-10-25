#!/usr/bin/env bash
# install.sh - from a fresh DGX Spark to a running API in one command.
#
#   scripts/install.sh [model]                           # the model's default profile, port 8000
#   PROFILE=1024k-c1-think scripts/install.sh [model]    # any profile of the model (PROFILE=list scripts/run.sh shows them)
#   VLLM=0.31 scripts/install.sh [model]                 # another vLLM release the model supports (VLLM_RELEASES in model.env)
#
# Runs setup-host (asks for the sudo password once), get-model, build and run, then waits until the API answers.
# Every step skips work that is already done, so the script is safe to run again (for example after an interruption
# or after a git pull). Switching profile later only needs scripts/run.sh, not this script.
set -euo pipefail
. "$(dirname "$0")/lib.sh"
select_model "${1:-}"

export DATA_DIR VLLM PORT="${PORT:-8000}"
export PROFILE="${PROFILE:-$DEFAULT_PROFILE}"

step() { echo; echo "== $*"; }

step "1/5 host directories and checks"
if [ "$(id -u)" -eq 0 ]; then
  OWNER_USER="${SUDO_USER:-root}" "$ROOT/scripts/setup-host.sh" "$MODEL"
else
  sudo DATA_DIR="$DATA_DIR" OWNER_USER="$(id -un)" "$ROOT/scripts/setup-host.sh" "$MODEL"
fi
if ! docker info >/dev/null 2>&1; then
  echo "FAIL docker is not usable by $(id -un) yet. If setup-host warned about the docker group: log out and in again," >&2
  echo "     then run scripts/install.sh again (finished steps are skipped)." >&2
  exit 1
fi

step "2/5 model weights"
"$ROOT/scripts/get-model.sh" "$MODEL"

step "3/5 image"
if docker image inspect "$IMAGE" >/dev/null 2>&1 && [ "${REBUILD:-0}" != "1" ]; then
  echo "OK   image $IMAGE exists (REBUILD=1 to build again, e.g. after git pull)"
else
  "$ROOT/scripts/build.sh" "$MODEL"
fi

step "4/5 start ($PROFILE, vLLM $VLLM)"
"$ROOT/scripts/run.sh" "$MODEL"

step "5/5 waiting for the API (first start compiles kernels, ~3-5 min)"
for _ in $(seq 1 180); do
  if curl -sf ${API_KEY:+-H "Authorization: Bearer $API_KEY"} "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "OK   ready: http://$(hostname -I 2>/dev/null | awk '{print $1}'):$PORT/v1  model $SERVED_NAME  profile $PROFILE  vLLM $VLLM"
    exit 0
  fi
  if [ -z "$(docker ps -q -f name="^${NAME}\$")" ]; then
    echo "FAIL the container stopped; last log lines:" >&2
    docker logs --tail 30 "$NAME" >&2 || true
    exit 1
  fi
  sleep 10
done
echo "FAIL not ready after 30 min; see: docker logs $NAME" >&2
exit 1
