# lib.sh - shared by the scripts: finds the model and loads its settings. Sourced, not run.
#
# The model is the first argument of the script (a folder name under models/). With no argument the only model in the
# repository is used; with several models the argument is required. Settings come from models/<model>/model.env.
# VLLM selects the vLLM release (one of the model's VLLM_RELEASES; default its VLLM_DEFAULT).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

models_available() {
  local d
  for d in "$ROOT"/models/*/; do
    if [ -f "${d}model.env" ]; then basename "$d"; fi
  done
}

# select_model [name]: sets MODEL and VLLM, loads models/$MODEL/model.env
select_model() {
  MODEL="${1:-}"
  if [ -z "$MODEL" ]; then
    local list
    list="$(models_available)"
    if [ "$(printf '%s\n' "$list" | grep -c .)" -eq 1 ]; then
      MODEL="$list"
    else
      echo "which model? usage: $(basename "$0") <model>   available: $(printf '%s\n' "$list" | paste -sd' ' -)" >&2
      exit 2
    fi
  fi
  if [ ! -f "$ROOT/models/$MODEL/model.env" ]; then
    echo "unknown model '$MODEL' (available: $(models_available | paste -sd' ' -))" >&2
    exit 2
  fi
  local served="${SERVED_NAME:-}"                    # a SERVED_NAME given in the environment wins over the model's
  # shellcheck disable=SC1090
  . "$ROOT/models/$MODEL/model.env"
  SERVED_NAME="${served:-$SERVED_NAME}"
  VLLM="${VLLM:-$VLLM_DEFAULT}"
  case " $VLLM_RELEASES " in
    *" $VLLM "*) ;;
    *) echo "$MODEL has no image for vLLM '$VLLM' (available: $VLLM_RELEASES)" >&2; exit 2 ;;
  esac
  DATA_DIR="${DATA_DIR:-/data}"
  NAME="${NAME:-vllm-gb10}"                          # one model runs at a time: one fixed container name
  IMAGE="${IMAGE:-vllm-dgx-spark-gb10:$MODEL-vllm$VLLM}"
  MODEL_DIR="${MODEL_DIR:-$DATA_DIR/models/$WEIGHTS_DIR}"
  # compile caches per image: a cache written by one vLLM build must not be read by another (docs/vllm-versions.md)
  CACHE_DIR="${CACHE_DIR:-$DATA_DIR/cache/$MODEL-vllm$VLLM}"
}
