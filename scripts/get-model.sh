#!/usr/bin/env bash
# get-model.sh - download Qwen3.8 Flash Next (NVFP4) from Hugging Face if it is not already complete.
#
#   scripts/get-model.sh                  # into $DATA_DIR/models/Qwen3.8-Flash-Next-NVFP4 (DATA_DIR=/data)
#
# The repository is public (no login needed). The download runs the Hugging Face CLI inside the official vLLM image, so the
# host needs only docker. It is pinned to one commit and resumes when run again. A finished download is checked file by
# file against the repository's file list (names and sizes); a complete directory is left untouched.
#
# The model is licensed by NVIDIA under its own terms: read https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
REPO="${REPO:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
REVISION="${REVISION:-fc694b54fb0174e0913e6adf86691ef85a4ead47}"
TARGET="${TARGET:-$DATA_DIR/models/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-vllm/vllm-openai:v0.31.0}"
UIDGID="$(id -u):$(id -g)"

mkdir -p "$TARGET"

check() {   # prints "complete" or the number of missing/short files
  docker run --rm --entrypoint python3 -v "$TARGET":/target:ro "$IMAGE" - "$REPO" "$REVISION" <<'PY'
import json, os, sys, urllib.request
repo, rev = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true", timeout=60) as r:
    files = json.load(r)["siblings"]
bad = [f["rfilename"] for f in files
       if not os.path.isfile(os.path.join("/target", f["rfilename"]))
       or (f.get("size") is not None and os.path.getsize(os.path.join("/target", f["rfilename"])) != f["size"])]
print("complete" if not bad else "%d of %d files missing or incomplete" % (len(bad), len(files)))
PY
}

state="$(check)"
if [ "$state" = "complete" ]; then
  echo "OK   $TARGET is complete ($REPO @ ${REVISION:0:7})"
  exit 0
fi
echo "...  $state - downloading $REPO @ ${REVISION:0:7} (~133 GB) into $TARGET"
docker run --rm --user "$UIDGID" -e HF_HOME=/tmp/hf -v "$TARGET":/target --entrypoint hf "$IMAGE" \
  download "$REPO" --revision "$REVISION" --local-dir /target
state="$(check)"
if [ "$state" = "complete" ]; then
  echo "OK   $TARGET is complete"
else
  echo "FAIL $state - run the script again to resume" >&2
  exit 1
fi
