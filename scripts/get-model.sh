#!/usr/bin/env bash
# get-model.sh - download a model's weights from Hugging Face if they are not already complete.
#
#   scripts/get-model.sh [model]          # into $DATA_DIR/models/<weights folder> (DATA_DIR=/data)
#
# The repository and the pinned revision come from models/<model>/model.env. The download runs the Hugging Face CLI inside
# the official vLLM image, so the host needs only docker. It resumes when run again. A finished download is checked file by
# file against the repository's file list (names and sizes); a complete directory is left untouched. Models that need a
# Hugging Face login are not handled here.
#
# The model is licensed by its publisher under its own terms: read the model card at https://huggingface.co/<repository>
set -euo pipefail
. "$(dirname "$0")/lib.sh"
select_model "${1:-}"

REPO="${REPO:-$HF_REPO}"
REVISION="${REVISION:-$HF_REVISION}"
TARGET="${TARGET:-$MODEL_DIR}"
DL_IMAGE="${DL_IMAGE:-$DOWNLOAD_IMAGE}"
UIDGID="$(id -u):$(id -g)"

mkdir -p "$TARGET"

check() {   # prints "complete" or the number of missing/short files
  docker run --rm --entrypoint python3 -v "$TARGET":/target:ro "$DL_IMAGE" - "$REPO" "$REVISION" <<'PY'
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
echo "...  $state - downloading $REPO @ ${REVISION:0:7} (~${WEIGHTS_GB} GB) into $TARGET"
docker run --rm --user "$UIDGID" -e HF_HOME=/tmp/hf -v "$TARGET":/target --entrypoint hf "$DL_IMAGE" \
  download "$REPO" --revision "$REVISION" --local-dir /target
state="$(check)"
if [ "$state" = "complete" ]; then
  echo "OK   $TARGET is complete"
else
  echo "FAIL $state - run the script again to resume" >&2
  exit 1
fi
