# pre-build.sh - model-specific step of scripts/build.sh (sourced by it; ROOT and MODEL are set).
#
# Qwen3.8 Flash Next: fetch the blazux overlay at its pinned commit and hand it to the Dockerfile as a build context.
# Sets BUILD_ARGS, which build.sh appends to `docker build`.
BLAZUX_REPO="https://github.com/blazux/qwen3.8-Flash-DGX.git"
BLAZUX_COMMIT="5108d90dedf15a31aa20432319f151b2fdda17a3"
WORK="$ROOT/.build"

mkdir -p "$WORK"
if [ ! -d "$WORK/blazux/.git" ]; then
  git clone --quiet "$BLAZUX_REPO" "$WORK/blazux"
fi
git -C "$WORK/blazux" fetch --quiet origin "$BLAZUX_COMMIT" 2>/dev/null || git -C "$WORK/blazux" fetch --quiet origin
git -C "$WORK/blazux" checkout --quiet "$BLAZUX_COMMIT"
echo "blazux overlay at $(git -C "$WORK/blazux" rev-parse --short HEAD)"
BUILD_ARGS=(--build-context "blazux=$WORK/blazux")
