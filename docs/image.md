# How an image is built, and how to add a model or a vLLM release

## Building an image

An image is the official vLLM container with our patches on top. Build it on the DGX Spark itself:

```bash
scripts/build.sh qwen3.8-flash-next                # vLLM 0.32.0, the default
VLLM=0.31 scripts/build.sh qwen3.8-flash-next      # vLLM 0.31.0
```

The first gives the image `vllm-dgx-spark-gb10:qwen3.8-flash-next-vllm0.32`, the second `...-vllm0.31`. Both can be on the
machine at the same time; `scripts/run.sh` starts the one that matches `VLLM` (so use the same `VLLM` for build and run). A
build takes about 20-30 minutes. What goes into the image: [architecture](../models/qwen3.8-flash-next/architecture.md#image-layers);
the exact versions: [pins](../models/qwen3.8-flash-next/pins.md).

## Which vLLM release

The two releases below are offered for Qwen3.8 Flash Next. The numbers are from all eight profiles on one DGX Spark; the full
comparison is in [vLLM 0.31.0 or 0.32.0](../models/qwen3.8-flash-next/vllm-versions.md).

### vLLM 0.32.0 (default)

Pros:
- Measured and verified from this repository's own build: all profiles, fill tests, tool calling
  ([numbers](../models/qwen3.8-flash-next/measurements.md)).
- Starts about 7-9 s faster after a restart of the same profile.
- 1-6 % more room for conversation memory (KV cache), and a little more free memory.
- Newer vLLM release; the starting point for releases after it.

Cons:
- Decodes at the same speed as 0.31.0, not faster (97-103 % per profile).
- Needs two fixes of our own: without them every context window above 256K crashes at ~265K tokens, and engine start-up
  takes 50-57 s instead of 33-43 s ([why](vllm-versions.md#problems-we-found-in-vllm-0320); the fixes: [mtp-draft-rope](../gb10/mtp-draft-rope/README.md),
  [cutedsl-warmup](../gb10/cutedsl-warmup/README.md)).

### vLLM 0.31.0 (`VLLM=0.31`)

Pros:
- Measured and verified from this repository's own build: all profiles, fill tests, tool calling
  ([measurements](../models/qwen3.8-flash-next/measurements.md)).
- Runs every context window without extra fixes to vLLM.

Cons:
- Starts about 7-9 s slower than 0.32.0 (about 127 s against 118 s until the server answers).
- About 1-6 % less room for conversation memory (KV cache) than 0.32.0.

## Repository layout

```
gb10/                        layers shared by the models (anything specific to the GB10 chip, not to one model)
  <layer>/                   one folder per layer, the same for every vLLM release; a file that has to differ per release
                             carries the release in its name (serving_think_guard-v031.diff, -v032.diff), and the README
                             says which releases the layer is for
models/<model>/
  model.env                  the model's settings for the scripts: name, Hugging Face repository and revision, weights
                             folder, default profile, vLLM releases offered and the default
  Dockerfile.vllm-<release>  the image, one per vLLM release
  overlay/                   model-specific patch files; release-specific ones carry the release in their name
  serve.py, profiles.json    the container's entrypoint (knows the differences between the releases) and the profiles
  pre-build.sh               optional: runs before `docker build` (fetch pinned sources, set extra build arguments)
  README.md, configuration.md, measurements.md, vllm-versions.md, architecture.md, pins.md   the model's documentation
scripts/                     the same scripts for every model and release: install, setup-host, get-model, build, run
tools/                       test tools (work with any model served over the OpenAI API)
docs/                        documentation that is not about one model
```

Why not one folder per vLLM release (`gb10/v0.31/`, `gb10/v0.32/`)? Most of a layer is the same on every release
(`gb10-decode` is one script that finds the code of either release itself; `think_guard.py` is identical). Copying whole folders
per release would mean every fix has to be made in several places. Only the files that really differ exist per release, side
by side, so a diff between them shows exactly what the new release changed.

## Adding a model

1. Copy `models/qwen3.8-flash-next/` to `models/<new-model>/` and change what differs. The folder name is the name the scripts
   take (`scripts/install.sh <new-model>`).
2. `model.env`: display name, served name, Hugging Face repository and the exact revision to download, the folder the weights
   land in under `$DATA_DIR/models/`, their size, the default profile, the vLLM releases offered and the default.
   `PASS_ENV` lists the variables of `serve.py` that `scripts/run.sh` should hand on to the container.
3. `Dockerfile.vllm-<release>`, `serve.py`, `profiles.json`: how the image is built and how the server is started for each
   profile. Layers that are not specific to the model belong in `gb10/` and are used by the Dockerfiles that need them.
4. Pin everything the build downloads (base image digest, source commits) and write the pins in `pins.md`.
5. Run `scripts/install.sh <new-model>` on a Spark, then the tools in [tools.md](tools.md), and put the results in the model's
   `measurements.md`. Add the model to the table in the [README](../README.md).

## Adding a vLLM release to a model

1. Copy the model's newest `Dockerfile.vllm-<release>` to `Dockerfile.vllm-<new>`, change the `FROM` digest, and fix every
   step that no longer applies (each step fails the build when its anchor or hunk does not match). Put files that have to
   differ next to the old ones with the new release in their name.
2. Teach `serve.py` the command-line differences of the new release (`RELEASE_SETTINGS`).
3. Add the release to `VLLM_RELEASES` in `model.env` and to `pins.md`.
4. Build with `VLLM=<new>` and run the full test (all profiles, fill test, tools); write the comparison in the model's
   `vllm-versions.md` and the general findings in [vllm-versions.md](vllm-versions.md). Change `VLLM_DEFAULT` only when the
   new release is at least as good.

The scripts need no change for either: they read everything model-specific from `model.env` and the files in the model's
folder.
