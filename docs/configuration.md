# Configuration

What can be set when starting a model, the same for every model. Each model's own settings (its profiles, thinking mode,
sampling defaults) are in its folder: [models/qwen3.8-flash-next/configuration.md](../models/qwen3.8-flash-next/configuration.md).

## Profiles

A model comes with ready-made profiles: a context window (how much text one conversation can hold) combined with the number of
conversations that can use the full window at the same time. `PROFILE` selects one; `PROFILE=list scripts/run.sh <model>`
prints the model's profiles, and the model's README says which to choose.

```bash
PROFILE=256k-c5 scripts/run.sh qwen3.8-flash-next
```

## Environment variables

For the scripts, set before the command (`PORT=8001 scripts/run.sh`):

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE` | the model's default | see above |
| `VLLM` | the model's default | vLLM release, when the model offers more than one ([vllm-versions.md](vllm-versions.md)); use the same for build and run |
| `THINKING` | `0` | `1`: the model thinks by default for requests that do not say otherwise (models that support it) |
| `PORT` | `8000` | host port |
| `API_KEY` | none | when set, clients must send it as Bearer token ([how](security.md)) |
| `SERVED_NAME` | the model's name | model name in the API |
| `DRY_RUN` | `0` | `1` prints the server command line the container would run and starts nothing |
| `DATA_DIR` | `/data` | root of models and caches on the host ([layout](installation.md#directory-layout-on-the-host)) |
| `MODEL_DIR` | `$DATA_DIR/models/<weights folder>` | use model weights that are somewhere else |
| `NAME` | `vllm-gb10` | container name; one model runs at a time, starting another replaces it |
| `IMAGE` | `vllm-dgx-spark-gb10:<model>-vllm<release>` | the image to run |
| `CACHE_DIR` | `$DATA_DIR/cache/<model>-vllm<release>` | compile caches of this image |

A model can accept more variables of its own (for example memory settings); they are in its configuration page, and
`scripts/run.sh` passes them on to the container when they are set.

## Adding a model or a vLLM release

A model is a folder under `models/`, a vLLM release one more Dockerfile in it; nothing needs to change in the scripts. How:
[image.md](image.md#adding-a-model).
