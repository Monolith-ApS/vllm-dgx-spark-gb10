# yarn-mrope-58879

Backport of vLLM PR #58879 (issue #58870). `MRotaryEmbedding` passes YaRN a 4x enlarged `max_position_embeddings`, and the YaRN
correction range is computed from that value: (16, 24) instead of (14, 22) for Qwen3.8 Flash Next. 9 of the 32 inverse
frequencies differ from the Hugging Face formula (by up to 25 % at factor 2 / 512K, 50 % at 3, 75 % at 4). It affects every
request on a context above the native 262,144 tokens. 256K and below run without YaRN and are not affected.

Files: `yarn_mrope_58879.diff` (the two source hunks of the PR), `yarn_math.py` (the math), `test_yarn_mrope.py`
(compares vLLM's cos/sin cache with transformers' YaRN for factors 2/3/4 on CPU; expect `MATCH` in a patched image):

    docker run --rm --gpus all --memory 8g --entrypoint python3 <image> /opt/vllm-dgx-spark/test_yarn_mrope.py

Drop this patch when the PR is in the vLLM release you build from.
