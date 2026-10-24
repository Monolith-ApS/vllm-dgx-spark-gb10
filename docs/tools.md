# Test tools

Run from any machine that reaches the server (`LLM_API_URL`, default `http://127.0.0.1:8000/v1`; `LLM_API_KEY` if the server
has one). Python 3 standard library only. The examples use the model name `qwen3.8-flash-next`: use the name your server reports
(`curl -s http://localhost:8000/v1/models`). The tools work with any model served over the OpenAI API.

| Tool | Usage |
|---|---|
| `tools/answer_check.py` | `python3 tools/answer_check.py qwen3.8-flash-next` - 4 control prompts |
| `tools/tool_check.py` | `python3 tools/tool_check.py --model qwen3.8-flash-next --repeats 2` - tool-calling cases; `--parallel N` sends N at a time (default 1, keep it at or below the profile's concurrency) |
| `tools/fillc.py` | `python3 tools/fillc.py http://127.0.0.1:8000 qwen3.8-flash-next out.jsonl 3 518288` - capacity fill test |
| `tools/taskbench.py` | `python3 tools/taskbench.py http://127.0.0.1:8000/v1 qwen3.8-flash-next out.jsonl nothink,think 1` |

`tools/taskbench.py` runs the generated code locally in a subprocess (`python -I`, temporary directory, timeout).
