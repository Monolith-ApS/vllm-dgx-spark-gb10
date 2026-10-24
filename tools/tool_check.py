#!/usr/bin/env python3
"""Tool-calling check against an OpenAI-compatible endpoint (LLM_API_URL, default http://127.0.0.1:8000/v1). Fixed cases, each run R times; a case passes
when the model calls exactly the expected tools with the expected arguments (or answers without a tool when none is
needed). Multi-turn cases feed a tool result back and check the follow-up call. Prints one JSON line per case and a
summary. Key from LLM_API_KEY (default 'none'); never printed.
usage: python tool_check.py --model <served-model> [--repeats 3] [--parallel 1] [--label x]
--parallel: requests at the same time (default 1 = one after the other); keep it at or below the profile's concurrency."""
import argparse, json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("LLM_API_URL", "http://127.0.0.1:8000/v1")
KEY = os.environ.get("LLM_API_KEY") or "none"


def fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}


S, I = {"type": "string"}, {"type": "integer"}
TOOLS = [
    fn("get_weather", "Current weather for a city.", {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    fn("read_file", "Read a file from the repository.", {"path": S}, ["path"]),
    fn("run_tests", "Run the test suite, optionally filtered.", {"filter": S}, []),
    fn("create_ticket", "Create a work item.", {"title": S, "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                                                "labels": {"type": "array", "items": S}}, ["title", "priority"]),
    fn("search_code", "Search the code base for a regex.", {"pattern": S, "max_results": I}, ["pattern"]),
]
FILLER = ("The ledger service reconciles nightly batches against the bank feed and writes a summary row per account. " * 400)

# (name, messages, expected calls [(tool, {arg: value})] in any order; None = no tool call expected)
CASES = [
    ("single", [{"role": "user", "content": "What's the weather in Aarhus right now, in celsius?"}],
     [("get_weather", {"city": "Aarhus", "unit": "celsius"})]),
    ("parallel", [{"role": "user", "content": "Read both src/app.py and src/db.py."}],
     [("read_file", {"path": "src/app.py"}), ("read_file", {"path": "src/db.py"})]),
    ("enum_array", [{"role": "user", "content": "Create a high priority ticket titled 'Fix login timeout' with labels auth and bug."}],
     [("create_ticket", {"title": "Fix login timeout", "priority": "high", "labels": ["auth", "bug"]})]),
    ("int_arg", [{"role": "user", "content": "Search the code for the regex 'def (load|save)_config' and return at most 5 results."}],
     [("search_code", {"pattern": "def (load|save)_config", "max_results": 5})]),
    ("no_tool", [{"role": "user", "content": "What is 17 times 23? Answer with just the number."}], None),
    ("danish", [{"role": "user", "content": "Kør testene, men kun dem der matcher 'payment'."}],
     [("run_tests", {"filter": "payment"})]),
    ("multi_turn", [
        {"role": "user", "content": "Run the tests. If any fail, read the file of the first failing test."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                               "function": {"name": "run_tests", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "2 passed, 1 failed: tests/test_invoice.py::test_rounding FAILED"}],
     [("read_file", {"path": "tests/test_invoice.py"})]),
    ("multi_turn_2", [
        {"role": "user", "content": "Find where retry_policy is defined and open that file."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                               "function": {"name": "search_code", "arguments": "{\"pattern\": \"retry_policy\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "lib/net/retry.py:12: def retry_policy(max_attempts):"}],
     [("read_file", {"path": "lib/net/retry.py"})]),
    ("long_ctx", [{"role": "user", "content": "Background notes (ignore):\n" + FILLER +
                   "\n\nNow: read the file config/ledger.yaml."}],
     [("read_file", {"path": "config/ledger.yaml"})]),
]


def call(model, messages):
    body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "max_tokens": 8192}
    req = urllib.request.Request(URL + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)["choices"][0]["message"]


def norm(v):
    return v.strip().lower() if isinstance(v, str) else [norm(x) for x in v] if isinstance(v, list) else v


def check(msg, expected):
    calls = msg.get("tool_calls") or []
    if expected is None:
        return not calls and "391" in (msg.get("content") or ""), "content=%r" % (msg.get("content") or "")[:80]
    got = []
    for c in calls:
        try:
            got.append((c["function"]["name"], json.loads(c["function"]["arguments"] or "{}")))
        except json.JSONDecodeError:
            return False, "bad JSON args: %r" % c["function"]["arguments"][:120]
    want = [(n, {k: norm(v) for k, v in a.items()}) for n, a in expected]
    have = [(n, {k: norm(v) for k, v in a.items()}) for n, a in got]
    ok = len(have) == len(want) and all(any(h[0] == w[0] and all(h[1].get(k) == v for k, v in w[1].items()) for h in have) for w in want)
    return ok, "calls=%s" % json.dumps(got)[:200]


ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--parallel", type=int, default=1, help="requests at the same time; at most the server's concurrency")
ap.add_argument("--label", default="")
a = ap.parse_args()
total = passed = 0
jobs = [(name, msgs, exp, i) for name, msgs, exp in CASES for i in range(a.repeats)]


def run(job):
    name, msgs, exp, i = job
    t0 = time.time()
    try:
        ok, why = check(call(a.model, msgs), exp)
    except Exception as e:  # an HTTP or parse error counts as a failure
        ok, why = False, "error %s" % str(e)[:160]
    return name, i, ok, why, round(time.time() - t0, 1)


with ThreadPoolExecutor(max(1, a.parallel)) as ex:
    res = list(ex.map(run, jobs))
by = {}
for name, i, ok, why, s in res:
    by.setdefault(name, []).append(ok)
    if not ok:
        print(json.dumps({"label": a.label, "case": name, "run": i, "ok": ok, "why": why, "s": s}), flush=True)
total, passed = len(res), sum(r[2] for r in res)
print(json.dumps({"label": a.label, "model": a.model, "event": "tool_summary", "passed": passed, "total": total,
                  "per_case": {k: "%d/%d" % (sum(v), len(v)) for k, v in by.items()}}), flush=True)
