#!/usr/bin/env python3
"""Can C conversations all sit at full context at once? C agent sessions grow round-robin (one request at a time, ~16K
tokens of real source code per turn, like an agent reading files), each to TARGET tokens; then all C ask at once.

  fillc.py URL MODEL OUT C TARGET

A session whose prefix was evicted from the KV cache while the others grew pays a full re-prefill on its next turn:
the per-turn time to first token jumps (normal: ~10 s per 16K turn). At the end all C sessions send one more short turn
together and decode 256 tokens: vllm:num_preemptions_total must not move. Greedy, thinking off.
"""
import glob, json, os, sys, sysconfig, threading, time, urllib.error, urllib.request

url, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
C, target = int(sys.argv[4]), int(sys.argv[5])
TURN = 16000
# Filler: real source code read locally (default: this Python's standard library; FILL_SRC = another glob, e.g. a C tree).
FILL_SRC = os.environ.get("FILL_SRC", os.path.join(sysconfig.get_paths()["stdlib"], "*.py"))
src = "".join(open(f, encoding="utf-8", errors="replace").read() for f in sorted(glob.glob(FILL_SRC)))


def metric(name):
    with urllib.request.urlopen(url + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith(name):
                return float(line.split()[-1])
    return None


def chat(msgs, max_tokens):
    req = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": True,
           "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(), {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = ""; usage = {}
    with urllib.request.urlopen(r, timeout=900) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            if d.get("choices") and (d["choices"][0].get("delta") or {}).get("content"):
                if first is None:
                    first = time.time()
                text += d["choices"][0]["delta"]["content"]
    t1 = time.time()
    n = usage.get("completion_tokens", 0)
    return {"ttft": round((first or t1) - t0, 2), "decode": round((n - 1) / (t1 - first), 2) if first and n > 1 and t1 > first else None,
            "prompt": usage.get("prompt_tokens", 0), "completion": n, "text": text}


sessions = []
for i in range(C):
    sessions.append({"msgs": [{"role": "system", "content": f"Session {i}: you are reviewing a C code base, one file part per turn. Answer in one sentence."}],
                     "pos": (len(src) // C) * i, "ctx": 0, "turn": 0, "done": False})
p0 = metric("vllm:num_preemptions_total")
f = open(out, "w")
t_start = time.time()
while not all(s["done"] for s in sessions):
    for i, s in enumerate(sessions):
        if s["done"]:
            continue
        chars = int(TURN * 3.0)
        body = src[s["pos"]:s["pos"] + chars]
        s["pos"] = (s["pos"] + chars) % (len(src) - chars)
        s["msgs"].append({"role": "user", "content": f"File part {s['turn']}:\n```c\n{body}\n```\nWhat does this part do?"})
        try:
            r = chat(s["msgs"], 32)
        except urllib.error.HTTPError as e:  # e.g. the turn would pass max-model-len: this session is full
            err = e.read().decode("utf-8", "replace")[:300]
            s["msgs"].pop(); s["done"] = True
            row = {"session": i, "turn": s["turn"], "ctx": s["ctx"], "error": e.code, "detail": err}
            f.write(json.dumps(row) + "\n"); f.flush(); print(json.dumps(row), flush=True)
            continue
        s["msgs"].append({"role": "assistant", "content": r["text"]})
        s["ctx"] = r["prompt"] + r["completion"]
        row = {"session": i, "turn": s["turn"], "ctx": s["ctx"], "ttft": r["ttft"], "t": round(time.time() - t_start)}
        f.write(json.dumps(row) + "\n"); f.flush()
        print(json.dumps(row), flush=True)
        s["turn"] += 1
        if s["ctx"] + 1.4 * TURN + 512 > target:
            s["done"] = True
p1 = metric("vllm:num_preemptions_total")
# all sessions at full context, at once
res = [None] * C
def final(i):
    s = sessions[i]
    res[i] = chat(s["msgs"] + [{"role": "user", "content": "Summarize everything you have reviewed so far in detail."}], 256)
th = [threading.Thread(target=final, args=(i,)) for i in range(C)]
[t.start() for t in th]; [t.join() for t in th]
p2 = metric("vllm:num_preemptions_total")
ttfts = [r["ttft"] for r in map(json.loads, open(out)) if "ttft" in r]
summary = {"C": C, "target": target, "final_ctx": [s["ctx"] for s in sessions], "turn_ttft_max": max(ttfts),
           "turn_ttft_median": sorted(ttfts)[len(ttfts) // 2], "final_ttft": [r["ttft"] for r in res],
           "final_decode": [r["decode"] for r in res], "preemptions_fill": p1 - p0, "preemptions_final": p2 - p1,
           "minutes": round((time.time() - t_start) / 60, 1)}
f.write(json.dumps({"summary": summary}) + "\n")
print("SUMMARY", json.dumps(summary), flush=True)
