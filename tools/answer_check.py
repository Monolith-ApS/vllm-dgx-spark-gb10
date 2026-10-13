"""Control prompts: does the booted model actually answer correctly?

LLM_API_URL=http://127.0.0.1:8000/v1 python answer_check.py <served-model>   -> one line per prompt, then ANSWERS ok=N/M

No low max_tokens cap: reasoning models spend tokens thinking first, and a cap would look like a broken model.
"""
import json
import os
import re
import sys
import time
import urllib.request

PROXY = os.environ.get("LLM_API_URL", "http://127.0.0.1:8000/v1") + "/chat/completions"

PROMPTS = [
    ("arith", "What is 17 * 24 + 139? Reply with only the number.", lambda t: re.search(r"(?<!\d)547(?!\d)", t)),
    ("capital", "What is the capital of Denmark? Reply with one word.", lambda t: "copenhagen" in t.lower() or "københavn" in t.lower()),
    ("code", "Write a Python function is_prime(n) that returns True if n is prime. Reply with only the code.",
     lambda t: "def is_prime" in t and "return" in t),
    ("danish", "Oversæt til engelsk og svar kun med oversættelsen: 'Hunden sover under bordet.'",
     lambda t: "dog" in t.lower() and "table" in t.lower()),
]


def ask(model, prompt, key):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16384}
    req = urllib.request.Request(PROXY, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=900))
    msg = out["choices"][0]["message"]
    return (msg.get("content") or ""), out["choices"][0].get("finish_reason"), out.get("usage", {}).get("completion_tokens"), time.time() - t0


def main(model):
    key = (os.environ.get("LLM_API_KEY") or "none").strip()   # vLLM without --api-key accepts any value
    if not key or any(c in key for c in "\r\n"):  # http.client's error for a bad header value would quote the key
        sys.exit("API key is empty or malformed")
    ok = 0
    for name, prompt, check in PROMPTS:
        try:
            text, fin, toks, secs = ask(model, prompt, key)
            good = bool(check(text))
            ok += good
            print("  %-8s %s finish=%s tokens=%s %.1fs answer=%r" % (name, "OK  " if good else "FAIL", fin, toks, secs,
                                                                     re.sub(r"\s+", " ", text.strip())[:120]))
        except Exception as e:  # noqa: BLE001
            print("  %-8s ERROR %s" % (name, str(e)[:200]))
    print("ANSWERS ok=%d/%d" % (ok, len(PROMPTS)))
    return 0 if ok == len(PROMPTS) else 1  # scripts can gate on it


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
