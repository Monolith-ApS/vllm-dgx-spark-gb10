#!/usr/bin/env python3
"""Thinking vs non-thinking on the same single-prompt tasks: wall time, tokens, decode speed and whether the answer passes hidden tests.

  taskbench.py URL MODEL OUT.jsonl MODES REPEATS [TASKS]     MODES = think,nothink   URL = http://127.0.0.1:8000/v1

One request at a time (c1). Per run: TTFT, time to the first answer token (after the reasoning), total time, completion tokens,
reasoning characters, decode tok/s (completion tokens / time after the first token), hidden tests passed/total.
Sampling per mode = the masters' defaults: think 0.8/0.95/20 + presence 1.0 (think-kvfp8 spec), nothink 1.0/0.95/20 (kvfp8 spec).
The generated code runs locally in a subprocess (python -I, temp dir, 20 s timeout).
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request

SAMPLING = {"think": {"temperature": 0.8, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.0},
            "nothink": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}}
TAIL = "\n\nAnswer with the complete Python 3 code in one ```python block (standard library only, no input(), no prints needed)."

TASKS = {
    "intervals": ("Write `merge_intervals(intervals)` that takes a list of closed integer intervals [a, b] (a <= b, unsorted, possibly empty) "
                  "and returns the merged list sorted by start. Intervals that touch (e.g. [1,3] and [3,5]) or overlap merge; [1,2] and [3,4] do not.",
                  r'''
assert merge_intervals([]) == []
assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]
assert merge_intervals([[5,7],[1,2],[2,3],[6,6]]) == [[1,3],[5,7]]
assert merge_intervals([[1,10],[2,3],[4,5]]) == [[1,10]]
assert merge_intervals([[-5,-1],[-2,0]]) == [[-5,0]]
'''),
    "calc": ("Write `calc(s)` that evaluates an arithmetic expression string with non-negative integer literals, + - * /, parentheses, "
             "unary minus (also repeated, e.g. '--3') and arbitrary spaces. / is integer division truncating toward zero (like C). "
             "Normal precedence, left-associative. Do not use eval/exec/compile/ast.",
             r'''
src = open(__file__).read().split('# --- tests ---')[0]
assert not re.search(r'(?<![\w.])(eval|exec|compile)\s*\(', src) and 'import ast' not in src
assert calc('1 + 2 * 3') == 7
assert calc('(1+2)*3') == 9
assert calc('7/2') == 3 and calc('-7/2') == -3 and calc('7/-2') == -3
assert calc('2-3-4') == -5
assert calc('--3') == 3 and calc('-(2+3)*2') == -10
assert calc(' 100 / ( 3 - 1 ) / 5 ') == 10
assert calc('2*-3') == -6
assert calc('1-(-(-1))') == 0
assert calc('((((42))))') == 42
'''),
    "minwin": ("Write `min_window(s, t)` returning the shortest substring of s that contains every character of t with multiplicity. "
               "If several have the same length return the leftmost; return '' if none exists or t is empty.",
               r'''
assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'
assert min_window('a', 'aa') == ''
assert min_window('aa', 'aa') == 'aa'
assert min_window('abc', '') == ''
assert min_window('ab', 'b') == 'b'
assert min_window('cabwefgewcwaefgcf', 'cae') == 'cwae'
assert min_window('aaflslflsldkalskaaa', 'aaa') == 'aaa'
assert min_window('xyzxyz', 'zx') == 'zx'
'''),
    "csv": ("Write `parse_csv_line(line)` returning the list of fields of one RFC 4180 CSV line (no newlines inside). "
            "Fields may be quoted with double quotes; inside quotes a doubled quote \"\" means one quote and commas are literal. "
            "Unquoted fields are taken as is. An empty line is one empty field.",
            r'''
assert parse_csv_line('a,b,c') == ['a','b','c']
assert parse_csv_line('') == ['']
assert parse_csv_line('a,,c,') == ['a','','c','']
assert parse_csv_line('"x,y",z') == ['x,y','z']
assert parse_csv_line('"he said ""hi""",2') == ['he said "hi"','2']
assert parse_csv_line('""') == ['']
assert parse_csv_line('"",""') == ['','']
assert parse_csv_line(' a , b ') == [' a ',' b ']
'''),
    "topo": ("Write `build_order(tasks, deps)` where tasks is a list of unique strings and deps a list of pairs (a, b) meaning a must come before b. "
             "Return the lexicographically smallest valid order (a list). Raise ValueError if there is a cycle or a dependency names an unknown task.",
             r'''
assert build_order(['a','b','c'], []) == ['a','b','c']
assert build_order(['c','b','a'], [('c','a')]) == ['b','c','a']
assert build_order(['x','y','z','w'], [('z','y'),('w','x')]) == ['w','x','z','y']
for bad in ([('a','b'),('b','a')], [('a','q')]):
    try:
        build_order(['a','b'], bad); raise SystemExit('no ValueError')
    except ValueError:
        pass
assert build_order(['d','c','b','a'], [('d','a'),('c','a'),('b','a')]) == ['b','c','d','a']
'''),
    "roman": ("Write `to_roman(n)` for 1 <= n <= 3999 (ValueError otherwise) and `from_roman(s)` that accepts only canonical (standard minimal) "
              "upper-case Roman numerals and raises ValueError for anything else (e.g. 'IIII', 'IC', 'VX', 'MMMM', '', 'iv').",
              r'''
assert to_roman(1994) == 'MCMXCIV' and to_roman(3999) == 'MMMCMXCIX' and to_roman(4) == 'IV'
assert all(from_roman(to_roman(i)) == i for i in range(1, 4000))
for bad in ['IIII','IC','VX','MMMM','','iv','IIV','XM','VV','LL','DD','IL','XD','CMC','IXI']:
    try:
        from_roman(bad); raise SystemExit('accepted ' + bad)
    except ValueError:
        pass
for bad in [0, 4000, -1]:
    try:
        to_roman(bad); raise SystemExit('accepted %d' % bad)
    except ValueError:
        pass
'''),
    "justify": ("Write `full_justify(words, width)` (classic text justification): greedily pack words into lines of exactly `width` characters; "
                "distribute extra spaces between words as evenly as possible, extra ones go to the left gaps; a line with one word is left-justified "
                "and padded; the last line is left-justified with single spaces and padded with spaces to width. Return the list of lines.",
                r'''
assert full_justify(['This','is','an','example','of','text','justification.'], 16) == ['This    is    an','example  of text','justification.  ']
assert full_justify(['What','must','be','acknowledgment','shall','be'], 16) == ['What   must   be','acknowledgment  ','shall be        ']
assert full_justify(['Science','is','what','we','understand','well','enough','to','explain','to','a','computer.','Art','is','everything','else','we','do'], 20) == ['Science  is  what we','understand      well','enough to explain to','a  computer.  Art is','everything  else  we','do                  ']
assert full_justify(['a'], 1) == ['a']
assert full_justify(['a','b','c','d'], 3) == ['a b','c d']
'''),
    "skyline": ("Write `get_skyline(buildings)` for the skyline problem: buildings are [left, right, height] with left < right, height > 0. "
                "Return the key points [x, h] of the outer contour sorted by x, with no two consecutive points of equal height, ending with [x, 0].",
                r'''
assert get_skyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]) == [[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
assert get_skyline([[0,2,3],[2,5,3]]) == [[0,3],[5,0]]
assert get_skyline([]) == []
assert get_skyline([[1,2,1],[1,2,2],[1,2,3]]) == [[1,3],[2,0]]
assert get_skyline([[1,5,3],[2,4,3]]) == [[1,3],[5,0]]
assert get_skyline([[1,3,2],[3,5,4],[5,7,2]]) == [[1,2],[3,4],[5,2],[7,0]]
'''),
    "wildcard": ("Write `wild_match(s, p)` returning True if the whole string s matches pattern p, where '?' matches exactly one character and "
                 "'*' matches any sequence (including empty). It must handle len(s), len(p) up to 2000 quickly.",
                 r'''
assert wild_match('aa','a') is False and wild_match('aa','*') is True and wild_match('cb','?a') is False
assert wild_match('adceb','*a*b') is True and wild_match('acdcb','a*c?b') is False
assert wild_match('','') is True and wild_match('','*') is True and wild_match('','?') is False
assert wild_match('abc','abc*') is True and wild_match('abc','a**c') is True
import time as _t; t0 = _t.time()
assert wild_match('a'*2000, '*'*1000 + 'b') is False
assert wild_match('ab'*1000, '*a*b'*500) is True
assert _t.time() - t0 < 5
'''),
    "rain2d": ("Write `trap_rain_water(height_map)` returning how much rain water a 2D elevation map (list of lists of non-negative ints, possibly empty) "
               "can trap; water leaks off the border.",
               r'''
assert trap_rain_water([[1,4,3,1,3,2],[3,2,1,3,2,4],[2,3,3,2,3,1]]) == 4
assert trap_rain_water([[3,3,3,3,3],[3,2,2,2,3],[3,2,1,2,3],[3,2,2,2,3],[3,3,3,3,3]]) == 10
assert trap_rain_water([]) == 0 and trap_rain_water([[5]]) == 0
assert trap_rain_water([[12,13,1,12],[13,4,13,12],[13,8,10,12],[12,13,12,12],[13,13,13,13]]) == 14
assert trap_rain_water([[5,5,5],[5,1,5],[5,5,5]]) == 4
'''),
}


def ask(url, model, prompt, mode):
    body = {"model": model, "messages": [{"role": "user", "content": prompt + TAIL}], "max_tokens": 24000, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": mode == "think"}, **SAMPLING[mode]}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    t0 = time.time()
    ttft = tans = None
    reasoning, content, usage = [], [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            j = json.loads(line[5:])
            if j.get("usage"):
                usage = j["usage"]
            for ch in j.get("choices", []):
                d = ch.get("delta", {})
                rc = d.get("reasoning_content") or d.get("reasoning")
                if rc:
                    ttft = ttft or time.time() - t0
                    reasoning.append(rc)
                if d.get("content"):
                    ttft = ttft or time.time() - t0
                    tans = tans or time.time() - t0
                    content.append(d["content"])
    total = time.time() - t0
    ct = usage.get("completion_tokens", 0)
    return {"ttft": round(ttft or 0, 2), "t_answer": round(tans or total, 2), "total_s": round(total, 2), "completion_tokens": ct,
            "reasoning_chars": len("".join(reasoning)), "answer_chars": len("".join(content)),
            "tok_s": round(ct / max(total - (ttft or 0), 1e-6), 1), "content": "".join(content)}


def grade(content, tests):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", content, re.S)
    if not blocks:
        return 0, "no code block"
    code = max(blocks, key=len)
    cases = [c for c in tests.strip().split("\n") if c.strip()]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sol.py")
        open(p, "w", encoding="utf-8").write("import re, sys\n" + code + "\n# --- tests ---\n" + tests)
        r = subprocess.run([sys.executable, "-I", p], capture_output=True, text=True, timeout=60, cwd=d)
    if r.returncode == 0:
        return 1, "pass"
    err = (r.stderr or r.stdout).strip().splitlines()
    return 0, " | ".join(x.strip() for x in err[-2:])[:300] if err else "exit %d" % r.returncode


def main():
    url, model, out, modes, reps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5])
    names = sys.argv[6].split(",") if len(sys.argv) > 6 else list(TASKS)
    for rep in range(reps):
        for name in names:
            for mode in modes:
                prompt, tests = TASKS[name]
                try:
                    res = ask(url, model, prompt, mode)
                    try:
                        ok, why = grade(res["content"], tests)
                    except subprocess.TimeoutExpired:
                        ok, why = 0, "timeout"
                except Exception as ex:  # noqa: BLE001
                    res, ok, why = {"error": str(ex)[:300]}, 0, "request failed"
                rec = {"task": name, "mode": mode, "rep": rep, "pass": ok, "why": why, **{k: v for k, v in res.items() if k != "content"}}
                open(out, "a", encoding="utf-8").write(json.dumps(dict(rec, content=res.get("content", ""))) + "\n")
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
