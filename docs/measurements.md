# Measurements

NVIDIA DGX Spark (GB10, 128 GB, DGX OS), the image built from this repository, MTP 7, thinking off unless stated. One profile
at a time, nothing else running.

## Decode speed

Code-writing test: "edit" returns a 250-line C file with small changes (3,000 completion tokens), "new" writes a new C module
(~1,500-1,800 tokens); greedy. "All at once" = as many identical requests in parallel as the profile's concurrency, total
throughput.

| Profile | 1 request, edit | 1 request, new | All at once, edit | All at once, new |
|---|---|---|---|---|
| `128k-c8` | 79.7 tok/s | 82.0 tok/s | 222.3 tok/s (8) | 238.2 tok/s (8) |
| `256k-c5` | 78.7 | 78.7 | 175.0 (5) | 193.0 (5) |
| `384k-c4` | 77.1 | 78.4 | 161.4 (4) | 171.7 (4) |
| `480k-c3` | 76.8 | 79.0 | 140.2 (3) | 148.4 (3) |
| `512k-c3` | 78.3 | 79.1 | 138.8 (3) | 148.3 (3) |
| `640k-c2` | 78.1 | 75.0 | 117.8 (2) | 109.6 (2) |
| `768k-c2` | 78.8 | 79.8 | 118.1 (2) | 115.2 (2) |
| `1024k-c1` | 79.1 | 82.9 | - | - |

## Capacity

`tools/fillc.py`: all sessions grow round-robin towards context - 6,000 tokens, then all send one more turn at once;
pass = 0 preemptions and slowest time to first token of that final turn <= 30 s.

| Profile | KV cache size | Needed (sessions x (context - 6,000)) | Sessions filled to | Preemptions | Time to first token, final turn | MemAvailable after start / after fill | Ready after |
|---|---|---|---|---|---|---|---|
| `128k-c8` | 1,000,913 tokens | 1,000,576 | 103-115K x 8 | 0 | 5.5-23.5 s | 6.1 GiB / 4.9 GiB | 172 s |
| `256k-c5` | 1,394,334 | 1,280,720 | 237-245K x 5 | 0 | 5.3-15.3 s | 5.0 GiB / 7.5 GiB | 172 s |
| `384k-c4` | 1,557,642 | 1,548,864 | 369-375K x 4 | 0 | 7.3-14.8 s | 5.0 GiB / 7.5 GiB | 172 s |
| `480k-c3` | 1,586,754 | 1,456,560 | 464-474K x 3 | 0 | 4.8-8.3 s | 6.0 GiB / 4.5 GiB | 172 s |
| `512k-c3` | 1,599,889 | 1,554,864 | 498-504K x 3 | 0 | 6.3-11.2 s | 6.0 GiB / 4.4 GiB | 172 s |
| `640k-c2` | 1,651,057 | 1,298,720 | 636-639K x 2 | 0 | 5.3-6.8 s | 6.1 GiB / 4.6 GiB | 161 s |
| `768k-c2` | 1,700,080 | 1,560,864 | 760-769K x 2 | 0 | 6.5-9.0 s | 5.8 GiB / 4.5 GiB | 161 s |
| `1024k-c1` | 1,686,709 | 1,042,576 | 1,028K x 1 | 0 | 7.2 s | 7.7 GiB / 5.6 GiB | 161 s |

Ready after = container start to API ready, weights in the page cache, compile cache warm.

Mamba cache mode on `512k-c3`: `all` and `align` gave the same KV cache size, decode speed and fill result.

## Tool calling and control prompts

Every profile: tool calling (`tools/tool_check.py`, 9 cases x 2) 18/18; control prompts (`tools/answer_check.py`) 4/4.

## Thinking on vs off

`tools/taskbench.py`: 10 single-prompt Python tasks with hidden tests, one run each, 1 request at a time,
`512k-c3`.

| | Thinking off | Thinking on |
|---|---|---|
| Tasks passed | 9/10 | 9/10 |
| Total time | 132 s | 1,955 s |
| Median time per task | 7.6 s | 132 s |
| Completion tokens | 7,847 | 68,767 |
| Decode speed | 60.6 tok/s | 35.2 tok/s |

Sampling used: thinking off `temperature 1.0, top_p 0.95, top_k 20`; thinking on `temperature 0.8, top_p 0.95, top_k 20,
presence_penalty 1.0` (the image defaults).
