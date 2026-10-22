# Measurements

NVIDIA DGX Spark (GB10, 128 GB, DGX OS), the image built from this repository, MTP 7, thinking off unless stated. One profile
at a time, nothing else running.

Decode speed, capacity, tool calling and control prompts: measured 2026-10-22 on DGX OS with the 7.0 kernel
(7.0.0-1019-nvidia), NVIDIA driver 580.178.04 and the 2026-10 platform firmware (EC 0x03000d07, SoC 0x02009e0f). Same vLLM
build, patches and command line per profile as the image; the requests went through an OpenAI-compatible proxy on the
local network. See [Platform update](#platform-update-2026-10) for the change against the first measurement (2026-10-13).

## Decode speed

Code-writing test: "edit" returns a 250-line C file with small changes (3,000 completion tokens), "new" writes a new C module
(~1,500-1,800 tokens); greedy. "All at once" = as many identical requests in parallel as the profile's concurrency, total
throughput.

| Profile | 1 request, edit | 1 request, new | All at once, edit | All at once, new |
|---|---|---|---|---|
| `128k-c8` | 79.3 tok/s | 75.2 tok/s | 223.1 tok/s (8) | 256.1 tok/s (8) |
| `256k-c5` | 78.9 | 81.6 | 177.6 (5) | 189.1 (5) |
| `384k-c4` | 79.4 | 82.8 | 164.6 (4) | 176.4 (4) |
| `480k-c3` | 78.8 | 80.0 | 138.7 (3) | 148.9 (3) |
| `512k-c3` | 79.1 | 79.7 | 140.6 (3) | 151.8 (3) |
| `640k-c2` | 79.2 | 72.9 | 117.7 (2) | 112.3 (2) |
| `768k-c2` | 79.1 | 82.9 | 116.4 (2) | 117.7 (2) |
| `1024k-c1` | 78.3 | 78.9 | - | - |

One run per cell; single runs of the same test vary by a few percent (up to ~8 % for "new").

## Capacity

`tools/fillc.py`: all sessions grow round-robin towards context - 6,000 tokens, then all send one more turn at once;
pass = 0 preemptions and slowest time to first token of that final turn <= 30 s.

| Profile | KV cache size | Needed (sessions x (context - 6,000)) | Sessions filled to | Preemptions | Time to first token, final turn | MemAvailable after start / after fill | Ready after |
|---|---|---|---|---|---|---|---|
| `128k-c8` | 1,016,233 tokens | 1,000,576 | 103-115K x 8 | 0 | 4.4-21.6 s | 5.6 GiB / 4.9 GiB | 130 s |
| `256k-c5` | 1,412,413 | 1,280,720 | 237-245K x 5 | 0 | 5.0-15.3 s | 4.5 GiB / 3.7 GiB | 130 s |
| `384k-c4` | 1,570,327 | 1,548,864 | 369-375K x 4 | 0 | 4.9-12.2 s | 4.7 GiB / 3.9 GiB | 127 s |
| `480k-c3` | 1,586,754 | 1,456,560 | 464-474K x 3 | 0 | 5.5-8.8 s | 5.9 GiB / 4.9 GiB | 130 s |
| `512k-c3` | 1,597,186 | 1,554,864 | 498-504K x 3 | 0 | 7.0-11.3 s | 6.0 GiB / 5.0 GiB | 124 s |
| `640k-c2` | 1,676,371 | 1,298,720 | 636-639K x 2 | 0 | 5.4-6.9 s | - / 4.7 GiB | 127 s |
| `768k-c2` | 1,714,537 | 1,560,864 | 760-769K x 2 | 0 | 8.1-8.7 s | 5.9 GiB / 4.7 GiB | 124 s |
| `1024k-c1` | 1,683,713 | 1,042,576 | 1,028K x 1 | 0 | 5.7 s | 7.7 GiB / 6.1 GiB | 127 s |

Ready after = container start to API ready, weights in the page cache, compile cache warm (an existing, stopped container
started again; 2026-10-13 counted from `docker run` with 10 s polling, 161-172 s). `640k-c2`: MemAvailable after start was
not read.

Mamba cache mode on `512k-c3`: `all` and `align` gave the same KV cache size, decode speed and fill result.

## Tool calling and control prompts

Every profile: tool calling (`tools/tool_check.py`, 9 cases x 2) 18/18; control prompts (`tools/answer_check.py`) 4/4.

## Platform update (2026-10)

All profiles were measured twice: 2026-10-13 and 2026-10-22, after the DGX OS update to the 7.0 kernel, driver 580.178.04
and new platform firmware. Same image, model and command lines. Nothing changed beyond run-to-run variation:

- Decode, 1 request: -8.3 % to +5.6 % per cell, mean +0.4 %. The largest moves are single "new" runs (`128k-c8` 82.0 -> 75.2,
  `384k-c4` 78.4 -> 82.8).
- Decode, all at once: -2.0 % to +7.5 % per cell, mean +1.3 %.
- KV cache size: -0.2 % to +1.6 %. Every profile still holds all its sessions at full context with 0 preemptions.
- Tool calling 18/18 and control prompts 4/4 on every profile, both times.

## Thinking on vs off

`tools/taskbench.py`: 10 single-prompt Python tasks with hidden tests, one run each, 1 request at a time.

MTP draft depth with thinking on (`1024k-c1-think`, 24,000-token cap per task):

| MTP draft tokens | Decode speed | Mean accepted length | Tasks passed | Total time |
|---|---|---|---|---|
| 3 (`-think` default) | 43.4 tok/s | 2.27 | 7/10 | 2,384 s |
| 5 | 39.6 tok/s | 2.63 | 9/10 | 2,296 s |
| 7 | 35.1 tok/s | 2.65 | 9/10 | 2,576 s |

MTP 3 was the fastest on 9 of the 10 tasks. The draft depth does not change what the model generates; the failed tasks
reached the 24,000-token cap without an answer.

Thinking on vs off, MTP 7, `512k-c3`:

| | Thinking off | Thinking on |
|---|---|---|
| Tasks passed | 9/10 | 9/10 |
| Total time | 132 s | 1,955 s |
| Median time per task | 7.6 s | 132 s |
| Completion tokens | 7,847 | 68,767 |
| Decode speed | 60.6 tok/s | 35.2 tok/s |

Sampling used: thinking off `temperature 1.0, top_p 0.95, top_k 20`; thinking on `temperature 0.8, top_p 0.95, top_k 20,
presence_penalty 1.0` (the image defaults).
