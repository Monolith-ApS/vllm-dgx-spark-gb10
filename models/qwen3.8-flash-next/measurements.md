# Qwen3.8 Flash Next: measurements

All numbers are from one NVIDIA DGX Spark (GB10, 128 GB, DGX OS) running the image built from this repository. Thinking is off
unless stated, and MTP 7 is used (the model drafts 7 tokens ahead and the server checks them in one step; this makes decoding
faster). One profile at a time, nothing else running.

Decode speed, capacity, tool calling and control prompts were measured on 2026-10-22 and 2026-10-23 on DGX OS with the 7.0 kernel
(7.0.0-1019-nvidia), NVIDIA driver 580.178.04 and the 2026-10 platform firmware (EC 0x03000d07, SoC 0x02009e0f). Each profile ran
with the vLLM build, patches and command line of the image; the requests went through an OpenAI-compatible proxy on the local
network.

## Words used on this page

| Word | Meaning |
|---|---|
| tok/s | tokens per second; a token is roughly three quarters of a word |
| One request / all at once | one conversation alone, or as many conversations as the profile has slots, all generating at the same time |
| KV cache | the memory where the server keeps what each conversation has said so far. A profile only works if it holds the full context window of every slot |
| Preemption | the server throws a conversation's cache away to make room and has to recompute it later, which is slow. Passing means 0 |
| Time to first token | how long a request waits before the first word comes back |
| MemAvailable | memory the machine still has free (`grep MemAvailable /proc/meminfo`). On GB10, running out can freeze the whole machine |

## Decode speed

Code-writing test: "edit" returns a 250-line C file with small changes (3,000 completion tokens), "new" writes a new C module
(~1,500-1,800 tokens); greedy decoding.

"All at once" is the total speed of all conversations together, as many identical requests in parallel as the profile's
concurrency (the number in brackets). Each conversation gets roughly the total divided by that number: `128k-c8` gives about 29
tok/s to each of its 8 conversations.

Each run measures "edit" and "new" once, so a profile with 4 runs has 8 values per column; the table shows the lowest, the mean
and the highest of them. This is what to expect from a profile, run-to-run variation included (single runs of the same test
vary by a few percent, up to ~8 % for "new"). The runs were taken on 2026-10-22 and 2026-10-23, in two or three separate boots
of each profile.

| Profile | Runs | 1 request, tok/s (min / mean / max) | All at once, total tok/s (min / mean / max) |
|---|---|---|---|
| `128k-c8` | 6 | 75.2 / 80.0 / 83.6 | 220.7 / 234.3 / 256.1 (8) |
| `256k-c5` | 5 | 72.8 / 79.0 / 81.0 | 174.8 / 182.6 / 200.2 (5) |
| `384k-c4` | 6 | 77.4 / 80.0 / 82.8 | 161.1 / 170.2 / 178.4 (4) |
| `480k-c3` | 4 | 78.8 / 79.9 / 82.5 | 138.7 / 144.8 / 151.2 (3) |
| `512k-c3` | 4 | 78.6 / 80.3 / 83.5 | 127.7 / 141.5 / 151.8 (3) |
| `640k-c2` | 4 | 72.9 / 78.0 / 82.1 | 105.4 / 113.7 / 118.2 (2) |
| `768k-c2` | 4 | 76.0 / 79.7 / 84.4 | 114.1 / 119.0 / 126.1 (2) |
| `1024k-c1` | 6 | 72.4 / 77.9 / 80.7 | - |

The test sends one warm-up request first and then identical prompts, so the prefix cache is warm. A client that always sends new
prompts has to compute them from scratch, so its first token takes longer; decode speed was not measured that way.

## Capacity

`tools/fillc.py`: every conversation grows turn by turn, round-robin, towards context - 6,000 tokens; then all of them send one
more turn at the same time. The test passes when there are 0 preemptions and the slowest time to first token of that last turn
is 30 s or less.

**Does it fit?** The KV cache must hold every slot at full length ("needed" = conversations x (context - 6,000)). "Headroom" is
how much bigger the KV cache is than that.

| Profile | KV cache size | Needed | Headroom | Conversations filled to | Preemptions | Time to first token, last turn |
|---|---|---|---|---|---|---|
| `128k-c8` | 1,016,233 tokens | 1,000,576 | +1.6 % | 103-115K x 8 | 0 | 4.4-21.6 s |
| `256k-c5` | 1,351,397 | 1,280,720 | +5.5 % | 237-245K x 5 | 0 | 5.3-15.3 s |
| `384k-c4` | 1,570,327 | 1,548,864 | +1.4 % | 369-375K x 4 | 0 | 4.9-12.2 s |
| `480k-c3` | 1,586,754 | 1,456,560 | +8.9 % | 464-474K x 3 | 0 | 5.5-8.8 s |
| `512k-c3` | 1,597,186 | 1,554,864 | +2.7 % | 498-504K x 3 | 0 | 7.0-11.3 s |
| `640k-c2` | 1,676,371 | 1,298,720 | +29.1 % | 636-639K x 2 | 0 | 5.4-6.9 s |
| `768k-c2` | 1,714,537 | 1,560,864 | +9.8 % | 760-769K x 2 | 0 | 8.1-8.7 s |
| `1024k-c1` | 1,683,713 | 1,042,576 | +61.5 % | 1,028K x 1 | 0 | 5.7 s |

**Memory and start time.**

| Profile | MemAvailable after start | MemAvailable after the fill test | Ready after |
|---|---|---|---|
| `128k-c8` | 5.6 GiB | 4.9 GiB | 130 s |
| `256k-c5` | 5.8 GiB | 5.0 GiB | 127 s |
| `384k-c4` | 4.7 GiB | 3.9 GiB | 127 s |
| `480k-c3` | 5.9 GiB | 4.9 GiB | 130 s |
| `512k-c3` | 6.0 GiB | 5.0 GiB | 124 s |
| `640k-c2` | not read | 4.7 GiB | 127 s |
| `768k-c2` | 5.9 GiB | 4.7 GiB | 124 s |
| `1024k-c1` | 7.7 GiB | 6.1 GiB | 127 s |

Ready after = container start to API ready, with the weights in the page cache and the compile cache warm (an existing, stopped
container started again; API readiness polled every 3 s). Over 26 boots of these profiles on 2026-10-22/23 the
value was 124-136 s, median 127 s. The first measurement on 2026-10-13 counted from `docker run` with 10 s polling and gave
161-172 s, so the two are not the same method. The very first start on a new machine takes longer (3-5 minutes, see the
[installation guide](../../docs/installation.md)).

`256k-c5` runs with `gpu_memory_utilization` 0.89 (was 0.90): at 0.90 MemAvailable after the fill test fell to 3.7 GiB; at 0.89
it stays at 5.0 GiB and the KV cache is 4.3 % smaller, still 5.5 % above what five full conversations need.
`384k-c4` stays at 0.90 and has the least free memory of the profiles (4.7 GiB after start, 3.9 GiB after the fill test); lower
it to 0.89 if other processes on the machine need memory.

Mamba cache mode on `512k-c3`: `all` and `align` gave the same KV cache size, decode speed and fill result.

## Tool calling and control prompts

Every profile: tool calling (`tools/tool_check.py`, 9 cases x 2) 18/18; control prompts (`tools/answer_check.py`) 4/4.

## Thinking on vs off

Add `-think` to a profile name to have thinking on by default (see [configuration.md](configuration.md#thinking-mode)). It uses 3
MTP draft tokens instead of 7, because reasoning text accepts fewer of them.

**Decode speed with thinking on**, measured on 2026-10-23, 2 runs per profile (so 4 values per column), same test as above:

| Profile | 1 request, tok/s (min / mean / max) | All at once, total tok/s (min / mean / max) |
|---|---|---|
| `128k-c8-think` | 69 / 69 / 69 | 203 / 206 / 210 |
| `256k-c5-think` | 67 / 69 / 69 | 158 / 162 / 168 |
| `384k-c4-think` | 66 / 68 / 69 | 136 / 148 / 155 |
| `480k-c3-think` | 68 / 69 / 69 | 119 / 126 / 135 |
| `512k-c3-think` | 67 / 69 / 70 | 126 / 127 / 128 |
| `640k-c2-think` | 66 / 68 / 69 | 97 / 103 / 110 |
| `768k-c2-think` | 67 / 68 / 69 | 99 / 102 / 107 |
| `1024k-c1-think` | 68 / 68 / 69 | - |

**Does thinking pay off?** `tools/taskbench.py`: 10 single-prompt Python tasks with hidden tests, one run each, 1 request at a
time.

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

Thinking costs about 15 times the time and 9 times the tokens here and passed the same number of tasks, so leave it off unless
a task needs the reasoning. Ten tasks, one run each, is a small sample.

Sampling used: thinking off `temperature 1.0, top_p 0.95, top_k 20`; thinking on `temperature 0.8, top_p 0.95, top_k 20,
presence_penalty 1.0` (the image defaults).

## vLLM 0.31 and 0.32

This model's image is built on vLLM 0.31.0 ([pins](pins.md)). The comparison with 0.32.0 on the eight profiles (start-up time,
decode, capacity, and the two problems we found in 0.32.0) is in
[vllm-versions.md](vllm-versions.md); what we found in vLLM itself is in [docs/vllm-versions.md](../../docs/vllm-versions.md).
