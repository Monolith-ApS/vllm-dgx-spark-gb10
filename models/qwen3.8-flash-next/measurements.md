# Qwen3.8 Flash Next: measurements

All numbers are from one NVIDIA DGX Spark (GB10, 128 GB, DGX OS) running the image built from this repository. Thinking is off
unless stated, and MTP 7 is used (the model drafts 7 tokens ahead and the server checks them in one step; this makes decoding
faster). One profile at a time, nothing else running.

Each section shows the default, vLLM 0.32.0 (measured on 2026-10-25), and then vLLM 0.31.0 (`VLLM=0.31`, measured on 2026-10-22
and 2026-10-23). Both on DGX OS with the 7.0 kernel (7.0.0-1019-nvidia), NVIDIA driver 580.178.04 and the 2026-10 platform
firmware (EC 0x03000d07, SoC 0x02009e0f). Each profile ran with the vLLM build, patches and command line of the image; the
requests went through an OpenAI-compatible proxy on the local network.

In short: 0.32.0 decodes at the same speed as 0.31.0 (97-103 %), has a 2-6 % larger KV cache and a little more free memory.
How to choose between them: [vllm-versions.md](vllm-versions.md).

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

Each run measures "edit" and "new" once, so a profile with 3 runs has 6 values per column; the tables show the lowest, the mean
and the highest of them. This is what to expect from a profile, run-to-run variation included (single runs of the same test
vary by a few percent, up to ~8 % for "new").

**vLLM 0.32.0** (default), 3 runs per profile in one boot:

| Profile | Runs | 1 request, tok/s (min / mean / max) | All at once, total tok/s (min / mean / max) |
|---|---|---|---|
| `128k-c8` | 3 | 75.7 / 79.6 / 81.8 | 218.5 / 231.6 / 253.3 (8) |
| `256k-c5` | 3 | 76.2 / 78.2 / 79.8 | 162.1 / 177.2 / 186.4 (5) |
| `384k-c4` | 1 (1) | 76.8 / 77.4 / 78.1 | 158.7 / 161.8 / 165.0 (4) |
| `480k-c3` | 3 | 76.4 / 79.7 / 83.6 | 138.2 / 140.8 / 147.1 (3) |
| `512k-c3` | 3 | 77.4 / 79.2 / 80.9 | 141.6 / 145.6 / 151.2 (3) |
| `640k-c2` | 3 | 74.6 / 78.2 / 79.9 | 113.7 / 116.5 / 120.1 (2) |
| `768k-c2` | 3 | 78.2 / 79.4 / 81.7 | 116.7 / 120.8 / 126.8 (2) |
| `1024k-c1` | 3 | 71.6 / 77.0 / 80.0 | same as 1 request (1 conversation) |

(1) `384k-c4` at its current `gpu_memory_utilization` 0.895, one run; at 0.90 the same night it gave 72.4 / 77.9 / 81.6 and
161.6 / 164.6 / 167.6 over 3 runs.

**vLLM 0.31.0**, runs taken in two or three separate boots of each profile:

| Profile | Runs | 1 request, tok/s (min / mean / max) | All at once, total tok/s (min / mean / max) |
|---|---|---|---|
| `128k-c8` | 6 | 75.2 / 80.0 / 83.6 | 220.7 / 234.3 / 256.1 (8) |
| `256k-c5` | 5 | 72.8 / 79.0 / 81.0 | 174.8 / 182.6 / 200.2 (5) |
| `384k-c4` | 6 | 77.4 / 80.0 / 82.8 | 161.1 / 170.2 / 178.4 (4) |
| `480k-c3` | 4 | 78.8 / 79.9 / 82.5 | 138.7 / 144.8 / 151.2 (3) |
| `512k-c3` | 4 | 78.6 / 80.3 / 83.5 | 127.7 / 141.5 / 151.8 (3) |
| `640k-c2` | 4 | 72.9 / 78.0 / 82.1 | 105.4 / 113.7 / 118.2 (2) |
| `768k-c2` | 4 | 76.0 / 79.7 / 84.4 | 114.1 / 119.0 / 126.1 (2) |
| `1024k-c1` | 6 | 72.4 / 77.9 / 80.7 | same as 1 request (1 conversation) |

The test sends one warm-up request first and then identical prompts, so the prefix cache is warm. A client that always sends new
prompts has to compute them from scratch, so its first token takes longer; decode speed was not measured that way.

## Capacity

`tools/fillc.py`: every conversation grows turn by turn, round-robin, towards context - 6,000 tokens; then all of them send one
more turn at the same time. The test passes when there are 0 preemptions and the slowest time to first token of that last turn
is 30 s or less.

**Does it fit?** The KV cache must hold every slot at full length ("needed" = conversations x (context - 6,000)). "Headroom" is
how much bigger the KV cache is than that.

**vLLM 0.32.0** (default):

| Profile | GPU memory utilization | KV cache size (tokens) | Needed (tokens) | Headroom | Conversations filled to | Preemptions | Time to first token, last turn |
|---|---|---|---|---|---|---|---|
| `128k-c8` | 0.89 | 1,034,958 | 1,000,576 | +3.4 % | 103-114K x 8 | 0 | 3.9-19.4 s |
| `256k-c5` | 0.89 | 1,383,035 | 1,280,720 | +8.0 % | 236-244K x 5 | 0 | 4.9-15.2 s |
| `384k-c4` | 0.895 | 1,583,011 | 1,548,864 | +2.2 % | 368-374K x 4 | 0 | 5.0-11.6 s |
| `480k-c3` | 0.89 | 1,632,166 | 1,456,560 | +12.1 % | 464-473K x 3 | 0 | 5.6-8.9 s |
| `512k-c3` | 0.89 | 1,659,344 | 1,554,864 | +6.7 % | 497-504K x 3 | 0 | 5.6-8.1 s |
| `640k-c2` | 0.89 | 1,721,374 | 1,298,720 | +32.5 % | 636-638K x 2 | 0 | 6.5-7.9 s |
| `768k-c2` | 0.89 | 1,763,689 | 1,560,864 | +13.0 % | 759-769K x 2 | 0 | 7.6-8.2 s |
| `1024k-c1` | 0.88 | 1,779,583 | 1,042,576 | +70.7 % | 1,028K x 1 | 0 | 4.9 s |

**vLLM 0.31.0**:

| Profile | GPU memory utilization | KV cache size (tokens) | Needed (tokens) | Headroom | Conversations filled to | Preemptions | Time to first token, last turn |
|---|---|---|---|---|---|---|---|
| `128k-c8` | 0.89 | 1,016,233 | 1,000,576 | +1.6 % | 103-115K x 8 | 0 | 4.4-21.6 s |
| `256k-c5` | 0.89 | 1,351,397 | 1,280,720 | +5.5 % | 237-245K x 5 | 0 | 5.3-15.3 s |
| `384k-c4` | 0.90 | 1,570,327 | 1,548,864 | +1.4 % | 369-375K x 4 | 0 | 4.9-12.2 s |
| `480k-c3` | 0.89 | 1,586,754 | 1,456,560 | +8.9 % | 464-474K x 3 | 0 | 5.5-8.8 s |
| `512k-c3` | 0.89 | 1,597,186 | 1,554,864 | +2.7 % | 498-504K x 3 | 0 | 7.0-11.3 s |
| `640k-c2` | 0.89 | 1,676,371 | 1,298,720 | +29.1 % | 636-639K x 2 | 0 | 5.4-6.9 s |
| `768k-c2` | 0.89 | 1,714,537 | 1,560,864 | +9.8 % | 760-769K x 2 | 0 | 8.1-8.7 s |
| `1024k-c1` | 0.88 | 1,683,713 | 1,042,576 | +61.5 % | 1,028K x 1 | 0 | 5.7 s |

**Memory and start time.**

**vLLM 0.32.0** (default):

| Profile | GPU memory utilization | MemAvailable after start | MemAvailable after the fill test | Ready after |
|---|---|---|---|---|
| `128k-c8` | 0.89 | 6.1 GiB | 5.4 GiB | 133 s |
| `256k-c5` | 0.89 | 6.2 GiB | 5.5 GiB | 130 s |
| `384k-c4` | 0.895 | 5.6 GiB | 4.8 GiB | 127 s |
| `480k-c3` | 0.89 | 6.2 GiB | 5.3 GiB | 127 s |
| `512k-c3` | 0.89 | 6.2 GiB | 5.3 GiB | 121 s |
| `640k-c2` | 0.89 | 6.8 GiB | 5.4 GiB | 127 s |
| `768k-c2` | 0.89 | 6.8 GiB | 5.3 GiB | 127 s |
| `1024k-c1` | 0.88 | 7.7 GiB | 6.2 GiB | 127 s |

**vLLM 0.31.0**:

| Profile | GPU memory utilization | MemAvailable after start | MemAvailable after the fill test | Ready after |
|---|---|---|---|---|
| `128k-c8` | 0.89 | 5.6 GiB | 4.9 GiB | 130 s |
| `256k-c5` | 0.89 | 5.8 GiB | 5.0 GiB | 127 s |
| `384k-c4` | 0.90 | 4.7 GiB | 3.9 GiB | 127 s |
| `480k-c3` | 0.89 | 5.9 GiB | 4.9 GiB | 130 s |
| `512k-c3` | 0.89 | 6.0 GiB | 5.0 GiB | 124 s |
| `640k-c2` | 0.89 | 5.8 GiB (1) | 4.7 GiB | 127 s |
| `768k-c2` | 0.89 | 5.9 GiB | 4.7 GiB | 124 s |
| `1024k-c1` | 0.88 | 7.7 GiB | 6.1 GiB | 127 s |

Free memory follows `gpu_memory_utilization`, not the context window: vLLM reserves that share of memory up front and fills it
with KV cache, whatever the profile. Most profiles run at 0.89 and land within about 0.5 GiB of each other (on 0.31.0 `640k-c2`
and `768k-c2` both 4.7 GiB after the fill test: 4,822 and 4,845 MiB); `384k-c4` (0.90) has less, `1024k-c1` (0.88) more.

(1) The reading after start failed in this run; the value is from a second start of the same profile and image the same day
(the other profiles differed by at most 0.2 GiB between the two starts).

Ready after = container start to API ready, with the weights in the page cache and the compile cache warm (an existing, stopped
container started again; API readiness polled every 3 s). Over 26 boots of these profiles on 2026-10-22/23 (vLLM 0.31.0) the
value was 124-136 s, median 127 s. The first measurement on 2026-10-13 counted from `docker run` with 10 s polling and gave
161-172 s, so the two are not the same method. The very first start on a new machine takes longer (3-5 minutes, see the
[installation guide](../../docs/installation.md)).

`256k-c5` runs with `gpu_memory_utilization` 0.89 (was 0.90): on 0.31.0 at 0.90 MemAvailable after the fill test fell to
3.7 GiB; at 0.89 it stays at 5.0 GiB and the KV cache is 4.3 % smaller, still 5.5 % above what five full conversations need.
`384k-c4` runs with `gpu_memory_utilization` 0.895 (was 0.90): at 0.90 MemAvailable after the fill test was 4.2 GiB on 0.32.0
(3.9 GiB on 0.31.0); at 0.895 it is 4.8 GiB and the KV cache, 1,583,011 tokens, is still 2.2 % above what four full
conversations need. 0.895 was measured on 0.32.0 only; on 0.31.0, whose KV cache is smaller, the profile keeps 0.90
(`gpu_memory_utilization_by_release` in `profiles.json`).

Mamba cache mode on `512k-c3` (0.31.0): `all` and `align` gave the same KV cache size, decode speed and fill result. 0.32.0 only
has `align`.

## Tool calling and control prompts

Every profile, on both releases: tool calling (`tools/tool_check.py`, 9 cases x 2) 18/18; control prompts
(`tools/answer_check.py`) 4/4.

## Thinking on vs off

Add `-think` to a profile name to have thinking on by default (see [configuration.md](configuration.md#thinking-mode)). It uses 3
MTP draft tokens instead of 7, because reasoning text accepts fewer of them.

The numbers in this section were measured on vLLM 0.31.0. On vLLM 0.32.0 the decode speed with thinking on was the same within
+/- 3 % ([vllm-versions.md](vllm-versions.md)).

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

This model's image is built on vLLM 0.32.0 by default, or on 0.31.0 with `VLLM=0.31` ([pins](pins.md)). How to choose, start-up
times and the two problems we found in 0.32.0: [vllm-versions.md](vllm-versions.md); what we found in vLLM itself:
[docs/vllm-versions.md](../../docs/vllm-versions.md).
