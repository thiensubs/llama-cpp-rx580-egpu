# Measurements

Every number in the README, with how it was taken. Machine and model as in the
README. Nothing here is pooled or averaged unless stated.

## Decode cost, CPU vs GPU

Method: `llama-cli` with `-n 1` and `-n 65`, identical prompt, `/usr/bin/time -p`
around each. The difference isolates 64 generated tokens from model load and
prompt processing. Three prompts (5, 27, 121 words) per configuration.

| config | CPU-s per token | cores busy | generation |
|---|---|---|---|
| CPU `-ngl 0 -t 4` | 0.326 / 0.368 / 0.345 | 384 / 382 / 355 % | 11.8 / 10.4 / 10.3 t/s |
| GPU `-ngl 999 -t 1` | — | 20 % | 25 t/s |

CPU-side numbers come from the process, not the wall clock, so other load on the
machine cannot flatter them.

## Shared vs private buffers (Finding 1)

`test-backend-ops perf -b MTL0 -o MUL_MAT -p 'type_a=q5_K,type_b=f32,m=4096,n=1'`

| buffer type | per run | GFLOPS |
|---|---|---|
| shared (eGPU default) | 41 919.17 µs | 2.80 |
| private (GGML_METAL_SHARED_BUFFERS_ENABLE unset after patch) | 1 031.79 µs | 113.74 |

Per-graph GPU time, instrumented with `GGML_METAL_GPU_TIME`
(`cb.GPUEndTime - cb.GPUStartTime`, one command buffer per graph):

| weights host-resident (mmap) | weights in VRAM |
|---|---|
| 1203 ms/token | 51.5 ms/token |

## simdgroup width (Finding 2)

`repro/simdprobe.m`, `[[threads_per_simdgroup]]` plus correctness of
`simd_sum`, `simd_max`, `simd_shuffle`, `simd_broadcast`,
`simd_prefix_inclusive_sum`:

```
[0] AMD Radeon RX 580   width=64  sum(0..63)=2016 correct
[1] AMD Radeon Pro 450  width=64  correct
[2] Intel HD Graphics   width=8   correct
```

## VRAM bandwidth (repro/bwprobe.m)

Streaming read of a 1 GB private buffer, sum-reduce kernel, GPU timestamps:

| device | GB/s |
|---|---|
| RX 580 | 217 (spec 256) |
| Pro 450 | 74 |

## Context and KV cache

`METAL bigbuf` allocation lines at startup, then one 100-token completion at
temperature 0 for the rate.

| `-c` | KV | allocated | generation |
|---|---|---|---|
| 4096 | fp16 | 1250 + 173 MB | — |
| 8192 | fp16 | 1250 + 320 MB | — |
| 32768 | fp16 | 1250 + 403 + 1201 MB | 24.45 t/s |
| 65536 | fp16 | 1250 + 805 + 2375 MB | 24.46 t/s |
| 65536 | q8_0 | 1250 + 428 MB | 16.49 t/s |
| 131072 | fp16 | 1250 + 1611 + 429 + 4295 MB, then failure | — |
| 131072 | q8_0 | 1250 + 856 MB | 17.28 t/s |

Output text was byte-identical across all rows that completed (same prompt,
temperature 0). KV is small for this model because only 11 of 24 layers hold
attention state: 13 are short convolutions, 5 use a 256-token sliding window.

## Prompt processing

Cold ~4k-token prompt, `cache_prompt: false`:

| `-ub` | wall |
|---|---|
| 512 (default) | 116.1 s |
| 1024 | 132.1 s |

## Prefill cache behaviour

Server warm, same prompt repeated via the OpenAI-compatible endpoint:

| situation | new prompt tokens | served from cache | wall |
|---|---|---|---|
| back-to-back | 516 | 3415 | 23.5 s |
| after 300 s idle | 516 | 3415 | 23.5 s |
| first call after server start | 3931 | 0 | 116.6 s |

Conclusion: the prompt cache is durable across idle; the slow case is the first
call after a (re)start, not idleness. Weights load once per start — one
`METAL bigbuf` line served 526 generations with flat RSS.

## Threadgroup shape for q5_K (Finding 6)

`llama-bench tg64`, `-ngl 999 -t 1`, three repetitions, one rebuild per row
(`N_R0_Q5_K` / `N_SG_Q5_K` in `ggml-metal-impl.h`):

| rows/simdgroup | simdgroups | tg64 |
|---|---|---|
| 1 (default) | 2 | 25.46 ± 0.07 |
| 1 | 4 | 24.70 ± 0.42 |
| 1 | 8 | — (build ok, not run) |
| 2 | 1 | 28.19 ± 0.31 |
| 2 | 2 | 28.12 ± 0.33 |
| 2 | 4 | 27.75 ± 0.34 |
| 2 | 8 | 25.82 ± 0.02 |

End to end through `llama-server` with sampling on, same prompt at temperature 0:
24.45 t/s before, 26.6-26.8 t/s after.

## Energy

A local usage log of 5-minute buckets pairing tokens with the wattage
the power meter sampled in the same window, counting only energy drawn while a
request was in flight: **0.267–0.333 Wh per 1000 tokens** across four runs.

## Quantisation vs speed (Finding 7)

`llama-bench`, `-ngl 999 -lm none -t 1 -p 32 -n 64 -r 3 -ub 256`, same 1.73B
model, one binary, weights converted from the same f16 file:

| weights | file | tg64 | pp32 | bytes/weight |
|---|---|---|---|---|
| f16 | 3.22 GiB | 36.07 ± 1.92 | 76.89 ± 0.24 | 16.0 |
| Q5_K_M | 1.16 GiB | 28.21 ± 0.24 | 49.02 ± 0.06 | 5.8 |
| Q4_0 | 0.98 GiB | 4.30 ± 0.39 | 14.87 ± 0.51 | 4.9 |
| Q8_0 | 1.71 GiB | 3.37 ± 0.05 | 13.76 ± 2.33 | 8.5 |

The Q4_0 and Q8_0 rows are not kernel measurements. On this backend their
MUL_MAT is reported as "not supported" (`test-backend-ops perf -o MUL_MAT
-p 'type_a=q8_0' -b MTL0`), so llama.cpp runs those matmuls on the CPU. Adding
both types to the width-64 whitelist makes them run on the GPU and fail
correctness instead (`test-backend-ops test -o MUL_MAT -p 'type_a=...' -b MTL0`):

| type | result |
|---|---|
| q4_0 | 0/121 cases passed |
| q8_0 | 5 cases failed |

Bandwidth actually moved per token, from the same runs (file size / time):

| weights | GB/s |
|---|---|
| f16 | 125 |
| Q5_K | 35.5 |

Roofline from `repro/bwprobe.m` on the same card, 217 GB/s:

| weights | bytes | read time | measured | remainder | t/s |
|---|---|---|---|---|---|
| f16 | 3.46 GB | 16.0 ms | 27.2 ms | 11.1 ms | 36.8 |
| Q5_K | 1.26 GB | 5.8 ms | 35.5 ms | 29.7 ms | 28.2 |

Q5_K saves 10.2 ms of traffic and spends about 19 ms rebuilding the numbers.

Isolated kernel, `test-backend-ops perf -o MUL_MAT -p 'q5_K'` with
`m=4096, n=1, k=14336`: 892 µs per run, 40.4 MB of weights, 45 GB/s,
131 GFLOPS. That is 21% of the 217 GB/s the same card reaches when it streams
plain floats, so the loss is in the unpack, not in the memory system and not in
dispatch.

Live through `llama-server` with sampling on, 128 tokens: 26.7 t/s on Q5_K,
35.1 t/s on f16. With the vision server resident beside it, 34.9 t/s. Allocated
buffers on the card: f16 text 3458 + 805 + 1221 = 5484 MB, vision 379 + 1250 +
379 = 2008 MB, of 8590 MB usable.

Caveat: the Q4_0 file is a QAD conversion, not a plain one, and it allocates in
217 MB chunks where the other models use a single 1250 MB buffer. Part of its
gap may be tensor layout rather than the CPU fallback. Everything else above is
a single machine, one model, three repetitions.
