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

## Energy

A local usage log of 5-minute buckets pairing tokens with the wattage
the power meter sampled in the same window, counting only energy drawn while a
request was in flight: **0.267–0.333 Wh per 1000 tokens** across four runs.
