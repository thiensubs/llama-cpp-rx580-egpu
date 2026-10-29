# llama.cpp on an Intel Mac with an RX 580 eGPU

Findings and a patch from one machine, published because the failure modes are
specific and reproducible: an AMD card is never selected by Apple-family
checks, ends up reading its weights over Thunderbolt, and produces silent
garbage for one architecture before any of that.

## ⚡ TL;DR — the smaller model is slower

| weights | file | generation | card utilisation |
|---|---|---|---|
| **f16** | 3.22 GiB | **36.8 t/s** 🏆 | 125 GB/s |
| Q5_K_M | 1.16 GiB | 28.2 t/s | 35.5 GB/s |
| Q4_0 | 0.98 GiB | 4.3 t/s 🐌 | CPU fallback |
| Q8_0 | 1.71 GiB | 3.4 t/s 🐌 | CPU fallback |

The card streams at 217 GB/s. Q5_K needs 5.8 ms of reads and spends 29.7 ms
unpacking 5.5-bit blocks; f16 needs 16.0 ms of reads and 11.1 ms on everything
else. Compression trades bandwidth for integer work, and this GPU sits on the
wrong side of that trade — so it is also the *less accurate* choice, not just
the slower one. Details in
[Finding 7](#finding-7--quantisation-is-slower-than-f16-on-this-card).

**If you run any MoE model, read [Finding 8](#finding-8--no-moe-model-runs-on-the-gpu-on-a-non-apple-device) first.**
On every non-Apple GPU its expert matmuls run on the CPU while the weights sit
in VRAM. One gate causes it; fixing it took a 7.9B MoE from *no request
finishing* to 7.8 t/s.

**Status: research notes, not an upstream-ready change.** The patch contains
temporary instrumentation and disables flash attention on this device. Read the
"not done" section before using anything here.

## The machine

| part | detail |
|---|---|
| host | MacBook Pro, Intel, macOS |
| GPU 0 | AMD Radeon RX 580, 8 GB, Thunderbolt enclosure (`MTLDeviceLocationExternal`) |
| GPU 1 | AMD Radeon Pro 450, 2 GB, internal |
| GPU 2 | Intel HD 530, integrated |
| llama.cpp | `a7af149` |
| model | LFM3-1.7B, f16 and Q5_K_M (Finding 7) |

## Headline

Same model, same machine, one binary:

| config | generation | CPU per token | cores busy |
|---|---|---|---|
| CPU, 4 threads, Q5_K | 11.8 t/s | 0.346 s | 374% |
| GPU, whole graph, Q5_K | 28.2 t/s | 0.008 s | 20% |
| GPU, whole graph, f16 | 36.8 t/s | 0.008 s | 20% |

Prompt processing 49 t/s on Q5_K and 77 t/s on f16. Energy: 0.27 Wh per 1000
tokens.

## Finding 1 — an external GPU is forced onto shared (host) buffers

`ggml-metal-device.m`:

```objc
dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX && TARGET_CPU_X86_64
    // In case of eGPU, shared memory may be preferable.
    dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
```

"Shared" means zero-copy host memory: the 1.26 GB of weights never enter VRAM
and the card reads them across PCIe on every token.

Same kernel, both ways (`test-backend-ops perf`, q5_K, m=4096 k=14336):

| buffers | per run | throughput |
|---|---|---|
| shared (default for eGPU) | 41 919 µs | 2.80 GFLOPS |
| private VRAM | 1 032 µs | 113.74 GFLOPS |

Decode with the graph on the GPU: **1203 ms/token shared vs 51.5 ms private.**

Fix: stop forcing it. `GGML_METAL_SHARED_BUFFERS_ENABLE` still lets users opt in.

**The flip alone is not enough, and I found that the hard way.** The Metal device
also advertised `caps.buffer_from_host_ptr = true`, so the model loader kept
mapping the file into host memory and then asserted
`ggml_metal_buffer_is_shared` (`ggml-metal.cpp:42`) — every tool that uses mmap,
including `llama-bench` and a default `llama-cli`, aborted at load. The
capability has to follow the same flag:

```c
/* .buffer_from_host_ptr = */ ggml_metal_device_get_props(ctx_dev)->use_shared_buffers,
```

With private buffers, mapping a file is not possible, so the loader copies
instead — which is exactly what `-lm none` was forcing by hand.

## Finding 2 — simdgroup width 32 is hardcoded; AMD is 64

```c
#define N_SIMDWIDTH 32 // assuming SIMD group size is 32
```

and the capability gate that guards ~20 ops:

```objc
dev->props.has_simdgroup_reduction = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
```

An AMD Polaris card satisfies neither, so the backend concludes the hardware
cannot do simdgroup reductions. It can — see `repro/simdprobe.m`:

```
[0] AMD Radeon RX 580   simdgroup width=64  sum/max/shuffle/broadcast OK
[1] AMD Radeon Pro 450  simdgroup width=64  OK
[2] Intel HD Graphics   simdgroup width=8   OK
```

The real blockers are two: the width is a compile-time constant, and the matvec
kernels hardcode a 32-lane lane->row mapping (`tiisg/4`, `tiisg%4`, `i += 4`).

Fix (in `patches/rx580-metal-width64.patch`):
probe the width at device init, inject it as `N_SIMDWIDTH` through
`MTLCompileOptions.preprocessorMacros` (the Metal library is compiled per device
at runtime), and make the kernels width-parametric:

| kernel | change |
|---|---|
| `mul_mv` q5_K | `NIX = NW/8`, `tid = tiisg/NIX`, `ix = tiisg%NIX`, stride and `y1` advance follow `NIX` |
| `mul_mv` q6_K | `NIX = NW/16`, same shape |
| `mul_mv_t_t`, `_4` | `NLB = NB/NF` lanes per block, `NBL = NW/NLB` blocks per simdgroup |
| `norm`, `softmax` | threadgroup starts at the device width; `smem = width * 4` |

At width 32 the generated code is identical to before, so Apple GPUs are
untouched — verified by construction, not by testing on Apple hardware.

`test-backend-ops -b MTL0`: q5_K 87/87 OK, q6_K OK, f32 280 OK, f16 291 OK,
RMS_NORM OK, SOFT_MAX 216 OK.

## Finding 3 — `use_concurrency` races on non-Apple GPUs

Symptom: one architecture produced `GGGGGGGGGGGG` on Metal and correct text on
CPU. The bug was not in the model or the kernels.

Cause: up to 8 command buffers submitted from different threads into one
`MTLCommandQueue`. Apple GPUs preserve enqueue order; AMD does not, so the graph
raced. `GGML_METAL_CONCURRENCY_DISABLE=1` made the same binary produce correct
output, which isolated it.

Fix: force `use_concurrency = false` when the device name does not start with
`"Apple "`.

## Finding 4 — `SSM_CONV` refused unnecessarily

`case GGML_OP_SSM_CONV: return has_simdgroup_reduction;` — but `kernels/ssm.metal`
contains no simdgroup calls; the kernel is scalar. 62/62 `test-backend-ops`
cases went from "not supported" to OK once the gate required only F32 +
contiguous instead.

## Finding 5 — mmap keeps the weights out of VRAM even when buffers are private

`ggml_backend_dev_buffer_from_host_ptr` -> `ggml_metal_buffer_map()` ->
`newBufferWithBytesNoCopy(...Shared)`. The model loader walks that path when the
file is memory-mapped, so weights stay host-resident regardless of the buffer
type. Log line when it happens: `METAL bigbuf(nocopy) size=1250 MB`.

`-lm none` forces the ordinary upload path instead: `METAL bigbuf size=1250 MB
is_shared=0`, and the GPU stops reading weights across the bus.

Do **not** "fix" `ggml_metal_buffer_map()` to allocate private buffers: the
loader calls `ggml_backend_buffer_get_base` on mapped buffers and asserts
`is_shared` (`ggml-metal.cpp:42`). Verified abort.

## Finding 6 — the q5_K threadgroup shape is tuned for another GPU

`N_R0_Q5_K` (rows handled per simdgroup) and `N_SG_Q5_K` (simdgroups per
threadgroup) default to 1 and 2. Swept on the RX 580 with `llama-bench tg64`
(`-ngl 999 -t 1`), each build remeasured:

| rows/simdgroup | simdgroups | threads/tg | tg64 |
|---|---|---|---|
| 1 (default) | 2 | 128 | 25.46 ± 0.07 |
| 1 | 4 | 256 | 24.70 ± 0.42 |
| 2 | 2 | 128 | 28.12 ± 0.33 |
| 2 | 1 | 64 | 28.19 ± 0.31 |
| 2 | 4 | 256 | 27.75 ± 0.34 |
| 2 | 8 | 512 | 25.82 ± 0.02 |

One constant changes, `N_R0_Q5_K 1 -> 2`: **+10%** (25.46 → 28.12). Through the
server, with sampling, 24.45 → 26.7 t/s. Output text unchanged at temperature 0.

## Finding 7 — quantisation is slower than f16 on this card

Every quantised file is smaller than the f16 one, and every one of them
generates fewer tokens per second. `llama-bench tg64`, `-ngl 999 -lm none -t 1
-ub 256`, three repeats, one model:

| weights | file | tg64 | pp32 | GB/s moved |
|---|---|---|---|---|
| f16 | 3.22 GiB | 36.07 ± 1.92 | 76.89 ± 0.24 | 125 |
| Q5_K_M | 1.16 GiB | 28.21 ± 0.24 | 49.02 ± 0.06 | 35.5 |
| Q4_0 | 0.98 GiB | 4.30 ± 0.39 | 14.87 ± 0.51 | CPU |
| Q8_0 | 1.71 GiB | 3.37 ± 0.05 | 13.76 ± 2.33 | CPU |

The 217 GB/s that `repro/bwprobe.m` measures says reading them should cost
16.0 ms (f16) and 5.8 ms (Q5_K). Measured per token: 27.2 ms and 35.5 ms. Q5_K
saves 10.2 ms of traffic and spends about 19 ms rebuilding 5.5-bit blocks —
shifts and masks, on a GPU with weak integer throughput. The isolated kernel
agrees: q5_K `MUL_MAT` at m=4096, k=14336 is 892 µs for 40.4 MB of weights,
45 GB/s and 131 GFLOPS, 21% of the same card's streaming rate.

Live through `llama-server` with sampling on: 26.7 → 35.1 t/s, and 34.9 t/s with
the vision server sharing the card (5484 MB + 2008 MB of 8590 MB).

Q4_0 and Q8_0 are not slow kernels, they are missing ones. `test-backend-ops
perf` reports their `MUL_MAT` as "not supported" here, so llama.cpp runs those
matmuls on the CPU. Admitting them to the width-64 whitelist
(`ggml_metal_supports_mul_mat_wide`, `ggml-metal-device.m`) makes them run and
fail correctness instead: Q4_0 0/121 cases passed, Q8_0 5 failures. That
whitelist admits F32, F16, Q5_K and Q6_K only, and the narrow list is earned
rather than cautious.

On an Apple GPU the same quantised weights win, which is why the defaults look
the way they do. This is a property of the card, not of the format.

## Finding 8 — no MoE model runs on the GPU on a non-Apple device

`MUL_MAT_ID` — the op that runs routed expert matmuls — is refused whenever
`has_simdgroup_reduction` is false, which is **every AMD and Intel GPU**:

```objc
case GGML_OP_MUL_MAT_ID:
    if (!has_simdgroup_reduction) {
        return false;
    }
```

The weights still get placed in VRAM, so the model looks loaded and the
placement table looks correct. Only the compute goes to the CPU, which makes a
mis-dispatch indistinguishable from a slow model. Ling-3.0-tiny Q6_K (7.9B,
128 experts, 8 used) measured on the RX 580:

| | before | after |
|---|---|---|
| prompt processing | 2.97 t/s | **27.2 t/s** |
| generation | did not finish a request | **7.8 t/s** |
| `MTL0_Private` buffer | 5996 MiB | 5996 MiB |

The kernel already existed and was already built on the validated matvec
templates. Pointing the gate at the same width-64 whitelist `MUL_MAT` uses
gives `224/224` on `test-backend-ops -o MUL_MAT_ID`. Full write-up and the
self-check commands: [findings/moe-on-non-apple-gpus.md](findings/moe-on-non-apple-gpus.md).

## Recipes

Text model, whole graph on the card:

```bash
GGML_METAL_DEVICE_NAME="Radeon RX 580" llama-server \
  -m model.gguf -ngl 999 -lm none -t 1 -c 65536 -ub 256 --metrics
```

Keep the weights at f16 here (Finding 7). Q5_K is the fallback when VRAM is
short, not a speed win.

`-t 1` because with the graph on the GPU the CPU threads only spin
(0.008 vs 0.067 CPU-s/token). Context sizing, measured:

| context | KV | buffers | generation |
|---|---|---|---|
| 32k | fp16 | 2854 MB | 24.45 t/s |
| 64k | fp16 | 4430 MB | 24.46 t/s |
| 64k | q8_0 | 1678 MB | 16.49 t/s |
| 128k | fp16, `-ub 512` | — | **fails to allocate** (7156 MB) |
| 128k | fp16, `-ub 256` | 5290 MB | 26.7 t/s |
| 128k | q8_0, `-ub 512` | 2106 MB | 17.28 t/s |

The second allocated buffer is not the KV cache — it is the compute graph, and it
scales with **ubatch** as well as context. At `-c 65536`: `-ub 512` -> 2375 MB,
`-ub 256` -> 1221 MB, `-ub 128` -> 644 MB, with the KV constant at 805 MB. This is
why 128k looked impossible at full precision until ubatch was cut: the KV at 128k
is only 1.61 GB (6 global layers x 4 kv-heads x 128 dims x fp16, matching Liquid's
published 1.6 GB), while the compute buffer at `-ub 512` was 4.3 GB.

The same ubatch matters for latency: `llama-server` re-evaluates `ubatch + 4`
prompt tokens on every request, so a 4k prompt costs 516 tokens of rework at
`-ub 512` (16.9 s) and 260 at `-ub 256` (8.2 s). Below 256 the prompt cache stops
working altogether (4001 tokens re-processed per request).

Vision (LFM3-VL-2B + projector): add `--image-max-tokens 256`. Without it the
encoder sizes its compute buffer from the image — a 1680x2866 chart asked for
16.9 GB and died.

## Reproduce

```bash
# 1. does this card do 64-lane reductions correctly?
clang -fobjc-arc -framework Foundation -framework Metal repro/simdprobe.m -o simdprobe && ./simdprobe

# 2. what is VRAM read bandwidth?
clang -fobjc-arc -framework Foundation -framework Metal repro/bwprobe.m -o bwprobe && ./bwprobe 1 0

# 3. correctness of the port
GGML_METAL_DEVICE_INDEX=0 ./build/bin/test-backend-ops -b MTL0 -o MUL_MAT \
  -p 'type_a=q5_K,type_b=f32'
```

Measured here: RX 580 217 GB/s, Pro 450 74 GB/s.

## What is not done

- **Flash attention is disabled at width 64.** Its vec kernels carry 1140
  template instantiations sized for 32 lanes; they are fenced behind
  `#if N_SIMDWIDTH == 32`. That is a correctness-preserving cut, not a port.
- **Temporary instrumentation is still in the patch**: `GGML_SPLIT_DEBUG`
  (graph split dump), `GGML_METAL_GPU_TIME` (per-graph GPU time), and the
  `METAL bigbuf` prints. Strip them before proposing anything.
- **One machine, one model, one llama.cpp commit.** No p95 numbers, no
  sustained-load runs.
- **Nothing here was submitted upstream.** llama.cpp's contribution rules
  prohibit AI-written PR descriptions and automated submissions; any PR must be
  written and defended by a human who owns the change.

## Layout

```
findings/measurements.md   the numbers behind every claim above
findings/moe-on-non-apple-gpus.md   the MUL_MAT_ID gate - read this if you run
                           any MoE model on AMD or Intel
patches/rx580-metal-width64.patch   full local patch (width port + the four fixes)
patches/upstream-metal-fixes.patch  the two self-contained fixes (concurrency, SSM_CONV)
repro/simdprobe.m          simd width + primitive correctness per device
repro/bwprobe.m            VRAM streaming bandwidth per device
agents/                    launchers and clients used to drive the server
                           (they read $HOME/models/*.gguf unless MODEL/MMPROJ are set)
```

Licence: same as llama.cpp (MIT) for the patch; the notes are free to reuse.
