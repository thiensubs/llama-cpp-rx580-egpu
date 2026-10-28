# llama.cpp on an Intel Mac with an RX 580 eGPU

Findings and a patch from one machine, published because the failure modes are
specific and reproducible: an AMD card is never selected by Apple-family
checks, ends up reading its weights over Thunderbolt, and produces silent
garbage for one architecture before any of that.

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
| model | LFM3-1.7B Q5_K_M (mixed Q5_K / Q6_K / F32) |

## Headline

Same model, same machine, one binary:

| config | generation | CPU per token | cores busy |
|---|---|---|---|
| CPU, 4 threads | 11.8 t/s | 0.346 s | 374% |
| GPU, whole graph | 25 t/s | 0.008 s | 20% |

Prompt processing 36 t/s. Energy: 0.27 Wh per 1000 tokens.

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

## Recipes

Text model, whole graph on the card:

```bash
GGML_METAL_DEVICE_NAME="Radeon RX 580" llama-server \
  -m model.gguf -ngl 999 -lm none -t 1 -c 65536 --metrics
```

`-t 1` because with the graph on the GPU the CPU threads only spin
(0.008 vs 0.067 CPU-s/token). Context sizing, measured:

| context | KV | buffers | generation |
|---|---|---|---|
| 32k | fp16 | 2854 MB | 24.45 t/s |
| 64k | fp16 | 4430 MB | 24.46 t/s |
| 64k | q8_0 | 1678 MB | 16.49 t/s |
| 128k | fp16 | — | fails to allocate |
| 128k | q8_0 | 2106 MB | 17.28 t/s |

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
patches/rx580-metal-width64.patch   full local patch (width port + the four fixes)
patches/upstream-metal-fixes.patch  the two self-contained fixes (concurrency, SSM_CONV)
repro/simdprobe.m          simd width + primitive correctness per device
repro/bwprobe.m            VRAM streaming bandwidth per device
agents/                    launchers and clients used to drive the server
                           (they read $HOME/models/*.gguf unless MODEL/MMPROJ are set)
```

Licence: same as llama.cpp (MIT) for the patch; the notes are free to reuse.
