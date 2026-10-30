# Finding 9 — `GATED_DELTA_NET` ran on the CPU, because 32 threads are hardcoded

Hybrid models with linear-attention layers (Ling, and anything else using
gated delta net) were split into 50 sub-graphs per token on this backend. The
GPU work in each piece was 0.03-0.64 ms while the synchronisation around it
cost 0.5-1.6 ms, so more than half of every token went into handing work back
and forth.

## How the culprit was found

Nothing in the log named the op. `GGML_SCHED_DEBUG=2` produced no output in
this build, so a five-line guarded logger was added at the point where the
scheduler creates a split boundary (`ggml-backend.cpp`, `ggml_backend_sched_split_graph`):

```c
if (node_backend_id != cur_backend_id || need_new_split) {
    if (getenv("GGML_SPLIT_LOG")) {
        fprintf(stderr, "SPLIT %4d %-20s -> %-10s (was %-10s) new_split=%d\n", ...);
    }
```

which printed, once per layer:

```
SPLIT  77 GATED_DELTA_NET      -> CPU        (was MTL0      )
SPLIT  80 CPY                  -> MTL0       (was CPU       )
SPLIT 173 GATED_DELTA_NET      -> CPU        (was MTL0      )
SPLIT 176 CPY                  -> MTL0       (was CPU       )
```

Any reader debugging a split problem can reproduce this in ten minutes; it is
worth carrying as a permanent diagnostic.

## The bug

Two hardcoded 32s, one on each side:

```c
// ggml-metal-device.cpp, pipeline getter
const int nsg = op->src[2]->ne[0]/32;          // elements per thread
GGML_ASSERT(ne20 % 32 == 0);

// ggml-metal-ops.cpp, dispatch
dispatch_threadgroups(..., /* threads per threadgroup */ 32, nsg, 1);
```

The kernel maps a state row as `is = tx*NSG + j` and reduces with `simd_sum`.
On a 32-lane device one simdgroup is exactly the x-dimension, so the reduction
covers one output row. On a 64-lane device the threadgroup only launches 32
threads across x, so **one simdgroup spans two y rows** and `simd_sum` adds
state from two different output rows. The kernel is not wrong about lanes; it
was never given a full simdgroup.

Opening the capability gate alone, without fixing this, produced **18.7 t/s of
garbage**:

```
prompt: What is 17 times 23?
output: "#number 17 / # 17+ / # Scalling 17545 1751754"
```

## The fix

Four lines, none of them in the kernel body:

```c
const int nw  = ...->simd_width;
const int nsg = op->src[2]->ne[0]/nw;   // was /32
GGML_ASSERT(ne20 % nw == 0);            // was % 32

dispatch_threadgroups(..., nw, nsg, 1); // was 32
```

And the support gate, which is the one that actually decides whether the op
runs on the GPU at all:

```c
case GGML_OP_GATED_DELTA_NET:
-   return has_simdgroup_reduction && op->src[2]->ne[0] % 32 == 0;
+   return op->src[2]->ne[0] % dev->props.simd_width == 0;
```

`has_simdgroup_reduction` is false on every AMD and Intel GPU, so with that
term in place the op is rejected before the kernel is ever considered and the
three lines above change nothing. Measured on the RX 580, Ling-3.0-tiny Q6_K,
`llama-cli -n 160`: 2.4 t/s with only the three lines fixed, 19.9 t/s once the
gate was corrected as well.

`NSG` is a template parameter and `is = tx*NSG + j` already derives from the
thread count, so the kernel needed no edit at all. At width 32 the generated
dispatch is identical to before.

## Result

Ling-3.0-tiny Q6_K (7.9B, hybrid GDN + attention, 128 experts) on the RX 580:

| | before | after |
|---|---|---|
| graph splits per token | 50 | **0** |
| prompt processing | 30.3 t/s | **49.2 t/s** |
| generation | 7.8 t/s | **20.8-21.2 t/s** |
| greedy output | — | **byte-identical to the CPU path** |

The last row is the correctness test. There is no `test-backend-ops` case for
`GATED_DELTA_NET` (`0/0` runnable), so the check used here is: run the same
greedy prompt twice, once with the op forced to the CPU and once with it on the
GPU, and compare the generated text. Both produce `391` and the same
explanation, character for character.

## Caveat

Verified on one card and one model, but over **three independent prompts with a
proper control arm**: two binaries with isolated library directories, one where
`GATED_DELTA_NET` is forced to the CPU and one where it runs on the GPU.

| prompt | CPU arm | GPU arm | output |
|---|---|---|---|
| 17 x 23 | 7.9 t/s | 20.7 t/s | identical |
| Vietnamese tones | 8.1 t/s | 21.4 t/s | identical |
| first six primes | 7.7 t/s | 20.8 t/s | identical |

A note for anyone repeating this: `llama-server` is a thin shim and the
behaviour lives in `libggml-metal.dylib`. Swapping only the executable between
builds silently reuses whichever library is on the library path - the first
attempt at this A/B compared two identical CPU builds and reported a clean
"identical, no speedup" before the mistake was spotted. Copy the whole
`bin` directory per arm and set `DYLD_LIBRARY_PATH` explicitly.

The remaining risk is shape coverage: these runs exercise the head sizes in
Ling-3.0-tiny only. A different `S_v` produces a different `nsg`, which is the
value the port changed, so a model with other head sizes should repeat the
greedy A/B before trusting it.
