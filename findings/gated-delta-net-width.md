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

Three lines, none of them in the kernel body:

```c
const int nw  = ...->simd_width;
const int nsg = op->src[2]->ne[0]/nw;   // was /32
GGML_ASSERT(ne20 % nw == 0);            // was % 32

dispatch_threadgroups(..., nw, nsg, 1); // was 32
```

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

Verified on one card, one model, one prompt for the exact-match test. The 2.7x
is a solid measurement; the claim "bit-identical" is one sample. Anyone
reusing this should repeat the greedy A/B on their own device before trusting
it, and `GGML_SPLIT_LOG` makes the before-state visible in one run.
