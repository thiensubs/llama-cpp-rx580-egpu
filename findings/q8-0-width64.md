# Finding 10 — the Q8_0 matvec assumed 32 lanes in two places

Q8_0 is the highest-quality quantisation that llama.cpp ships, and on this
backend it was refused outright at width 64. Enabling it without fixing the
kernel gave the right speed and the **wrong text**; fixing it gives the speed
and byte-identical output.

## Symptom

`test-backend-ops test -o MUL_MAT -p 'type_a=q8_0'` reported 7 failures out of
123 shapes, including `m=6,n=4096,k=5120` — large k, so "only small shapes are
broken" was never true.

In a real model the cost was visible as graph splits: Ling-3.0-tiny carries a
few Q8_0 tensors, and each one bounced to the CPU once per layer:

```
SPLIT  398 MUL_MAT   a=q8_0  b=f32  -> CPU   (was MTL0)
SPLIT  401 ROPE      a=f32   b=i32  -> MTL0  (was CPU )
```

## Bug 1 — the per-block lane mapping

```c
const short ix = tiisg/(NW/NQ);   // NW=64 -> 8, NW=32 -> 4
const short il = tiisg%(NW/NQ);
device const float * yb = y + ib0*QK8_0 + il*NQ;
```

A Q8_0 block holds 32 values. With `NQ = 8`, four lanes cover a block, so the
divisor is `QK8_0/NQ`, which is 4 at **any** width. Using `NW/NQ` makes `il`
run to 7 at width 64, so `il*NQ` reaches 56 inside a 32-element block: the
kernel reads past the block into the next one.

## Bug 2 — the activation pointer kept the old stride

The loop bound had to become the number of blocks a whole simdgroup covers:

```c
for (int ib = ib0; ib < nb; ib += (NW*NSG)/NLB)   // was NSG*NQ
```

but the pointer advance was left alone:

```c
yb += NSG*NQ*QK8_0;                                // 16 blocks, loop steps 32
```

Every iteration after the first then read activations belonging to a different
set of blocks. This one is subtle because it is correct at width 32 and correct
when `nb` fits in a single iteration — which is why `k=256` and `k=1024` passed
while `k=4096` and `k=5120` failed.

## Bug 3 - the support gate refused the type

The two lane bugs above are unreachable until the op is allowed on the device.
`ggml_metal_supports_mul_mat_wide` lists the types ported for the real
simdgroup width, and Q8_0 was not in it, so `MUL_MAT` reported "not supported"
before any kernel ran and the layer bounced to the CPU:

```c
case GGML_TYPE_Q5_K:
case GGML_TYPE_Q6_K:
case GGML_TYPE_Q8_0:   // added
    return true;
```

## Result

```
test-backend-ops test -o MUL_MAT -p 'type_a=q8_0' -b MTL0
116/116 tests passed
```

Ling-3.0-tiny Q6_K on an RX 580, three greedy prompts:

| | Q8_0 on CPU | Q8_0 on GPU (fixed) |
|---|---|---|
| generation | 21.3 t/s | **24.9-25.2 t/s** |
| output | verified | **byte-identical** |

Worth recording how the difference announced itself: the *unfixed* kernel also
reached 25.2 t/s, but its three outputs were paraphrases of the correct
answers. A wrong kernel in a 16-bit format does not produce garbage — it
produces confident, slightly different text. Byte-identical greedy output is
the check that separates a numerical reordering from a bug, and here it agreed
with the 116/116 result rather than contradicting it.

## Who this affects

Every non-Apple GPU running llama.cpp: any model with Q8_0 tensors — which
includes most Q6_K and "high quality" GGUFs, since quantisers commonly keep
some tensors at Q8_0 — has been running those matmuls on the CPU, once per
layer, with a graph split on each side.
