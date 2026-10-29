# Finding 8 — no MoE model runs on the GPU on a non-Apple device

Every mixture-of-experts model silently runs its expert matmuls on the CPU on
this backend, while the weights sit in VRAM. The model loads, the placement
table looks right, and generation crawls. One capability gate causes it, and
the gate is one line.

## The gate

`ggml-metal-device.m`, `ggml_metal_device_supports_op`:

```objc
case GGML_OP_MUL_MAT_ID:
    if (!has_simdgroup_reduction) {
        return false;
    }
```

`has_simdgroup_reduction` is set from `MTLGPUFamilyApple7 || Metal3`, so it is
**false on every AMD and Intel GPU**. `MUL_MAT_ID` is the op that runs the
routed expert matmuls, so on those devices the whole expert half of the graph
is assigned to the CPU — the same failure mode as the `SSM_CONV` gate in
Finding 4, and the same root cause: a capability flag for one Apple feature
used as a proxy for an unrelated one.

The kernel to run it **already existed and was already correct**. `mul_mv.metal`
instantiates `kernel_mul_mv_id_q5_K_f32` (and the q6_K/q4_K/id variants) from
the same matvec templates that pass 87/87 on this device.

## Evidence

Ling-3.0-tiny Q6_K: 7.9B parameters, 128 experts, 8 used per token, 1 shared.
Measured on a Radeon RX 580 eGPU under macOS, `-ngl 999 -lm none -t 1 -c 4096
-ub 256`.

| | before | after |
|---|---|---|
| prompt processing | 2.97 t/s | **27.2 t/s** |
| generation | requests did not finish (180 s timeout) | **7.8 t/s** |
| `MTL0_Private` model buffer | 5996 MiB | 5996 MiB |
| `CPU` model buffer | 195 MiB | 195 MiB |

The placement line is the diagnostic: 6.0 GB of experts were **already** in
VRAM in both runs. Only the compute moved. That is why the model looked like a
slow model instead of a mis-dispatched one.

Correctness after the change, against the CPU reference:

```
test-backend-ops test -o MUL_MAT_ID -b MTL0
224/224 tests passed
```

For scale: a dense 9B at Q5_K manages 3.7 t/s on the same card with a file of
the same size. The 7.9B MoE at Q6_K is **2.1x faster** because a token only
streams the experts it routes to.

## The fix

Mirror what `MUL_MAT` already does for width-64 devices: fall back to the
type whitelist instead of refusing outright.

```objc
case GGML_OP_MUL_MAT_ID:
    if (!has_simdgroup_reduction) {
        // the expert matmul reuses the same matvec kernels, so it is
        // subject to the same width-64 list as MUL_MAT
        return dev->props.simd_width > 0 && ggml_metal_supports_mul_mat_wide(op);
    }
```

`ggml_metal_supports_mul_mat_wide` admits F32, F16, Q5_K and Q6_K. On a device
with simdgroup reduction nothing changes.

## Who this affects

Anyone running llama.cpp's Metal backend on a non-Apple GPU: AMD eGPUs on Intel
Macs, Intel integrated GPUs, and any Mac where the Metal device is not an
Apple family. On those machines no MoE model — Qwen MoE, DeepSeek, Mixtral,
Ling, gpt-oss — has ever used the GPU for its experts.

## How to check your own device

```bash
# 1. is the op even offered?
./build/bin/test-backend-ops support -o MUL_MAT_ID -b MTL0

# 2. does the model's placement look right but the speed not?
#    a MoE load should show most MiB in MTL0_Private and little in CPU;
#    if the ratio is right and tokens/s is not, the op is on the CPU.
```

A caveat worth stating: this was measured on one card with one model family.
The 224/224 result covers the shapes `test-backend-ops` generates, not every
expert count or quantisation. Run the correctness suite on your own device
before trusting it.
