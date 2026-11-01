# Patches

Three files, and the order matters. Every one applies to `a7af149` and nothing else.

| file | covers | prerequisite |
|---|---|---|
| `rx580-metal-width64.patch` | findings 1-10 (findings 1-7 for a dense model, 8-10 for MoE/hybrid) | clean `a7af149` |
| `rx580-metal-width64-fa-q4k.patch` | findings 11-12 (flash attention at width 64, the Q4_K wide path) | the patch above, applied first |
| `upstream-metal-fixes.patch` | findings 3 and 4 only, for anyone who wants the two safe ones alone | clean `a7af149`, **instead of** the first |

## Apply

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout a7af149

git apply --check /path/to/rx580-metal-width64.patch        # must print nothing
git apply         /path/to/rx580-metal-width64.patch

git apply --check /path/to/rx580-metal-width64-fa-q4k.patch # must print nothing
git apply         /path/to/rx580-metal-width64-fa-q4k.patch
```

A rejected hunk is not a warning to ignore: it means the checkout is not `a7af149`
and the result is the slow state, not a partial fix.

## What each one was verified against

`rx580-metal-width64.patch` + `rx580-metal-width64-fa-q4k.patch`, applied to a
fresh `a7af149` tree and built from scratch with
`-DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release`:

```
build                                         0 errors
test-backend-ops -o FLASH_ATTN_EXT            4116/4116 passed
test-backend-ops  (forced (Q,NE) slice)        240/240 passed
test-backend-ops -o MUL_MAT,MUL_MAT_ID        1384/1384 passed
```

The flash-attention count is 4116, not 2024, because this patch also lets the vec
kernel serve prefill batches: `use_vec` no longer refuses `ne01 >= 20`, and the
reduce temp buffer in `ggml_metal_op_flash_attn_ext_extra_tmp` is sized by query
rows instead of being clamped to 32. On Ling-3.0-tiny that turns 22.62 t/s of
prefill into 61.38 (decode 23.40), measured on one 8,222-token prompt.

The second patch touches 17 files: the eight `kernels/fa_vec_*.metal`, `fa_aux.metal`,
`mul_mv.metal`, `ggml-metal-tuning.h/.cpp`, `ggml-metal-ops.h/.cpp`,
`ggml-metal-device.cpp/.m` and one argument in `ggml-metal.cpp`.

It is extracted from the working tree that produced the numbers, and it
deliberately does **not** carry everything that tree contains. Left out, because it
belongs to other work and would misrepresent this one: the grouped-MoE routing
fusion (`GGML_METAL_FUSION_TOPK_MOE_GROUPED`), the tiled GEMM for devices without
simdgroup matrix, the custom `argsort` kernel, and the debug prints
(`GGML_METAL_DISP_LOG`, `GGML_SPLIT_DEBUG`, `GGML_METAL_GPU_TIME`, `METAL bigbuf`).
A patch is a claim; those are different claims.

## Known gap in both patches

`q4_0`, `q4_1`, `q5_0` and `q5_1` mat-vec do not work at width 64. They are absent
from the wide allow-list on purpose: when forced onto it they fail every case
(`0/121`, `0/99`, `0/63`, `0/56`). See the last section of the README.
