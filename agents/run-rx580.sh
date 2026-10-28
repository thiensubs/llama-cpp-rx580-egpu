#!/bin/bash
# Start llama-server for LFM3 on the RX580 eGPU when it is attached, on the CPU
# when it is not, and on the CPU if the GPU path does not come up.
#
# GPU recipe (measured 2026-10-28): prompt 36 t/s, generation 25 t/s,
# 0.008 CPU-seconds per token.
#   -ngl 999        whole graph on the GPU
#   -lm none        REQUIRED: mmap makes llama.cpp wrap the weights with
#                   ggml_metal_buffer_map() -> shared host memory, and the GPU
#                   then reads them over PCIe (25 t/s -> 0.8 t/s)
#   -t 1            with the graph on the GPU the CPU threads only spin
#    eGPU defaults to shared buffers
#   GGML_METAL_DEVICE_NAME               select by name: the index of an eGPU
#                                        shifts when it is unplugged
set -u

MODEL=${MODEL:-$HOME/models/LFM3-1.7B-Q5_K_M.gguf}
BIN=${BIN:-llama-server}
PORT=${PORT:-8090}
CTX=${CTX:-65536}
# Optional KV quantisation, e.g. KVQ="-ctk q8_0 -ctv q8_0". Only needed for
# very long context: at -c 131072 fp16 KV fails to allocate, q8_0 fits in 2.1 GB
# at the cost of ~30% generation speed (24.5 -> 17.3 t/s).
KVQ=${KVQ:-}
BASE="http://127.0.0.1:$PORT"

die() { echo "run-rx580: $*" >&2; exit 1; }

[ -x "$BIN" ]  || die "server binary not found: $BIN"
[ -f "$MODEL" ] || die "model not found: $MODEL"

if curl -sf -m 1 "$BASE/health" >/dev/null 2>&1; then
    die "something already answers on port $PORT - stop it first"
fi

gpu_attached() {
    system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Radeon RX 580"
}

wait_healthy() {
    local pid=$1
    for _ in $(seq 1 60); do
        kill -0 "$pid" 2>/dev/null || return 1
        curl -sf -m 1 "$BASE/health" 2>/dev/null | grep -q '"ok"' && return 0
        sleep 1
    done
    return 1
}

if [ "${LFM3_FORCE:-}" != "cpu" ] && gpu_attached; then
    echo "run-rx580: RX 580 attached -> GPU (-ngl 999 -lm none -t 1)"
    GGML_METAL_DEVICE_NAME="Radeon RX 580" \
        "$BIN" -m "$MODEL" -ngl 999 -lm none -t 1 -c "$CTX" $KVQ --metrics --port "$PORT" &
    pid=$!
    # a half-seated eGPU can enumerate and then fail; do not leave a dead server
    if wait_healthy "$pid"; then
        wait "$pid"
        exit $?
    fi
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "run-rx580: GPU path did not come up, falling back to the CPU" >&2
else
    echo "run-rx580: RX 580 not available -> CPU (-ngl 0 -t 4)"
fi

exec "$BIN" -m "$MODEL" -ngl 0 -t 4 -c "$CTX" --metrics --port "$PORT"
