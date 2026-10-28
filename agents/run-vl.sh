#!/bin/bash
# LFM3-VL-2B as a service: the vision model behind an OpenAI-compatible endpoint.
#
#   ./run-vl.sh                      # :8093, RX 580 when attached
#   PORT=8094 ./run-vl.sh            # another port
#
# Why a separate server: vision and text are different checkpoints with their
# own projectors. The card holds one comfortably; running both at once is
# possible but tight, so this script uses a small -c (2048) that image Q&A
# does not miss, and says so instead of failing silently.
#
# --image-max-tokens 256 is not cosmetic: without it the vision encoder sizes
# its compute buffer from the image, and a 1680x2866 chart asked for 16.9 GB
# and died on an 8 GB card.
set -u

MODEL=${MODEL:-$HOME/models/lfm3-vl-2b/LFM3-VL-2B-Q5_K_M.gguf}
MMPROJ=${MMPROJ:-$HOME/models/lfm3-vl-2b/mmproj-LFM3-VL-2B-F16.gguf}
BIN=${BIN:-llama-server}
PORT=${PORT:-8093}
CTX=${CTX:-2048}
REASONING=${REASONING:-on}
BASE="http://127.0.0.1:$PORT"

die() { echo "run-vl: $*" >&2; exit 1; }

[ -x "$BIN" ]    || die "server binary not found: $BIN"
[ -f "$MODEL" ]  || die "model not found: $MODEL"
[ -f "$MMPROJ" ] || die "mmproj not found: $MMPROJ"

if curl -sf -m 1 "$BASE/health" >/dev/null 2>&1; then
    die "something already answers on port $PORT - stop it first"
fi

gpu_attached() { system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Radeon RX 580"; }

wait_healthy() {
    local pid=$1
    for _ in $(seq 1 90); do
        kill -0 "$pid" 2>/dev/null || return 1
        curl -sf -m 1 "$BASE/health" 2>/dev/null | grep -q '"ok"' && return 0
        sleep 1
    done
    return 1
}

if [ "${LFM3_FORCE:-}" != "cpu" ] && gpu_attached; then
    echo "run-vl: RX 580 attached -> GPU (-ngl 999 -lm none, -c $CTX)"
    GGML_METAL_DEVICE_NAME="Radeon RX 580" \
        "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --mmproj-offload \
        -ngl 999 -lm none -t 1 -c "$CTX" --image-max-tokens 256 --metrics \
        -rea "$REASONING" --port "$PORT" &
    pid=$!
    if wait_healthy "$pid"; then
        wait "$pid"
        exit $?
    fi
    kill "$pid" 2>/dev/null
    wait "$pid" 2>/dev/null
    echo "run-vl: GPU path did not come up (out of VRAM? stop :8090) - trying the CPU" >&2
else
    echo "run-vl: no RX 580 -> CPU (-ngl 0 -t 4), slow but correct"
fi

exec "$BIN" -m "$MODEL" --mmproj "$MMPROJ" --no-mmproj-offload --no-mmproj-auto \
    -ngl 0 -t 4 -c "$CTX" --image-max-tokens 256 --metrics -rea "$REASONING" --port "$PORT"
