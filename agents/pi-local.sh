#!/bin/bash
# Run Pi (the coding-agent harness LFM3 was post-trained in) against the local
# server on :8090.
#
# Bun, not node: Homebrew's node at /usr/local/bin/node links a simdjson soname
# that is no longer installed (libsimdjson.33 vs 34), so anything launched
# through its shebang aborts. Bun runs Pi's ESM entry point directly.
#
#   pi-local -p "list the files here" --mode text      # one shot
#   pi-local                                          # interactive
#
# It starts run-rx580.sh when :8090 is down and leaves it running: the first
# call after a start reads ~3.9k prompt tokens (~117 s), every later call hits
# the prompt cache (~22 s). Restarting the server is what makes it feel slow.
set -u

BUN=${BUN:-$(command -v bun 2>/dev/null || echo "$HOME/.local/share/mise/installs/bun/latest/bin/bun")}
MODEL=${PI_MODEL:-LFM3-1.7B}
URL=${LFM3_URL:-http://127.0.0.1:8090}

# Resolve this script through any symlink, so ~/.local/bin/pi-local can find
# its sibling launcher.
SELF_SCRIPT=$0
while [ -L "$SELF_SCRIPT" ]; do
    link=$(readlink "$SELF_SCRIPT")
    case $link in /*) SELF_SCRIPT=$link ;; *) SELF_SCRIPT=$(dirname "$SELF_SCRIPT")/$link ;; esac
done
LAUNCHER=$(cd "$(dirname "$SELF_SCRIPT")" && pwd -P)/run-rx580.sh

# Pi's entry point moved between versions (dist/cli.js -> dist/bundle/cli.js),
# so it is resolved from the installed symlink rather than hardcoded.
PI_BIN=${PI_BIN:-$(command -v pi 2>/dev/null || true)}
if [ -z "$PI_BIN" ]; then
    for candidate in "$HOME"/.local/share/mise/installs/node/*/bin/pi; do
        [ -e "$candidate" ] && PI_BIN=$candidate
    done
fi
[ -n "$PI_BIN" ] || { echo "pi-local: pi not installed (npm i -g @earendil-works/pi-coding-agent)" >&2; exit 1; }

PI_ENTRY=$PI_BIN
while [ -L "$PI_ENTRY" ]; do
    link=$(readlink "$PI_ENTRY")
    case $link in /*) PI_ENTRY=$link ;; *) PI_ENTRY=$(dirname "$PI_ENTRY")/$link ;; esac
done

[ -x "$BUN" ]     || { echo "pi-local: bun not found at $BUN" >&2; exit 1; }
[ -f "$PI_ENTRY" ] || { echo "pi-local: pi entry not found at $PI_ENTRY" >&2; exit 1; }
[ -x "$LAUNCHER" ] || { echo "pi-local: launcher not found at $LAUNCHER" >&2; exit 1; }

healthy() { curl -sf -m 2 "$URL/health" 2>/dev/null | grep -q '"ok"'; }

if ! healthy; then
    echo "pi-local: no server on $URL - starting run-rx580.sh" >&2
    PORT="${URL##*:}" nohup "$LAUNCHER" >/tmp/lfm3-8090.log 2>&1 &
    for _ in $(seq 1 90); do
        healthy && break
        sleep 1
    done
    if ! healthy; then
        echo "pi-local: server did not come up; last lines of /tmp/lfm3-8090.log:" >&2
        tail -5 /tmp/lfm3-8090.log >&2
        exit 1
    fi
fi

exec "$BUN" "$PI_ENTRY" --provider local --model "$MODEL" "$@"
