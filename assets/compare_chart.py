#!/usr/bin/env python3
"""Render the two real pi-local runs as one comparison figure (SVG).

usage: compare_chart.py <facts.json> <a.csv> <b.csv> <out.svg>

facts.json:
{"a": {"label": ..., "wall_s": ..., "gpu_busy_pct": ...,
       "requests": [{"prefill_tok": .., "prefill_s": .., "gen_tok": .., "gen_s": ..}, ...]},
 "b": {...}}
Rows in the CSVs come from sampler.py: t_s, rss_mb, cpu_pct, vram_used_mb, vram_free_mb, gpu_pct
"""
import csv
import json
import sys

W, PADL, PADR, RIGHT = 1560, 232, 300, 220
INK, MUTE, GRID, FAINT = "#16181d", "#6b7280", "#e6e8eb", "#f4f5f7"
C_A, C_B = "#1f9d55", "#6d4aff"          # A = LFM2.5, B = Ling
C_PRE, C_GEN = "#c9ced6", None            # prefill grey, generation = run colour

FACTS, CSV_A, CSV_B, OUT = sys.argv[1:5]
facts = json.load(open(FACTS))


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            def num(k):
                v = r.get(k) or ""
                try:
                    return float(v)
                except ValueError:
                    return None
            rows.append({"t": num("t_s"), "rss": num("rss_mb"), "vram": num("vram_used_mb"), "gpu": num("gpu_pct")})
    return [r for r in rows if r["t"] is not None]


A, B = load(CSV_A), load(CSV_B)
runs = [("a", A, C_A, facts["a"]), ("b", B, C_B, facts["b"])]
WALL = max(f["wall_s"] for _, _, _, f in runs)
TMAX = ((int(WALL) + 59) // 60) * 60

parts = []
add = parts.append
H = 940
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="-apple-system, Helvetica, Arial, sans-serif">')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
add(f'<text x="{PADL}" y="44" font-size="22" font-weight="700" fill="{INK}">Two real pi-local runs on an RX 580 eGPU</text>')
add(f'<text x="{PADL}" y="68" font-size="13" fill="{MUTE}">same task, same harness, same flags, one server at a time · bars: completed runs · memory traces: sampled repeats</text>')

X0, X1 = PADL, W - RIGHT
sx = lambda t: X0 + (X1 - X0) * min(t, TMAX) / TMAX

# ---------- panel 1: timeline ----------
Y, PH = 104, 108
add(f'<text x="{PADL}" y="{Y - 12}" font-size="13" font-weight="600" fill="{INK}">Timeline to done — prefill (grey) then generation (colour)</text>')
for t in range(0, int(TMAX) + 1, 60):
    add(f'<line x1="{sx(t):.1f}" y1="{Y}" x2="{sx(t):.1f}" y2="{Y + 2 * PH + 26}" stroke="{FAINT}" stroke-width="1"/>')
    add(f'<text x="{sx(t):.1f}" y="{Y + 2 * PH + 44}" font-size="11" fill="{MUTE}" text-anchor="middle">{t}s</text>')
for i, (key, rows, colour, f) in enumerate(runs):
    yb = Y + i * (PH + 26) + 26
    add(f'<text x="{X0 - 12}" y="{yb + 12}" font-size="12.5" font-weight="600" fill="{colour}" text-anchor="end">{f["label"]}</text>')
    start = 0.0
    for r in f["requests"]:
        w1 = (r["prefill_s"] / TMAX) * (X1 - X0)
        w2 = (r["gen_s"] / TMAX) * (X1 - X0)
        add(f'<rect x="{sx(start):.1f}" y="{yb}" width="{w1:.1f}" height="28" fill="{C_PRE}" stroke="#b9bfc8" stroke-width="0.8"/>')
        add(f'<rect x="{sx(start + r["prefill_s"]):.1f}" y="{yb}" width="{max(w2, 1.2):.1f}" height="28" fill="{colour}" opacity="0.9"/>')
        pt = r["prefill_tok"] / r["prefill_s"]
        gt = r["gen_tok"] / r["gen_s"]
        if w1 > 92:
            add(f'<text x="{sx(start) + 6:.1f}" y="{yb + 19}" font-size="11" fill="#3c424b">{r["prefill_tok"]:,} tok · {pt:.0f} t/s</text>')
        else:
            add(f'<text x="{sx(start + r["prefill_s"]) + 5:.1f}" y="{yb - 6}" font-size="10.5" fill="{MUTE}">{r["prefill_tok"]:,} @ {pt:.0f} t/s</text>')
        label = f'{r["gen_tok"]} tok · {gt:.0f} t/s'
        gx = sx(start + r["prefill_s"])
        if w2 > 66:
            add(f'<text x="{gx + 5:.1f}" y="{yb + 19}" font-size="10.5" fill="#ffffff">{label}</text>')
        else:
            add(f'<text x="{sx(start + r["prefill_s"] + r["gen_s"]) + 5:.1f}" y="{yb + 19}" font-size="10.5" fill="{colour}">{label}</text>')
        start += r["prefill_s"] + r["gen_s"]
    add(f'<line x1="{sx(start):.1f}" y1="{yb - 8}" x2="{sx(start):.1f}" y2="{yb + 36}" stroke="{colour}" stroke-width="1.4" stroke-dasharray="4 4"/>')
    add(f'<text x="{sx(start) + 6:.1f}" y="{yb + 46}" font-size="11" font-weight="600" fill="{colour}">done {f["wall_s"]:.0f} s</text>')

# ---------- panel 2: throughput bars ----------
Y2 = Y + 2 * PH + 96
add(f'<text x="{PADL}" y="{Y2 - 12}" font-size="13" font-weight="600" fill="{INK}">Throughput per model — prefill and decode, both turns</text>')
cats = []
for key, rows, colour, f in runs:
    for j, r in enumerate(f["requests"], 1):
        cats.append((f'{f["short"]} · turn {j}', r["prefill_tok"] / r["prefill_s"], colour, 0.45))
        cats.append((f'{f["short"]} · turn {j} gen', r["gen_tok"] / r["gen_s"], colour, 1.0))
VMAX = 60.0
BARH, GAP = 16, 9
for i, (label, v, colour, alpha) in enumerate(cats):
    y = Y2 + i * (BARH + GAP)
    add(f'<text x="{X0 - 10}" y="{y + 12}" font-size="11" fill="{INK}" text-anchor="end">{label}</text>')
    width = (X1 - X0 - 40) * min(v, VMAX) / VMAX
    add(f'<rect x="{X0}" y="{y}" width="{width:.1f}" height="{BARH}" fill="{colour}" opacity="{alpha}" rx="2"/>')
    add(f'<text x="{X0 + width + 8:.1f}" y="{y + 12}" font-size="11" fill="{MUTE}">{v:.1f} t/s</text>')
    add(f'<line x1="{X0}" y1="{y + BARH / 2}" x2="{X0}" y2="{y + BARH / 2}" stroke="{GRID}"/>')

# ---------- panel 3: memory ----------
Y3 = Y2 + len(cats) * (BARH + GAP) + 54
PH3 = 150
add(f'<text x="{PADL}" y="{Y3 - 12}" font-size="13" font-weight="600" fill="{INK}">Memory over the run — GPU VRAM used (thick) and host RSS (thin)</text>')
for frac in (0, 0.25, 0.5, 0.75, 1.0):
    yy = Y3 + PH3 - PH3 * frac
    add(f'<line x1="{X0}" y1="{yy:.1f}" x2="{X1}" y2="{yy:.1f}" stroke="{GRID}" stroke-width="1"/>')
    add(f'<text x="{X0 - 10}" y="{yy + 4:.1f}" font-size="11" fill="{MUTE}" text-anchor="end">{int(9000 * frac):,}</text>')
add(f'<text x="{X0 - 10}" y="{Y3 - 20}" font-size="10.5" fill="{MUTE}" text-anchor="end">MB</text>')
for t in range(0, int(TMAX) + 1, 60):
    add(f'<text x="{sx(t):.1f}" y="{Y3 + PH3 + 18}" font-size="11" fill="{MUTE}" text-anchor="middle">{t}</text>')
for key, rows, colour, f in runs:
    for metric, width, dash, op in (("vram", 2.6, "", 1.0), ("rss", 1.4, "4 3", 0.75)):
        pts = [(r["t"], r[metric]) for r in rows if r[metric] is not None]
        if not pts:
            continue
        d = " ".join(
            (f'M {sx(t):.1f} {Y3 + PH3 - PH3 * min(v, 9000) / 9000:.1f}' if i == 0
             else f'L {sx(t):.1f} {Y3 + PH3 - PH3 * min(v, 9000) / 9000:.1f}')
            for i, (t, v) in enumerate(pts))
        add(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{width}" opacity="{op}"'
            + (f' stroke-dasharray="{dash}"' if dash else '') + '/>')

# ---------- side stats ----------
lx, ly = X1 + 26, Y + 4
add(f'<text x="{lx}" y="{ly}" font-size="12" font-weight="700" fill="{INK}">measured</text>')
ly += 22
for key, rows, colour, f in runs:
    vram = [r["vram"] for r in rows if r["vram"] is not None]
    rss = [r["rss"] for r in rows if r["rss"] is not None]
    resident = [r for r in rows if (r["vram"] or 0) > 1000 and r["gpu"] is not None]
    gpu = [r["gpu"] for r in resident]
    busy = 100.0 * sum(1 for g in gpu if g >= 50) / len(gpu) if gpu else 0
    add(f'<circle cx="{lx + 5}" cy="{ly - 4}" r="5" fill="{colour}"/>')
    add(f'<text x="{lx + 18}" y="{ly}" font-size="12.5" font-weight="600" fill="{INK}">{f["label"]}</text>')
    ly += 17
    for line in (f'wall {f["wall_s"]:.0f} s', f'GPU busy {busy:.0f}% (resident)',
                 f'peak VRAM {max(vram):.0f} MB' if vram else 'peak VRAM —',
                 f'peak RSS {max(rss):.0f} MB' if rss else 'peak RSS —'):
        add(f'<text x="{lx + 18}" y="{ly}" font-size="11" fill="{MUTE}">{line}</text>')
        ly += 15
    ly += 12
add(f'<text x="{lx}" y="{ly + 4}" font-size="11" fill="{MUTE}">VRAM = inUseVidMemoryBytes</text>')
add(f'<text x="{lx}" y="{ly + 20}" font-size="11" fill="{MUTE}">RSS = llama-server resident</text>')
add(f'<text x="{lx}" y="{ly + 36}" font-size="11" fill="{MUTE}">sampled every ~4 s</text>')
add(f'<text x="{lx}" y="{ly + 58}" font-size="11" fill="{MUTE}">for contrast: RSS was 2.97 GB</text>')
add(f'<text x="{lx}" y="{ly + 74}" font-size="11" fill="{MUTE}">while Q4_K still ran on the CPU</text>')

add("</svg>")
open(OUT, "w").write("\n".join(parts))
print("wrote", OUT)
