#!/usr/bin/env python3
"""Render Figure 1 from its TikZ source to SVG.

The HTML build of this figure is broken, so it is redrawn here from the
LaTeX. TikZ coordinates are centimetres with y pointing up; this maps 1cm to 100
user units and flips y, so every coordinate below is the one in the .tex.

Type is the part that is easy to get wrong: TikZ sizes text in points on a
canvas measured in centimetres, so \\scriptsize (7pt) is 24.6 user units here,
not 8. Boxes declare a `text width`, and TeX wraps to it, so the wrapping is
done here too rather than hard-coding line breaks.

Usage:  python3 make_figure1.py assets/figure1.svg
"""
import sys
from html import escape

U = 100.0                      # user units per cm
PT = U / 28.4527               # user units per TeX point
H, W = 6.55, 22.85             # canvas, cm

SCRIPT, FOOT, SMALL = 7 * PT, 8 * PT, 9 * PT
LH = 8.6 * PT                  # baselineskip at \scriptsize

def X(cm): return round(cm * U, 2)
def Y(cm): return round((H - cm) * U, 2)
def L(cm): return round(cm * U, 2)

C = {
    "truth": "#3E6ECD", "truthfill": "#E9F1FF",
    "decep": "#CB4F4F", "decepfill": "#FCECEC",
    "accent": "#D97706", "accentfill": "#FFF4DB",
    "panelbg": "#F9FAFB", "muted": "#6B7280",
    "ink": "#111111", "box": "#D6D8DC", "rule": "#9AA0A6",
}

def mix(hexa, pct, other="#000000"):
    """TikZ's `colour!p!black`."""
    a = [int(hexa[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a[i]*pct/100 + b[i]*(100-pct)/100):02x}" for i in range(3))

# ---- text metrics ---------------------------------------------------------
NARROW, WIDE = set("ijltfrI.,;:'’!|()[]-“” "), set("mwMW@%")
def adv(ch, size):
    """Approximate advance for IBM Plex Sans, in user units."""
    if ch in NARROW: return 0.30 * size
    if ch in WIDE:   return 0.86 * size
    if ch.isupper(): return 0.66 * size
    if ch.isdigit(): return 0.55 * size
    return 0.53 * size

def measure(s, size): return sum(adv(c, size) for c in s)

def wrap(s, width_cm, size):
    """Greedy wrap to a TikZ `text width`, in cm."""
    limit, lines, cur = width_cm * U, [], ""
    for word in s.split():
        trial = word if not cur else cur + " " + word
        if measure(trial, size) <= limit or not cur:
            cur = trial
        else:
            lines.append(cur); cur = word
    if cur: lines.append(cur)
    return lines

out = []
def add(s): out.append(s)

def rect(x, y, w, h, fill, stroke, rx=6, sw=0.8, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{X(x)}" y="{Y(y+h)}" width="{L(w)}" height="{L(h)}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def line(x1, y1, x2, y2, stroke, sw=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{X(x1)}" y1="{Y(y1)}" x2="{X(x2)}" y2="{Y(y2)}" '
        f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"{d}/>')

def circle(x, y, r_cm, fill="none", stroke="none", sw=1.0):
    add(f'<circle cx="{X(x)}" cy="{Y(y)}" r="{L(r_cm)}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>')

def text(x, y, s, size=SCRIPT, fill=C["ink"], anchor="start", weight="400", italic=False):
    st = ' font-style="italic"' if italic else ""
    add(f'<text x="{X(x)}" y="{Y(y)}" font-family="\'IBM Plex Sans\',system-ui,sans-serif" '
        f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}"{st}>{escape(s)}</text>')

def para(x, y_top, rows, size=SCRIPT, fill=C["ink"], anchor="start", lh=LH):
    """rows: list of (string, weight). y_top is the first baseline, in cm."""
    for i, (s, w) in enumerate(rows):
        text(x, y_top - i * lh / U, s, size=size, fill=fill, anchor=anchor, weight=w)

BOXES = []          # (name, x, y, w, h, is_frame) for the geometry check

def node(name, x, y, content, width=None, fill="#ffffff", stroke=None,
         anchor="nw", xsep=5, ysep=4, rx=6, sw=0.8, dash=None, align="left"):
    """A TikZ node.

    content is a list of (text, weight, size, wrap?) rows. A row with wrap=False
    is an explicit `\\` break in the source and is never re-flowed; a row with
    wrap=True is a paragraph TeX would wrap to `width`. A node with no `width`
    sizes to its widest line, exactly as TikZ does.
    """
    xs, ys = xsep * PT / U, ysep * PT / U
    rows = []
    for txt, wt, size, do_wrap in content:
        if do_wrap and width:
            for ln in wrap(txt, width, size):
                rows.append((ln, wt, size))
        else:
            rows.append((txt, wt, size))

    inner = width if width else max(measure(t, sz) for t, _, sz in rows) / U
    w = inner + 2 * xs
    h = len(rows) * LH / U + 2 * ys

    if anchor == "nw":   bx, by = x, y - h
    elif anchor == "n":  bx, by = x - w / 2, y - h
    elif anchor == "w":  bx, by = x, y - h / 2
    elif anchor == "c":  bx, by = x - w / 2, y - h / 2
    else: raise ValueError(anchor)

    if fill != "none" or stroke:
        rect(bx, by, w, h, fill, stroke or "none", rx=rx, sw=sw, dash=dash)
    BOXES.append((name, bx, by, w, h, False))

    first = by + h - ys - LH * 0.78 / U
    for i, (txt, wt, size) in enumerate(rows):
        col = C["muted"] if wt == "muted" else C["ink"]
        weight = "600" if wt == "b" else "400"
        if align == "center":
            text(bx + w / 2, first - i * LH / U, txt, size=size, weight=weight,
                 fill=col, anchor="middle")
        else:
            text(bx + xs, first - i * LH / U, txt, size=size, weight=weight, fill=col)
    return bx, by, w, h

def frame(name, x, y, w, h, stroke, rx=6, sw=1.0, dash=None):
    rect(x, y, w, h, "none", stroke, rx=rx, sw=sw, dash=dash)
    BOXES.append((name, x, y, w, h, True))

def arrow(x1, y1, x2, y2, stroke, sw=1.0, dash=None, head=0.085):
    import math
    line(x1, y1, x2, y2, stroke, sw, dash)
    a = math.atan2(-(y2 - y1), x2 - x1)
    for k in (0.38, -0.38):
        add(f'<line x1="{X(x2)}" y1="{Y(y2)}" '
            f'x2="{X(x2) - L(head)*math.cos(a - k)}" '
            f'y2="{Y(y2) + L(head)*math.sin(a - k)}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"/>')

def badge(x, y, n):
    circle(x, y, 0.265, fill="#000000")
    text(x, y - SCRIPT * 0.34 / U, n, size=SCRIPT, fill="#ffffff", anchor="middle", weight="700")

# ------------------------------------------------------------------ panels --
for x0, x1 in ((0.0, 7.8), (8.2, 16.8), (17.2, 22.85)):
    rect(x0, 0.0, x1 - x0, 6.55, C["panelbg"], "#E4E6E8", rx=10, sw=0.8)

S, F, M = SCRIPT, FOOT, SMALL
def L_(t, w="n", sz=None): return (t, w, sz or S, False)   # explicit \\ line
def P_(t, w="n", sz=None): return (t, w, sz or S, True)    # wrapped paragraph

# ================================================================= stage 1 ==
badge(0.65, 5.98, "1")
text(0.95, 5.92, "Deception Mining", size=M, weight="700")
text(0.95, 5.60, "sample multiple trajectories from one prompt/state",
     size=S * 0.86, fill=C["muted"])

node("prompt", 0.45, 5.20, [
        L_("Prompt", "b"), L_("required rank: 7"), L_("hand: [7, K]"),
        P_("instruction: play one card face down and state its rank"),
     ], width=2.35, fill="#ffffff", stroke=C["box"], xsep=6, ysep=5)

node("honest", 0.45, 2.55,
     [L_("honest", "b"), L_("play 7"), L_("say “7”")],
     width=2.45, fill=C["truthfill"], stroke=mix(C["truth"], 65))
node("deceptive", 0.45, 1.30,
     [L_("deceptive", "b"), L_("play K"), L_("say “7”")],
     width=2.45, fill=C["decepfill"], stroke=mix(C["decep"], 65))

YS = -0.10                                   # the tex shifts this scope down
text(4.30, 5.40 + YS, "sᵢ = sentence i", size=S * 0.86, fill=C["muted"])

MINI_W, MINI_H = 0.72, 0.48                  # sentmini minimum width / height
COLS = (4.55, 5.35, 6.15)
ROWS = ((4.95, C["truth"]), (4.05, C["truth"]),
        (3.15, C["decep"]), (2.25, C["decep"]))
mini = {}                                    # (row, col) -> centre, for the fit
for r, (ry, dot) in enumerate(ROWS):
    ry += YS
    for i, cx in enumerate(COLS):
        rect(cx - MINI_W / 2, ry - MINI_H / 2, MINI_W, MINI_H, "#ffffff", C["box"], rx=3)
        text(cx, ry - S * 0.34 / U, f"s{i+1}", size=S, anchor="middle", weight="600")
        mini[(r, i)] = (cx, ry)
    line(4.91, ry, 4.99, ry, C["rule"], 0.8)
    line(5.71, ry, 5.79, ry, C["rule"], 0.8)
    line(6.51, ry, 6.85, ry, C["rule"], 0.8)
    circle(7.20, ry, 0.027, fill=dot)

# TikZ: fit=(m21)(m23)(m31)(m33) with inner sep=6pt. Computed from the nodes it
# fits rather than hardcoded, so the outline cannot cut through them.
FIT_SEP = 6 * PT / U
fx = [mini[k][0] for k in ((1, 0), (1, 2), (2, 0), (2, 2))]
fy = [mini[k][1] for k in ((1, 0), (1, 2), (2, 0), (2, 2))]
fx0, fx1 = min(fx) - MINI_W / 2 - FIT_SEP, max(fx) + MINI_W / 2 + FIT_SEP
fy0, fy1 = min(fy) - MINI_H / 2 - FIT_SEP, max(fy) + MINI_H / 2 + FIT_SEP
frame("pair", fx0, fy0, fx1 - fx0, fy1 - fy0, mix(C["accent"], 75))
for ry in (4.05, 3.15):
    circle(7.20, ry + YS, 0.046, stroke=mix(C["accent"], 75), sw=1.0)

# no `text width` in the tex: two explicit lines, box sizes to the wider one
kx, ky, kw, kh = node("keep", 5.40, 1.68 + YS,
     [L_("keep one honest +"), L_("one deceptive example")],
     fill=C["accentfill"], stroke=mix(C["accent"], 70), anchor="n", align="center")
arrow(5.40, fy0, 5.40, ky + kh + 0.05, "#8A8F98", 1.0)

circle(4.25, 0.52 + YS, 0.027, fill=C["truth"])
text(4.45, 0.52 + YS - 0.06, "honest", size=S * 0.86, fill=C["muted"])
circle(5.85, 0.52 + YS, 0.027, fill=C["decep"])
text(6.05, 0.52 + YS - 0.06, "deceptive", size=S * 0.86, fill=C["muted"])

arrow(7.78, 3.25, 8.18, 3.25, "#8A8F98", 1.1)

# ================================================================= stage 2 ==
badge(8.85, 5.98, "2")
text(9.15, 5.92, "Counterfactual Localization", size=M, weight="700")
text(9.15, 5.60, "fix a prefix and resample futures", size=S * 0.86, fill=C["muted"])

sent = [("s₁", "I need to play a card and claim it is a 7.", False),
        ("s₂", "I could just play the 7 honestly.", False),
        ("s₃", "Saving the 7 for later may be better.", True)]
low = None
for top, (tag, bodytxt, commit) in zip((4.85, 3.82, 2.79), sent):
    bx, by, bw, bh = node(f"sent{tag}", 8.55, top,
        [P_(f"{tag} {bodytxt}")], width=3.15,
        fill=C["accentfill"] if commit else "#ffffff",
        stroke=mix(C["accent"], 80) if commit else C["box"],
        sw=0.9 if commit else 0.8)
    low = by

frame("prefix", 8.42, low - 0.13, 3.61, (4.85 + 0.13) - (low - 0.13),
      mix(C["accent"], 80), dash="5 4")
lbl, lsz = "fixed prefix y₁:ₖ", S
lw = measure(lbl, lsz) / U + 0.18
add(f'<rect x="{X(10.225 - lw/2)}" y="{Y(5.06)}" width="{L(lw)}" height="{L(0.26)}" '
    f'fill="{C["panelbg"]}"/>')
text(10.225, 4.92, lbl, size=lsz, weight="700", anchor="middle", fill=mix(C["accent"], 85))

hub = (12.45, 2.35)
for cy, fill, stroke, title, quote, foot in [
    (4.58, C["truthfill"], mix(C["truth"], 65), "Generation 1",
     "“I should play the 7 and keep it simple.”", "(play 7, say “7”)"),
    (2.80, C["decepfill"], mix(C["decep"], 65), "Generation 2",
     "“I’ll hold onto the 7 and put down the king instead.”", "(play K, say “7”)"),
    (1.02, C["decepfill"], mix(C["decep"], 65), "Generation 3",
     "“Using the king here lets me save the 7 for later.”", "(play K, say “7”)")]:
    node(title, 12.90, cy,
         [L_(title, "b"), P_(quote), L_(foot, "muted", F)],
         width=3.25, fill=fill, stroke=stroke, anchor="w")
    arrow(hub[0], hub[1], 12.86, cy, "#9AA0A6", 0.95, dash="4 3.5")
circle(*hub, 0.017, fill="#8A8F98")

node("rate", 10.40, 0.90,
     [P_("counterfactual deception rate"), L_("p̂(k) = 2⁄3", "b", M)],
     width=2.45, fill=C["accentfill"], stroke=mix(C["accent"], 70),
     anchor="c", xsep=6, align="center")

arrow(16.78, 3.25, 17.18, 3.25, "#8A8F98", 1.1)

# ================================================================= stage 3 ==
badge(17.85, 5.98, "3")
text(18.15, 5.92, "Commitment Profile", size=M, weight="700")
text(18.15, 5.60, "p̂(k) across sentence boundaries", size=S * 0.86, fill=C["muted"])

arrow(17.85, 1.15, 17.85, 4.95, "#8A8F98", 1.1)
arrow(17.85, 1.15, 22.55, 1.15, "#8A8F98", 1.1)
add(f'<text x="{X(17.52)}" y="{Y(3.05)}" font-family="\'IBM Plex Sans\',sans-serif" '
    f'font-size="{S:.1f}" fill="{C["ink"]}" text-anchor="middle" '
    f'transform="rotate(-90 {X(17.52)} {Y(3.05)})">p̂(k)</text>')
text(20.20, 0.42, "sentence index", size=S, anchor="middle")
text(17.68, 1.15 - S * 0.34 / U, "0", size=S, anchor="end")
text(17.68, 4.78 - S * 0.34 / U, "1", size=S, anchor="end")

pts = [(18.55, 1.50), (19.45, 1.68), (20.35, 1.90), (21.25, 4.02), (22.15, 4.35)]
for i, (px, _) in enumerate(pts):
    line(px, 1.09, px, 1.21, "#B9BDC4", 0.8)
    text(px, 0.80, str(i + 1), size=S, anchor="middle")
d = " ".join(f"{'M' if i == 0 else 'L'}{X(px)} {Y(py)}" for i, (px, py) in enumerate(pts))
add(f'<path d="{d}" fill="none" stroke="{mix(C["decep"], 80)}" stroke-width="1.6" '
    f'stroke-linejoin="round" stroke-linecap="round"/>')
line(21.25, 1.15, 21.25, 4.00, mix(C["accent"], 85), 1.0, dash="5 4")
for px, py in pts:
    circle(px, py, 0.022, fill="#ffffff", stroke=mix(C["decep"], 80), sw=1.1)

node("jump", 19.75, 4.50,
     [L_("commitment"), L_("juncture:"), L_("Δp̂(k) is large", "b")],
     fill=C["accentfill"], stroke=mix(C["accent"], 70), anchor="c", align="center")

# ------------------------------------------------- geometry check -----------
# The two bugs this figure had - text spilling its box, and a box growing into
# the legend - are both detectable, so they are checked rather than eyeballed.
PANEL = {"p1": (0.0, 7.8), "p2": (8.2, 16.8), "p3": (17.2, 22.85)}
RESERVED = [("legend row", 4.10, 0.30, 2.30, 0.26)]   # x, y, w, h
problems = []
solid = [b for b in BOXES if not b[5]]
for i, (n1, x1, y1, w1, h1, _) in enumerate(solid):
    for n2, x2, y2, w2, h2, _ in solid[i + 1:]:
        if x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1:
            problems.append(f"boxes overlap: {n1} / {n2}")
    for rn, rx0, ry0, rw, rh in RESERVED:
        if x1 < rx0 + rw and rx0 < x1 + w1 and y1 < ry0 + rh and ry0 < y1 + h1:
            problems.append(f"{n1} overlaps {rn}")
    for fn, fx_, fy_, fw_, fh_, isf in BOXES:
        if not isf or fn != "pair": continue
        pass
    for pn, (px0, px1) in PANEL.items():
        if px0 - 0.01 <= x1 <= px1 and not (px0 <= x1 and x1 + w1 <= px1 + 0.01):
            problems.append(f"{n1} spills panel {pn}")
if problems:
    print("GEOMETRY CHECK FAILED:")
    for p_ in problems: print("   -", p_)
    sys.exit(1)
print(f"geometry check: {len(solid)} boxes, no overlaps, none spilling a panel")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {X(W)} {X(H)}" role="img" '
       f'aria-label="Deception mining and counterfactual localization">'
       f'<title>Figure 1 — Deception mining and counterfactual localization</title>'
       + "".join(out) + "</svg>")
open(sys.argv[1] if len(sys.argv) > 1 else "assets/figure1.svg", "w",
     encoding="utf-8").write(svg)
print(f"wrote {len(svg)} bytes, {len(out)} elements  "
      f"(\\scriptsize = {SCRIPT:.1f}u, \\small = {SMALL:.1f}u)")
