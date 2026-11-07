"""Architecture diagrams in the style of Raschka's "Build a Large Language Model (From Scratch)".

Data flows from the bottom (the input series) to the top (the forecast). White boxes are
layers, the dark box is the attention (or recurrent) mixer, the grey box the feed-forward
network, and orange marks the component that makes a model different from a plain
transformer. A blue panel is a block that repeats, with its repeat count on the left; a plus
sign is a residual (shortcut) connection.
"""
# --- cell: diagrams ---
from dataclasses import dataclass, field, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.textpath import TextPath

INK = "#222222"
FILL = {"plain": "#ffffff", "dark": "#4a4a4a", "mid": "#b8b8b8", "accent": "#f7c99b", "soft": "#e9e9f5",
        "io": "#ffffff"}
TEXT = {"dark": "#ffffff"}
OUTER, OUTER_EDGE = "#ececec", "#a6a6a6"
BLOCK, BLOCK_EDGE = "#cfe4f3", "#6f93b3"
BLOCK2, BLOCK2_EDGE = "#e3efd9", "#7f9f6a"     # a second repeated block type (e.g. a decoder)
W, H1, H2, GAP, R_PLUS = 2.7, 0.34, 0.50, 0.24, 0.085
SCALE = 0.95                 # inches per drawing unit (the figure size follows the drawing)
BOX_FONT, BOX_PAD = 7.6, 0.20  # font size inside boxes; horizontal room left free inside a box (drawing units)
# One font everywhere: DejaVu Sans ships with matplotlib, so a figure drawn on Colab, a Mac or Linux
# is the same picture, and the text-fitting below measures the font that is actually drawn.
FONT = "DejaVu Sans"
FONT_RC = {"font.family": [FONT], "font.sans-serif": [FONT], "mathtext.fontset": "dejavusans",
           "text.hinting": "default"}


@dataclass
class Box:
    text: str
    kind: str = "plain"
    name: str | None = None
    width: float | None = None
    math: str | None = None      # key of an expression in MATH (or a raw "$...$" string), drawn under the box


@dataclass
class Res:
    """Items wrapped by a residual connection that ends in a plus sign."""
    items: list


@dataclass
class Block:
    """A repeated group of items drawn on a coloured panel with 'N x' on its left."""
    items: list
    repeat: str = ""
    name: str | None = None
    second: bool = False
    label: str = ""


@dataclass
class Fan:
    """Several boxes side by side (experts, patch sizes, heads); inputs fan out, outputs merge."""
    labels: list
    kind: str = "plain"
    name: str | None = None
    caption: str = ""
    math: str | None = None
    width: float | None = None     # box width (default 1.0, for short labels)


@dataclass
class Note:
    target: str
    text: str
    side: str = "right"
    dy: float = 0.0


@dataclass
class Column:
    items: list
    title: str = ""
    inp: str = ""          # example input written under the column
    out: str = ""          # what comes out at the top
    container: bool = True


@dataclass
class Link:
    src: str
    dst: str
    text: str = ""


@dataclass
class Spec:
    key: str
    title: str
    columns: list
    notes: list = field(default_factory=list)
    links: list = field(default_factory=list)
    caption: str = ""
    legend: bool = True


def _h(text):
    lines = text.count("\n") + 1
    return H1 if lines == 1 else H2 + 0.16 * (lines - 2)


def _width(text, size, weight="normal"):
    """Width of ``text`` (its longest line) in drawing units at font ``size``."""
    prop = FontProperties(family=FONT, weight=weight)
    return max(TextPath((0, 0), line or " ", size=size, prop=prop).get_extents().width
               for line in text.split("\n")) / (72 * SCALE)


def _fit(text, w, size=BOX_FONT, weight="normal"):
    """Make ``text`` fit a box of width ``w``: break it onto two lines, then shrink the font if still needed."""
    room = w - BOX_PAD
    if _width(text, size, weight) <= room:
        return text, size
    if "\n" in text:                              # keep the author's line breaks: split only the long lines
        lines = [ln if _width(ln, size, weight) <= room else _wrap(ln.split(), 2, size, weight)[0]
                 for ln in text.split("\n")]
        s = "\n".join(lines)
        if s.count("\n") <= 2 and _width(s, size, weight) <= room:
            return s, size
    words = text.replace("\n", " ").split()
    best = None
    for n_lines in (2, 3):                       # the fewest lines that fit, each as short as possible
        best = _wrap(words, n_lines, size, weight) or best
        if best and best[1] <= room:
            return best[0], size
    text = best[0] if best else text
    while size > 5.6 and _width(text, size, weight) > room:
        size = round(size - 0.2, 1)
    return text, size


def _wrap(words, n_lines, size, weight):
    """(text, width) of the narrowest way to put ``words`` on ``n_lines`` lines, or None if too few words."""
    best = None
    for cut in _cuts(len(words), n_lines):
        s = "\n".join(" ".join(words[a:b]) for a, b in zip((0,) + cut, cut + (len(words),)))
        wd = _width(s, size, weight)
        if best is None or wd < best[1]:
            best = (s, wd)
    return best


def _cuts(n, k):
    """Every way to cut ``n`` words into ``k`` non-empty lines (as the k - 1 cut positions)."""
    from itertools import combinations
    return list(combinations(range(1, n), k - 1))


CORNER = 0.1                 # radius of the rounded corners of bent connectors (drawing units)


def _rounded(xs, ys, r=CORNER, n=12):
    """Points of a polyline whose corners are rounded (like draw.io's 'rounded' connectors): each corner is
    replaced by a curve of radius up to ``r`` that leaves and joins the straight segments smoothly."""
    pts = list(zip(xs, ys))
    if len(pts) < 3:
        return list(xs), list(ys)
    out = [pts[0]]
    for (x0, y0), (x1, y1), (x2, y2) in zip(pts, pts[1:], pts[2:]):
        d1, d2 = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if d1 == 0 or d2 == 0:
            out.append((x1, y1))
            continue
        rr = min(r, d1 / 2, d2 / 2)
        ax_, ay_ = x1 - (x1 - x0) / d1 * rr, y1 - (y1 - y0) / d1 * rr      # where the curve leaves the segment
        bx_, by_ = x1 + (x2 - x1) / d2 * rr, y1 + (y2 - y1) / d2 * rr      # where it joins the next one
        for i in range(n + 1):
            t = i / n                                                      # quadratic Bezier, corner as control
            out.append(((1 - t) ** 2 * ax_ + 2 * (1 - t) * t * x1 + t ** 2 * bx_,
                        (1 - t) ** 2 * ay_ + 2 * (1 - t) * t * y1 + t ** 2 * by_))
    out.append(pts[-1])
    return [p[0] for p in out], [p[1] for p in out]


class _Canvas:
    def __init__(self):
        self.cmds, self.pos = [], {}

    # every draw call is stored with a z-order and executed later on the real axes
    def add(self, z, fn):
        self.cmds.append((z, fn))

    def box(self, cx, y, b: Box):
        w = b.width or W
        weight = "bold" if b.kind == "dark" else "normal"
        text, size = _fit(b.text, w, weight=weight)
        h = _h(text)
        x0 = cx - w / 2
        fill, color = FILL.get(b.kind, "#ffffff"), TEXT.get(b.kind, INK)
        self.add(5, lambda ax: ax.add_patch(FancyBboxPatch((x0, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                                           fc=fill, ec=INK, lw=0.8, gid="box")))
        self.add(6, lambda ax: ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=size, color=color,
                                       weight=weight, linespacing=1.15, gid="boxtext"))
        if b.name:
            self.pos[b.name] = (x0, y, w, h)
        return y + h

    def arrow(self, x0, y0, x1, y1, z=4, style="-|>", ls="-", rad=0.0, lw=0.9):
        self.add(z, lambda ax: ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
                                                            color=INK, lw=lw, ls=ls,
                                                            connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0)))

    def line(self, xs, ys, z=4):
        xs, ys = _rounded(xs, ys)
        self.add(z, lambda ax: ax.plot(xs, ys, color=INK, lw=0.9, solid_capstyle="butt", solid_joinstyle="round"))

    def plus(self, cx, y):
        self.add(5, lambda ax: ax.add_patch(Circle((cx, y + R_PLUS), R_PLUS, fc="#ffffff", ec=INK, lw=0.8)))
        self.add(6, lambda ax: ax.plot([cx - R_PLUS * 0.6, cx + R_PLUS * 0.6], [y + R_PLUS] * 2, color=INK, lw=0.8))
        self.add(6, lambda ax: ax.plot([cx, cx], [y + R_PLUS * 0.4, y + R_PLUS * 1.6], color=INK, lw=0.8))
        return y + 2 * R_PLUS

    def panel(self, x0, y0, x1, y1, fc, ec, z=1, label="", label_side="left"):
        self.add(z, lambda ax: ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                                           boxstyle="round,pad=0,rounding_size=0.18",
                                                           fc=fc, ec=ec, lw=0.9, gid="panel")))
        if label:
            self.add(6, lambda ax: ax.text(x0 + 0.12, y1 - 0.1, label, ha="left", va="top", fontsize=8.2,
                                           weight="bold", color=INK, gid="title"))


def _place(cv, it, cx, y, depth=0):
    """Draw one item whose bottom edge is at y; returns its top edge (depth > 0: inside a block)."""
    if isinstance(it, Box):
        return cv.box(cx, y, it)
    if isinstance(it, Res):
        branch = y - GAP / 2
        top = _seq(cv, it.items, cx, y, first_arrow=False, depth=depth)
        cv.arrow(cx, top, cx, top + GAP)
        plus_y = top + GAP
        xr = cx + W / 2 + 0.22
        cv.line([cx, xr, xr], [branch, branch, plus_y + R_PLUS])
        cv.arrow(xr, plus_y + R_PLUS, cx + R_PLUS, plus_y + R_PLUS)
        return cv.plus(cx, plus_y)
    if isinstance(it, Block):
        inner = depth > 0
        x0, x1 = (cx - W / 2 - 0.22, cx + W / 2 + 0.33) if inner else (cx - W / 2 - 0.38, cx + W / 2 + 0.45)
        label = (f"{it.repeat} \u00d7  " if inner and it.repeat else "") + it.label
        top = _seq(cv, it.items, cx, y + 0.24 - GAP, first_arrow=True, depth=depth + 1)
        cv.line([cx, cx], [top, top + 0.2])
        y1 = top + (0.42 if label else 0.2)
        if label:
            cv.line([cx, cx], [top + 0.2, y1])
        fc, ec = (BLOCK2, BLOCK2_EDGE) if it.second else (BLOCK, BLOCK_EDGE)
        cv.panel(x0, y, x1, y1, fc, ec, z=2 + depth)
        if label:
            cv.add(6, lambda ax: ax.text(x0 + 0.1, y1 - 0.06, label, ha="left", va="top", fontsize=7.2,
                                         style="italic", weight="bold" if inner else "normal", color=INK,
                                         gid="label"))
        if it.repeat and not inner:
            cv.add(6, lambda ax: ax.text(x0 - 0.32, y + 0.06, f"{it.repeat} \u00d7", ha="right", va="bottom",
                                         fontsize=10, color=INK, gid="repeat"))
            cv.add(6, lambda ax: ax.text(x0 - 0.04, y - 0.02, "{", ha="right", va="bottom", fontsize=20,
                                         color=INK, family="DejaVu Serif", gid="brace"))
        if it.name:
            cv.pos[it.name] = (x0, y, x1 - x0, y1 - y)
        return y1
    if isinstance(it, Fan):
        k, gap = len(it.labels), 0.08
        wk = (W - gap * (k - 1)) / k
        labels = [_fit(t, wk, weight="bold" if it.kind == "dark" else "normal")[0] for t in it.labels]
        h = max(_h(t) for t in labels)
        xs = [cx - W / 2 + wk / 2 + j * (wk + gap) for j in range(k)]
        cv.line([xs[0], xs[-1]], [y - 0.07, y - 0.07])
        for x, t in zip(xs, labels):
            cv.arrow(x, y - 0.07, x, y)
            cv.box(x, y, Box(t, it.kind, width=wk))
            cv.line([x, x], [y + h, y + h + 0.07])
        cv.line([xs[0], xs[-1]], [y + h + 0.07, y + h + 0.07])
        if it.caption:
            cv.add(6, lambda ax: ax.text(cx - W / 2 - 0.08, y + h / 2, it.caption, ha="right", va="center",
                                         fontsize=6.8, style="italic", gid="caption"))
        if it.name:
            cv.pos[it.name] = (cx - W / 2, y, W, h)
        return y + h + 0.07
    raise TypeError(it)


def _seq(cv, items, cx, y, first_arrow=True, depth=0):
    """Place items one above the other, with an arrow into each (except the first, if asked)."""
    for i, it in enumerate(items):
        if first_arrow or i > 0:
            cv.arrow(cx, y, cx, y + GAP)
            y += GAP
        y = _place(cv, it, cx, y, depth)
    return y


def _pinned_font(fn):
    """Run a drawing function with the figure font settings (see FONT_RC)."""
    from functools import wraps

    @wraps(fn)
    def run(*args, **kwargs):
        with plt.rc_context(FONT_RC):
            return fn(*args, **kwargs)
    return run


@_pinned_font
def draw(spec: Spec, path=None, show=True):
    """Draw a Spec; returns the matplotlib figure (saved to ``path`` when given)."""
    cv = _Canvas()
    col_w = W + 1.6
    tops = []
    for j, col in enumerate(spec.columns):
        cx = j * (col_w + 0.5)
        y = 0.0
        # input label and arrow into the model
        if col.inp:
            cv.add(6, lambda ax, cx=cx, t=col.inp: ax.text(cx, -0.08, t, ha="center", va="top", fontsize=7.8,
                                                            family="monospace", color=INK, gid="input"))
        cv.arrow(cx, y, cx, y + 0.42)
        y_start = y + 0.42
        y_top = _seq(cv, col.items, cx, y_start + 0.12, first_arrow=True)
        cv.line([cx, cx], [y_start, y_start + 0.12])
        y_end = y_top + 0.42 if col.title else y_top + 0.25
        if col.container:
            cv.panel(cx - W / 2 - 0.62, y_start, cx + W / 2 + 0.68, y_end, OUTER, OUTER_EDGE, z=0, label=col.title)
        if col.out:     # an encoder that only feeds another column (through a Link) has no output arrow
            cv.arrow(cx, y_top, cx, y_end + 0.42)
            cv.add(6, lambda ax, cx=cx, y=y_end + 0.47, t=col.out: ax.text(cx, y, t, ha="center", va="bottom",
                                                                            fontsize=7.8, style="italic", color=INK,
                                                                            gid="output"))
        tops.append(y_end + 0.85)
    x_min = -W / 2 - 0.9
    x_max = (len(spec.columns) - 1) * (col_w + 0.5) + W / 2 + 0.9
    # cross-column links (e.g. encoder output into decoder cross-attention)
    for ln in spec.links:
        sx, sy, sw, sh = cv.pos[ln.src]
        dx, dy, dw, dh = cv.pos[ln.dst]
        cv.arrow(sx + sw, sy + sh / 2, dx, dy + dh / 2, z=7, rad=-0.15, lw=1.0)
        if ln.text:
            cv.add(7, lambda ax, x=(sx + sw + dx) / 2, y=(sy + dy) / 2 + sh, t=ln.text: ax.text(
                x, y, t, ha="center", va="bottom", fontsize=7, style="italic", gid="link"))
    # notes: bold text in the margins with dashed curved leaders
    margin_l, margin_r = x_min - 0.25, x_max + 0.25
    for n in spec.notes:
        x0, y0, w, h = cv.pos[n.target]
        ty = y0 + h / 2 + n.dy
        if n.side == "left":
            tx, ha, ax_pt = margin_l, "right", (x0, y0 + h / 2)
        else:
            tx, ha, ax_pt = margin_r, "left", (x0 + w, y0 + h / 2)
        cv.add(8, lambda ax, tx=tx, ty=ty, t=n.text, ha=ha: ax.text(tx, ty, t, ha=ha, va="center", fontsize=7.4,
                                                                   weight="bold", color=INK, linespacing=1.2,
                                                                   gid="note"))
        start = (tx + (0.05 if ha == "left" else -0.05) * -1, ty)
        cv.arrow(start[0], start[1], ax_pt[0], ax_pt[1], z=8, ls=(0, (3, 2)), rad=0.25 if n.side == "right" else -0.25,
                 lw=0.7)
    y_max = max(tops) + 0.2
    y_min = -0.55
    left_pad = 2.6 if any(n.side == "left" for n in spec.notes) else 0.3
    right_pad = 2.6 if any(n.side == "right" for n in spec.notes) else 0.3
    xl, xr = x_min - left_pad, x_max + right_pad
    if spec.legend:     # never cut the legend off on a narrow figure
        xr = max(xr, xl + 0.3 + _legend_width(_uses_second(spec)))
    scale = SCALE
    fig = plt.figure(figsize=((xr - xl) * scale, (y_max - y_min + (0.55 if spec.legend else 0.2)) * scale))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(xl, xr)
    ax.set_ylim(y_min - (0.45 if spec.legend else 0.1), y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    for z, fn in sorted(cv.cmds, key=lambda c: c[0]):
        fn(ax)
    ax.text(xl + 0.1, y_max - 0.05, spec.title, ha="left", va="top", fontsize=11, weight="bold", color=INK, gid="title")
    if spec.legend:
        _legend(ax, xl + 0.1, y_min - 0.32, second=_uses_second(spec))
    if path is not None:
        fig.savefig(path, dpi=200, facecolor="#ffffff")
    if not show:
        plt.close(fig)
    return fig


def _uses_second(spec):
    def walk(items):
        for it in items:
            if isinstance(it, Block) and (it.second or walk(it.items)):
                return True
            if isinstance(it, Res) and walk(it.items):
                return True
        return False
    return any(walk(c.items) for c in spec.columns)


def _legend_items(second=False):
    items = [(FILL["plain"], INK, "layer"), (FILL["dark"], INK, "attention or recurrent mixer"),
             (FILL["mid"], INK, "feed-forward"), (FILL["accent"], INK, "what is new in this model"),
             (BLOCK, BLOCK_EDGE, "repeated block")]
    if second:
        items.append((BLOCK2, BLOCK2_EDGE, "second repeated block"))
    return items


def _legend_width(second=False):
    return sum(0.62 + 0.058 * len(label) for _, _, label in _legend_items(second))


def _legend(ax, x, y, second=False):
    items = _legend_items(second)
    for fc, ec, label in items:
        ax.add_patch(FancyBboxPatch((x, y - 0.07), 0.28, 0.16, boxstyle="round,pad=0,rounding_size=0.03",
                                    fc=fc, ec=ec, lw=0.6))
        ax.text(x + 0.36, y, label, ha="left", va="center", fontsize=6.8, color=INK, gid="legend")
        x += 0.62 + 0.058 * len(label)


# ============================================================================ horizontal layout (book figures)
# Data flows from left (the input series) to right (the forecast). The main row shows the pipeline with each
# repeated block as one highlighted box; the rows underneath open each block up, again from left to right, with
# its residual connections drawn as arcs over the layers. Key expressions are written under the layers that
# compute them, and numbered orange circles point to the key aspects listed under the figure.

HS = 0.62                         # inches per drawing unit
HF, MF, KF = 7.0, 6.6, 7.0        # font sizes: boxes, maths, key-aspect list
HBW, HAG, HPAD = 1.42, 0.30, 0.13  # default box width, arrow length, inner padding
LINE = HF / 72 / HS * 1.18        # height of one text line in drawing units
ACCENT = "#e07b39"


from functools import lru_cache


@lru_cache(maxsize=None)
def _line_w(line, size, weight):
    """Width of one line in points: the widest of the printed (vector) width and the width on screen, where
    glyphs snap to whole pixels (notebooks show figures at 100-110 dpi), so text fits in print and on screen."""
    prop = FontProperties(family=FONT, weight=weight, size=size)
    widths = [TextPath((0, 0), line or " ", size=size, prop=prop).get_extents().width]
    with plt.rc_context(FONT_RC):
        for dpi in (100, 110):
            px = _SCREEN.setdefault(dpi, RendererAgg(10, 10, dpi)).get_text_width_height_descent(
                line or " ", prop, ismath="$" in line)[0]
            widths.append(px * 72 / dpi)
    return max(widths)


_SCREEN = {}


def _hw(text, size, weight="normal"):
    return max(_line_w(ln, size, weight) for ln in text.split("\n")) / (72 * HS)


def _glue(words):
    """Keep short connecting pieces with their neighbours, so that no line ends in '=', '×', '→', '(' or a
    one-letter word: 'P = 16', '64 × 9', '→ 96 steps' and 'a softmax' stay on one line."""
    out = []
    for w_ in words:
        core = "".join(ch for ch in w_ if ch.isalnum())
        if out and (out[-1].endswith(("=", "×", "→", "(")) or out[-1].strip("(").lower() in ("a", "")
                    or (out[-1].startswith("(") and len(out[-1]) <= 2) or w_ in ("=", "×")
                    or (out[-1].startswith("[") and "]" not in out[-1])          # '[context patches]'
                    or (len(core) == 1 and core.isalpha() and len(w_) > 1 and core.lower() != "a")):  # 'to H)'
            out[-1] = out[-1] + " " + w_
        else:
            out.append(w_)
    return out


def _greedy(words, room, size, weight):
    """Greedy line filling: as many words per line as fit in ``room``."""
    lines, cur = [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if cur and _hw(trial, size, weight) > room:
            lines.append(cur)
            cur = w_
        else:
            cur = trial
    lines.append(cur)
    return "\n".join(lines)


def _hfit(text, w, size=HF, weight="normal", max_lines=4):
    """Fit ``text`` into width ``w`` (drawing units): fewest lines that fit, then a smaller font if needed."""
    room = w - 2 * HPAD
    if "\n" in text and all(_hw(ln, size, weight) <= room for ln in text.split("\n")):
        return text, size
    words = _glue(text.replace("\n", " ").split())
    if len(words) > 12 or max_lines > 4:              # long text: greedy filling is good enough
        s = _greedy(words, room, size, weight)
        while s.count("\n") + 1 > max_lines and size > 5.4:
            size = round(size - 0.2, 1)
            s = _greedy(words, room, size, weight)
        return s, size
    best = None
    for n in range(1, max_lines + 1):
        if n > len(words):
            break
        best = None
        for cut in _cuts(len(words), n):
            s = "\n".join(" ".join(words[a:b]) for a, b in zip((0,) + cut, cut + (len(words),)))
            wd = _hw(s, size, weight)
            if best is None or wd < best[1]:
                best = (s, wd)
        if best[1] <= room:
            return best[0], size
    s = best[0]
    while size > 5.4 and _hw(s, size, weight) > room:
        size = round(size - 0.2, 1)
    return s, size


@lru_cache(maxsize=None)
def _box_dims(text, kind="plain", width=None):
    """Width, height, fitted text and font size of a box; wide labels get a wider box."""
    weight = "bold" if kind == "dark" else "normal"
    w = width or HBW
    if width is None and "\n" not in text and _hw(text, HF, weight) + 2 * HPAD + 0.12 < HBW:
        w = max(0.9, _hw(text, HF, weight) + 2 * HPAD + 0.12)          # short label: a narrower box
    s, size = _hfit(text, w, weight=weight)
    if width is None and s.count("\n") >= 2 and min(len(ln) for ln in s.split("\n")) <= 6:
        for w2 in (w + 0.15, w + 0.3, w + 0.45):           # a stray short line ('1368)'): a little wider instead
            s2, size2 = _hfit(text, w2, weight=weight)
            if s2.count("\n") < s.count("\n") and size2 >= size:
                w, s, size = w2, s2, size2
                break
    if (s.count("\n") >= 3 or size < HF) and width is None:      # too tall or shrunk: try a wider box
        fits = lambda s2, size2, w2: _hw(s2, size2, weight) <= w2 - 2 * HPAD   # noqa: E731
        for lines, smin in ((3, HF), (4, HF), (4, 6.4), (4, 5.8)):
            for w2 in (1.7, 2.0, 2.2, 2.4, 2.6, 2.8):
                s2, size2 = _hfit(text, w2, weight=weight, max_lines=lines)
                if s2.count("\n") < lines and size2 >= smin and fits(s2, size2, w2):
                    w, s, size = w2, s2, size2
                    break
            else:
                continue
            break
    h = 0.18 + 1.05 * LINE * (s.count("\n") + 1) * size / HF      # rendered lines run a little taller than LINE
    return w, max(h, 0.42), s, size


class _HCanvas(_Canvas):
    def __init__(self, math_lookup):
        super().__init__()
        self.math_lookup = math_lookup
        self.used_math = []
        self.anchor = {}                   # name -> where its numbered callout goes, when not the top-right corner
        self.extent = [0, 0, 0, 0]         # x0, x1, y0, y1 of everything drawn

    def grow(self, x0, x1, y0, y1):
        e = self.extent
        self.extent = [min(e[0], x0), max(e[1], x1), min(e[2], y0), max(e[3], y1)]

    def harrow(self, x0, x1, y, z=4):
        self.arrow(x0, y, x1, y, z=z)

    def hbox(self, x, yc, b: Box):
        """Draw box ``b`` with its left edge at x, centred on yc; returns its right edge."""
        w, h, s, size = _box_dims(b.text, b.kind, b.width)
        weight = "bold" if b.kind == "dark" else "normal"
        fill, color = FILL.get(b.kind, "#ffffff"), TEXT.get(b.kind, INK)
        y0 = yc - h / 2
        self.add(5, lambda ax: ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                                           fc=fill, ec=INK, lw=0.8, gid="box")))
        self.add(6, lambda ax: ax.text(x + w / 2, yc, s, ha="center", va="center", fontsize=size, color=color,
                                       weight=weight, linespacing=1.12, gid="boxtext"))
        if b.name:
            self.pos[b.name] = (x, y0, w, h)
        self.grow(x, x + w, y0, y0 + h)
        return x + w

    def math_under(self, x0, x1, ybottom, key):
        expr = self.math_lookup(key)
        if not expr:
            return
        if key not in self.used_math:
            self.used_math.append(key)
        xc = (x0 + x1) / 2
        self.add(6, lambda ax: ax.text(xc, ybottom - 0.08, expr, ha="center", va="top", fontsize=MF, color="#1f4e79",
                                       gid="math"))
        w = _hw(expr, MF)
        tall = any(t in expr for t in ("\\frac", "\\sqrt", "\\sum", "\\prod"))
        self.grow(xc - w / 2, xc + w / 2, ybottom - (0.52 if tall else 0.36), ybottom)


FAN_CAP = 0.26        # room for a caption above a Fan


def _fan_width(it):
    """Width of the boxes of a Fan: wide enough for its longest label on one line."""
    weight = "bold" if it.kind == "dark" else "normal"
    return max([it.width or 1.0] + [_hw(t, HF, weight) + 2 * HPAD + 0.1 for t in it.labels])


def _item_size(it):
    """(width, height) of an item laid out horizontally, without drawing it."""
    if isinstance(it, Box):
        w, h, _, _ = _box_dims(it.text, it.kind, it.width)
        return w, h
    if isinstance(it, Res):
        ws, hs = zip(*(_item_size(x) for x in it.items))
        return sum(ws) + HAG * len(ws) + 2 * R_PLUS + 0.12, max(hs) + 0.36
    if isinstance(it, Block):
        ws, hs = zip(*(_item_size(x) for x in it.items))
        return sum(ws) + HAG * (len(ws) + 1) + 0.1, max(hs) + 0.62
    if isinstance(it, Fan):
        w = _fan_width(it) + 0.3
        if it.caption:
            w = max(w, _hw(it.caption, 6.4) + 0.2)
        return w, len(it.labels) * 0.34 + 0.06 + (FAN_CAP if it.caption else 0)
    raise TypeError(it)


def _math_width(cv, it):
    if isinstance(it, Box) and it.math:
        expr = cv.math_lookup(it.math)
        return _hw(expr, MF) if expr else 0
    return 0


def _hplace(cv, it, x, yc):
    """Draw one item of a block's detail row with its left edge at x; returns its right edge."""
    if isinstance(it, Box):
        w, h = _item_size(it)
        mw = _math_width(cv, it)
        if mw > w:                              # maths wider than the box: widen the box so the arrows still meet it
            it = replace(it, width=mw)
            w, h = _item_size(it)
        x1 = cv.hbox(x, yc, it)
        if it.math:
            cv.math_under(x, x1, yc - h / 2, it.math)
        return x1
    if isinstance(it, Res):
        _, h = _item_size(it)
        xs = x - HAG * 0.55                     # the shortcut leaves the main line before the first layer
        top = yc + (h - 0.36) / 2 + 0.2
        cur = x
        for i, sub in enumerate(it.items):
            if i:
                cv.harrow(cur, cur + HAG, yc)
                cur += HAG
            cur = _hplace(cv, sub, cur, yc)
        cv.harrow(cur, cur + HAG, yc)
        px = cur + HAG + R_PLUS
        cv.line([xs, xs, px, px], [yc, top, top, yc + R_PLUS])
        cv.add(5, lambda ax: ax.add_patch(Circle((px, yc), R_PLUS, fc="#ffffff", ec=INK, lw=0.8)))
        cv.add(6, lambda ax: ax.plot([px - R_PLUS * 0.6, px + R_PLUS * 0.6], [yc, yc], color=INK, lw=0.8))
        cv.add(6, lambda ax: ax.plot([px, px], [yc - R_PLUS * 0.6, yc + R_PLUS * 0.6], color=INK, lw=0.8))
        cv.grow(xs, px + R_PLUS, yc - h / 2, top)
        return px + R_PLUS
    if isinstance(it, Block):
        w, h = _item_size(it)
        fc, ec = (BLOCK2, BLOCK2_EDGE) if it.second else (BLOCK, BLOCK_EDGE)
        y0, y1 = yc - h / 2 + 0.06, yc + h / 2
        cur = x + 0.05
        cv.harrow(cur, cur + HAG, yc)
        cur += HAG
        for i, sub in enumerate(it.items):
            if i:
                cv.harrow(cur, cur + HAG, yc)
                cur += HAG
            cur = _hplace(cv, sub, cur, yc)
        cv.harrow(cur, cur + HAG * 0.8, yc)
        x1 = cur + HAG * 0.8 + 0.05
        cv.panel(x, y0, x1, y1, fc, ec, z=3)
        label = (f"{it.repeat} × " if it.repeat else "") + (it.label or "")
        if label.strip():
            cv.add(6, lambda ax: ax.text(x + 0.08, y1 - 0.05, label.strip(), ha="left", va="top", fontsize=6.8,
                                         style="italic", weight="bold", color=INK, gid="label"))
        if it.name:
            cv.pos[it.name] = (x, y0, x1 - x, y1 - y0)
        cv.grow(x, x1, y0, y1)
        return x1
    if isinstance(it, Fan):
        w, h = _item_size(it)
        k = len(it.labels)
        bw = _fan_width(it)
        top = yc + h / 2 - 0.03 - (FAN_CAP if it.caption else 0)
        xl = x + (w - bw) / 2
        xr = xl + bw
        for j, lab in enumerate(it.labels):
            cy = top - 0.17 - j * 0.34
            cv.hbox(xl, cy, Box(lab, it.kind, width=bw))
            cv.line([x, xl], [yc, cy])
            cv.line([xr, x + w], [cy, yc])
        if it.caption:
            cv.add(6, lambda ax: ax.text(x + w / 2, top + 0.06, it.caption, ha="center", va="bottom", fontsize=6.4,
                                         style="italic", gid="caption"))
        if it.name:
            cv.pos[it.name] = (x, yc - h / 2, w, h)
            if it.caption:                       # a callout goes after the caption, not on top of it
                cv.anchor[it.name] = (x + w / 2 + _hw(it.caption, 6.4) / 2 + 0.17, top + 0.16)
        if it.math:
            cv.math_under(x, x + w, yc - h / 2, it.math)
        cv.grow(x, x + w, yc - h / 2, top + 0.25)
        return x + w
    raise TypeError(it)


def _series(n=120, seed=0, shift=0.0):
    import numpy as np
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 40 + shift) + 0.012 * t + 0.25 * rng.standard_normal(n).cumsum() / 4


def _label_depth(text, width):
    """How far an italic input/output caption reaches below its picture (drawing units), with a small margin."""
    if not text:
        return 0.2
    s, size = _hfit(text, width, size=6.4, max_lines=3)
    return 0.06 + (s.count("\n") + 1) * size / 72 / HS * 1.32 + 0.08


def _input_slot(cv, x, yc, col, multi):
    """A small time-series picture of the input (several stacked series when the model reads several rows)."""
    w, h = 1.55, 0.95
    y0 = yc - h / 2
    cv.add(5, lambda ax: ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                                     fc="#ffffff", ec="#9a9a9a", lw=0.7)))
    k = 3 if multi else 1
    colors = ["#222222", "#4f81bd", "#9bbb59"]
    for j in range(k):
        s = _series(seed=j + 1, shift=j)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        band = (h - 0.16) / k
        by = y0 + 0.08 + (k - 1 - j) * band
        xs = [x + 0.08 + (w - 0.16) * i / (len(s) - 1) for i in range(len(s))]
        ys = [by + 0.06 + v * (band - 0.12) for v in s]
        cv.add(6, lambda ax, xs=xs, ys=ys, c=colors[j]: ax.plot(xs, ys, color=c, lw=0.8))
    if multi:
        cov = "covariate" in (col.inp or "").lower()
        cv.add(6, lambda ax: ax.text(x + w + 0.03, y0 + h - 0.05, "target" if cov else "variates", fontsize=5.6,
                                     va="top", color="#222222",
                                     gid="io_tag"))
        if cov:
            cv.add(6, lambda ax: ax.text(x + w + 0.03, y0 + 0.05, "covariates", fontsize=5.6, va="bottom",
                                         color="#4f81bd", gid="io_tag"))
    if col.inp:
        s, size = _hfit(col.inp, 1.95, size=6.4, max_lines=3)
        cv.add(6, lambda ax: ax.text(x + w / 2, y0 - 0.06, s, ha="center", va="top", fontsize=size, style="italic",
                                     color=INK, linespacing=1.1, gid="input"))
    below = _label_depth(col.inp, 1.95)
    cv.grow(x - 0.2, x + w + (0.55 if multi else 0), y0 - below, y0 + h)
    return x + w, x + w + (0.5 if multi else 0)        # the picture's edge, and where the next item may start


def _output_slot(cv, x, yc, col):
    import numpy as np
    w, h = 1.55, 0.95
    y0 = yc - h / 2
    cv.add(5, lambda ax: ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                                     fc="#ffffff", ec="#9a9a9a", lw=0.7)))
    s = _series(n=160, seed=1)
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    past, fut = s[:110], s[110:]
    tx = lambda i: x + 0.08 + (w - 0.16) * i / 159
    ty = lambda v: y0 + 0.1 + v * (h - 0.2) * 0.8
    cv.add(6, lambda ax: ax.plot([tx(i) for i in range(110)], [ty(v) for v in past], color="#9a9a9a", lw=0.8))
    med = past[-1] + (fut - fut[0]) * 0.6
    fx = [tx(i) for i in range(109, 160)]
    fm = [ty(v) for v in np.r_[past[-1], med]]
    spread = np.linspace(0, 0.28, len(fm))
    out = (col.out or "").lower()
    if "sample" in out:
        rng = np.random.default_rng(3)
        for k in range(4):
            path = np.r_[0, rng.standard_normal(len(fm) - 1).cumsum()] * 0.035
            cv.add(6, lambda ax, p=path: ax.plot(fx, [a + b for a, b in zip(fm, p)], color=ACCENT, lw=0.5, alpha=0.7))
    elif "quantile" in out or "distribution" in out:
        lo = [a - b * (h - 0.2) * 0.5 for a, b in zip(fm, spread)]
        hi = [a + b * (h - 0.2) * 0.5 for a, b in zip(fm, spread)]
        cv.add(5, lambda ax: ax.fill_between(fx, lo, hi, color=ACCENT, alpha=0.25, lw=0))
    cv.add(6, lambda ax: ax.plot(fx, fm, color=ACCENT, lw=1.1))
    cv.add(6, lambda ax: ax.plot([tx(109)] * 2, [y0 + 0.06, y0 + h - 0.06], color="#bbbbbb", lw=0.6, ls="--"))
    if col.out:
        s2, size = _hfit(col.out, 2.0, size=6.4, max_lines=3)
        cv.add(6, lambda ax: ax.text(x + w / 2, y0 - 0.06, s2, ha="center", va="top", fontsize=size, style="italic",
                                     color=INK, linespacing=1.1, gid="output"))
    cv.grow(x - 0.2, x + w + 0.2, y0 - _label_depth(col.out, 2.0), y0 + h)
    return x + w


def _block_box(cv, x, yc, it, letter):
    """A repeated block in the main row: one highlighted box with its repeat count, opened up below."""
    name = _block_name(it)
    text = f"{name}\n× {it.repeat}" if it.repeat and it.repeat != "1" else name
    w, h = 1.45, 0.82
    fc, ec = (BLOCK2, BLOCK2_EDGE) if it.second else (BLOCK, BLOCK_EDGE)
    y0 = yc - h / 2
    cv.add(5, lambda ax: ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.1",
                                                     fc=fc, ec=ec, lw=1.6, gid="box")))
    s, size = _hfit(text, w, size=7.4, weight="bold")
    cv.add(6, lambda ax: ax.text(x + w / 2, yc + 0.02, s, ha="center", va="center", fontsize=size, weight="bold",
                                 color=INK, gid="boxtext"))
    cv.add(6, lambda ax: ax.text(x + w / 2, y0 - 0.05, f"details in {letter}", ha="center", va="top", fontsize=5.8,
                                 style="italic", color="#45627d", gid="tag"))
    if it.name:
        cv.pos[it.name] = (x, y0, w, h)
    cv.grow(x, x + w, y0, y0 + h)
    return x + w


def _main_row(cv, col, yc, blocks, multi_in):
    """Draw one column as a left-to-right row; top-level blocks become boxes collected in ``blocks``."""
    x = 0.0
    if col.inp and not col.inp.startswith("(") and "token" not in col.inp:
        edge, x = _input_slot(cv, x, yc, col, multi_in)
        cv.harrow(edge, x + HAG, yc)                     # the arrow leaves the picture's edge, between its tags
        x += HAG
    elif col.inp:
        s, size = _hfit(col.inp, 1.5, size=6.4, max_lines=3)
        cv.add(6, lambda ax: ax.text(0.75, yc, s, ha="center", va="center", fontsize=size, style="italic",
                                     gid="input"))
        cv.grow(0, 1.5, yc - 0.3, yc + 0.3)
        x = 1.5
        cv.harrow(min(x, 0.75 + _hw(s, size) / 2 + 0.1), x + HAG, yc)     # from just after the text
        x += HAG
    for i, it in enumerate(col.items):
        if i:
            cv.harrow(x, x + HAG, yc)
            x += HAG
        if isinstance(it, Block):
            letter = "ABCDEFG"[len(blocks)]
            blocks.append((it, letter))
            x = _block_box(cv, x, yc, it, letter)
        elif isinstance(it, Res):
            for j, sub in enumerate(it.items):
                if j:
                    cv.harrow(x, x + HAG, yc)
                    x += HAG
                x = _hplace(cv, sub, x, yc)
        else:
            x = _hplace(cv, it, x, yc)
    if col.out:
        cv.harrow(x, x + HAG, yc)
        x = _output_slot(cv, x + HAG, yc, col)
    return x


def _detail_row(cv, it, letter, yc, min_x1=0.0):
    """Open up a repeated block: its layers from left to right on the block's colour (the panel reaches at
    least ``min_x1``)."""
    fc, ec = (BLOCK2, BLOCK2_EDGE) if it.second else (BLOCK, BLOCK_EDGE)
    x = 0.35
    outer, cv.extent = cv.extent, [x, x, yc, yc]       # measure what this row draws
    cv.harrow(x, x + HAG, yc)
    cur = x + HAG
    for i, sub in enumerate(it.items):
        if i:
            cv.harrow(cur, cur + HAG, yc)
            cur += HAG
        cur = _hplace(cv, sub, cur, yc)
    cv.harrow(cur, cur + HAG, yc)
    x1 = max(cur + HAG + 0.1, min_x1)
    lo, hi = cv.extent[2], cv.extent[3]
    cv.extent = outer
    inner = set().union(*(_names(x) for x in it.items))
    for nm in _TARGETS & inner:                         # leave room for a numbered callout on a top corner
        if nm in cv.pos:
            hi = max(hi, cv.pos[nm][1] + cv.pos[nm][3] + 0.15)
    y0, y1 = lo - 0.14, hi + 0.36
    cv.panel(0.05, y0, x1, y1, fc, ec, z=1)
    name = _block_name(it)
    title = f"{letter} · inside the {name}" + (f" (repeated {it.repeat} ×)" if it.repeat and it.repeat != "1" else "")
    cv.add(6, lambda ax: ax.text(0.16, y1 - 0.06, title, ha="left", va="top", fontsize=7.4, weight="bold",
                                 color=INK, gid="title"))
    cv.grow(0.05, x1, y0, y1)
    return y0, y1


def _walk(it):
    """Every item inside ``it`` (depth first)."""
    for sub in getattr(it, "items", []):
        yield sub
        yield from _walk(sub)


def _direct(items):
    """Boxes of a row, looking inside residual groups but not inside nested blocks (they have their own room)."""
    for it in items:
        if isinstance(it, Res):
            yield from _direct(it.items)
        elif not isinstance(it, Block):
            yield it


def _block_name(it):
    """What a repeated block is called in the figure."""
    if it.label:
        return it.label
    texts = " ".join(getattr(m, "text", "") for m in _walk(it)).lower()
    kind = "transformer" if "attention" in texts else ("sLSTM" if "slstm" in texts else "")
    if it.second:
        return f"{kind} decoder block".strip()
    return f"{kind} block".strip()


def _row_extent(fn):
    """Vertical extent of what ``fn(canvas)`` draws, measured on a scratch canvas."""
    scratch = _HCanvas(lambda k: MATH_LOOKUP(k))
    fn(scratch)
    return scratch.extent


MATH_LOOKUP = lambda key: key if key and key.startswith("$") else None   # noqa: E731  (set by architectures)


_TARGETS = set()          # names that carry a numbered callout in the figure being drawn


@_pinned_font
def draw_h(spec: Spec, path=None, show=True):
    """Draw a Spec from left to right (see the comment above); returns the figure."""
    _TARGETS.clear()
    _TARGETS.update(n.target for n in spec.notes)
    cv = _HCanvas(lambda k: MATH_LOOKUP(k))
    rows, y = [], 0.0
    blocks = []
    multi = [("univariate" not in (c.inp or "").lower()
              and any(w in (c.inp or "").lower() for w in ("covariate", "variate", "rows", "targets")))
             for c in spec.columns]
    gaps = []
    # main rows (one per column), then one detail row per repeated block, each placed under the previous
    for j, col in enumerate(spec.columns):
        e = _row_extent(lambda c, col=col, m=multi[j]: _main_row(c, col, 0.0, [], m))
        yc = y - e[3]
        _main_row(cv, col, yc, blocks, multi[j])
        gap = 0.6 if spec.links and j < len(spec.columns) - 1 else 0.35
        gaps.append(yc + e[2] - gap / 2)          # the free line between this row and the next
        y = yc + e[2] - gap
    for k, (it, letter) in enumerate(blocks):
        linked = k == 0 and len(spec.columns) == 1 and it.name in cv.pos      # a dashed line down to the detail
        e = _row_extent(lambda c, it=it, letter=letter: _detail_row(c, it, letter, 0.0))
        reach = 0.0
        if linked:                  # a panel that ends a little before the block is widened to meet the line
            x = cv.pos[it.name][0] + cv.pos[it.name][2] / 2
            if e[1] - 0.25 <= x < e[1] + 1.5:
                reach = x + 0.4
                e = _row_extent(lambda c, it=it, letter=letter: _detail_row(c, it, letter, 0.0, reach))
        yc = y - e[3]
        py0, py1 = _detail_row(cv, it, letter, yc, reach)
        if linked:
            bx, by, bw, bh = cv.pos[it.name]
            x = bx + bw / 2                     # a panel that ends well before the block: turn left just above it
            xe = x if x < e[1] - 0.25 else e[1] - 0.6
            xs, ys = ([x, x], [by - 0.26, py1]) if xe == x else ([x, x, xe, xe], [by - 0.26, py1 + 0.15, py1 + 0.15, py1])
            xs, ys = _rounded(xs, ys)
            cv.add(2, lambda ax, xs=xs, ys=ys: ax.plot(xs, ys, color=BLOCK_EDGE, lw=0.9, ls=(0, (3, 2))))
        y = yc + e[2] - 0.3
    # cross-row links (encoder states into the decoder)
    for ln in spec.links:
        src = cv.pos.get(ln.src)
        dst = next((cv.pos[it.name] for it, _ in blocks if it.name and ln.dst in _names(it)), cv.pos.get(ln.dst))
        if src and dst:
            sx, sy, sw, sh = src
            dx, dy, dw, dh = dst
            ym = gaps[0]
            x_s, x_d = sx + sw / 2, dx + dw / 2
            yb = ym - CORNER - 0.02                            # the line turns down (rounded) before the arrow
            lx, ly = _rounded([x_s, x_s, x_d, x_d], [sy, ym, ym, yb])
            cv.add(7, lambda ax, lx=lx, ly=ly: ax.plot(lx, ly, color=INK, lw=1.0, solid_capstyle="butt"))
            cv.arrow(x_d, yb, x_d, dy + dh, z=7, lw=1.0)
            label = ln.text or "encoder states (read by cross-attention)"
            cv.add(7, lambda ax, x=(x_s + x_d) / 2, ym=ym, t=label: ax.text(
                x, ym + 0.04, t, ha="center", va="bottom", fontsize=6.2, style="italic", color=INK, gid="link"))
    # numbered callouts for the key aspects
    # number the key aspects in reading order: row by row from the top, left to right within a row
    notes = _reading_order([n for n in spec.notes if n.target in cv.pos], cv.pos)
    texts, taken = (_text_boxes(cv) if notes else []), []
    for i, n in enumerate(notes, 1):
        cx, cy = _callout_spot(cv.pos[n.target], cv.anchor.get(n.target), texts, taken)
        taken.append((cx, cy))
        cv.grow(cx - 0.15, cx + 0.15, cy - 0.15, cy + 0.15)
        cv.add(9, lambda ax, cx=cx, cy=cy: ax.add_patch(Circle((cx, cy), 0.13, fc=ACCENT, ec="#ffffff", lw=0.8)))
        cv.add(10, lambda ax, cx=cx, cy=cy, i=i: ax.text(cx, cy - 0.005, str(i), ha="center", va="center", fontsize=6.6,
                                                         weight="bold", color="#ffffff", gid="callout"))
    x0, x1, y_lo, y_hi = cv.extent
    x1 = max(x1, x0 + 7.5)
    # legend and key aspects under the drawing
    y_leg = y_lo - 0.3
    keys = [(" ".join(n.text.split()), i) for i, n in enumerate(notes, 1)]
    lines = []
    for txt, i in keys:
        s = _greedy(f"{i}. {txt}".split(), x1 - x0 - 0.4, KF, "normal")
        lines.append(s)
    key_h = sum(LINE * (s.count("\n") + 1) + 0.06 for s in lines) + (0.3 if lines else 0)
    used = _legend_used(spec)
    leg_rows = _legend_layout(x0, x1 - x0, used)[1] if spec.legend else 1
    key_h += (leg_rows - 1) * LEG_ROW
    fig_w = (x1 - x0 + 0.5) * HS
    fig_h = (y_hi - y_leg + 0.75 + key_h) * HS
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0 - 0.25, x1 + 0.25)
    ax.set_ylim(y_leg - 0.3 - key_h, y_hi + 0.45)
    ax.set_aspect("equal")
    ax.axis("off")
    for z, fn in sorted(cv.cmds, key=lambda c: c[0]):
        fn(ax)
    ax.text(x0, y_hi + 0.38, spec.title, ha="left", va="top", fontsize=10, weight="bold", color=INK, gid="title")
    if spec.legend:
        _legend_h(ax, x0, y_leg, x1 - x0, used)
    yk = y_leg - 0.32 - (leg_rows - 1) * LEG_ROW
    if lines:
        ax.text(x0, yk, "Key aspects", ha="left", va="top", fontsize=KF, weight="bold", color=ACCENT, gid="keys")
        yk -= LINE + 0.06
        for s in lines:
            ax.text(x0, yk, s, ha="left", va="top", fontsize=KF, color=INK, linespacing=1.12, gid="keys")
            yk -= LINE * (s.count("\n") + 1) + 0.06
    fig.used_math = cv.used_math
    if path is not None:
        fig.savefig(path, dpi=200, facecolor="#ffffff")
        if str(path).endswith(".png"):              # a vector copy for print, fonts embedded as TrueType
            with plt.rc_context({"pdf.fonttype": 42}):
                fig.savefig(Path(path).with_suffix(".pdf"), facecolor="#ffffff")
    if not show:
        plt.close(fig)
    return fig


def _text_boxes(cv):
    """Extents (drawing units) of every text drawn so far, from a scratch rendering at the final scale."""
    x0, x1, y0, y1 = cv.extent
    fig = plt.figure(figsize=((x1 - x0 + 1) * HS, (y1 - y0 + 1) * HS))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x0 - 0.5, x1 + 0.5)
    ax.set_ylim(y0 - 0.5, y1 + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for _, fn in sorted(cv.cmds, key=lambda c: c[0]):
        fn(ax)
    fig.canvas.draw()
    r, inv = fig.canvas.get_renderer(), ax.transData.inverted()
    out = []
    for t in ax.texts:
        if t.get_text().strip():
            e = t.get_window_extent(r)
            (a, b), (c, d) = inv.transform([(e.x0, e.y0), (e.x1, e.y1)])
            out.append((a, b, c, d))
    plt.close(fig)
    return out


def _callout_spot(box, preferred, texts, taken, r=0.16):
    """Where a numbered circle goes: the preferred spot or a corner of its box (top right first), the first
    one that covers no text and no other circle."""
    x0, y0, w, h = box
    spots = ([preferred] if preferred else []) + [
        (x0 + w + 0.06, y0 + h + 0.02), (x0 - 0.06, y0 + h + 0.02), (x0 + w + 0.06, y0 - 0.02),
        (x0 - 0.06, y0 - 0.02), (x0 + w + 0.2, y0 + h / 2), (x0 + w / 2, y0 + h + 0.2)]

    def free(c):
        cx, cy = c
        return (not any(a - r < cx < c2 + r and b - r < cy < d + r for a, b, c2, d in texts)
                and all((cx - tx) ** 2 + (cy - ty) ** 2 >= (2 * r) ** 2 for tx, ty in taken))
    return next((c for c in spots if free(c)), spots[0])


def _reading_order(notes, pos):
    """Notes in reading order: rows from top to bottom (items whose centres lie within 0.3 share a row),
    left to right inside a row."""
    cy = {n.target: pos[n.target][1] + pos[n.target][3] / 2 for n in notes}
    rows = []
    for n in sorted(notes, key=lambda n: -cy[n.target]):
        if rows and abs(cy[rows[-1][0].target] - cy[n.target]) < 0.3:
            rows[-1].append(n)
        else:
            rows.append([n])
    return [n for row in rows for n in sorted(row, key=lambda n: pos[n.target][0])]


def _names(it):
    out = {it.name} if getattr(it, "name", None) else set()
    for sub in getattr(it, "items", []):
        out |= _names(sub)
    return out


LEG_ROW = 0.3


def _legend_used(spec):
    """What the figure draws, so the legend lists only those entries."""
    used = set()

    def visit(items, in_block):
        for it in items:
            if isinstance(it, (Box, Fan)):
                used.add(it.kind)
            elif isinstance(it, Block):
                used.add("second" if it.second else "block")
                visit(it.items, True)
            elif isinstance(it, Res):
                if in_block:                    # the main row draws no plus sign; detail panels do
                    used.add("res")
                visit(it.items, in_block)

    for c in spec.columns:
        visit(c.items, False)
    return used


def _legend_layout(x, width, used):
    """Legend entries with their positions, wrapped into rows that fit ``width``; returns (entries, rows)."""
    every = [("plain", FILL["plain"], INK, "layer"), ("dark", FILL["dark"], INK, "attention or recurrent mixer"),
             ("mid", FILL["mid"], INK, "feed-forward"), ("accent", FILL["accent"], INK, "new in this model"),
             ("block", BLOCK, BLOCK_EDGE, "repeated block"),
             ("second", BLOCK2, BLOCK2_EDGE, "second repeated block"),
             ("res", None, None, "add the shortcut (residual)")]
    items = [(fc, ec, label) for key, fc, ec, label in every if key in used]
    out, cx, row = [], x, 0
    for fc, ec, label in items:
        step = 0.33 + _hw(label, 6.4) + 0.32
        if cx > x and cx + step - 0.32 > x + width:
            cx, row = x, row + 1
        out.append((cx, row, fc, ec, label))
        cx += step
    return out, row + 1


def _legend_h(ax, x, y, width, used):
    entries, _ = _legend_layout(x, width, used)
    for cx, row, fc, ec, label in entries:
        yy = y - row * LEG_ROW
        if fc is None:
            ax.add_patch(Circle((cx + 0.13, yy), R_PLUS, fc="#ffffff", ec=INK, lw=0.7))
            ax.plot([cx + 0.13 - R_PLUS * 0.6, cx + 0.13 + R_PLUS * 0.6], [yy, yy], color=INK, lw=0.7)
            ax.plot([cx + 0.13, cx + 0.13], [yy - R_PLUS * 0.6, yy + R_PLUS * 0.6], color=INK, lw=0.7)
        else:
            ax.add_patch(FancyBboxPatch((cx, yy - 0.07), 0.26, 0.15, boxstyle="round,pad=0,rounding_size=0.03",
                                        fc=fc, ec=ec, lw=0.6))
        ax.text(cx + 0.33, yy, label, ha="left", va="center", fontsize=6.4, color=INK, gid="legend")
