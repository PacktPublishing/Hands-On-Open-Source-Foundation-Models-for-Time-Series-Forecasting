"""Step-by-step architecture explainers: pastel numbered panels, plain one-line explanations.

Every figure is a column of numbered steps joined by small down arrows. A step has a bold heading, one or two
plain sentences and a picture made of a few widgets (rows of cells, white chips, a token grid, a small fan
chart). The figure draws itself in inches with y pointing down, so the layout does not depend on the dpi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FONT_DIR = Path(__file__).with_name("fonts")
FONT = "Liberation Sans"          # metric twin of Arial / Helvetica, SIL OFL 1.1, shipped in fonts/
for f in sorted(FONT_DIR.glob("*.ttf")):
    font_manager.fontManager.addfont(str(f))

RC = {"font.family": FONT, "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
      "mathtext.bf": f"{FONT}:bold", "mathtext.cal": f"{FONT}:italic", "mathtext.sf": FONT, "mathtext.tt": FONT,
      "mathtext.fallback": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
      # text extents depend on hinting; pinned so the layout is the same on Colab, a Mac and Linux
      "text.hinting": "no_hinting"}

INK, GREY, LIGHT = "#1a1a1a", "#555555", "#9a9a9a"
CHIP_EDGE = "#c9c9c9"
# (fill, border) pairs read off the reference pictures: yellow, green, orange, blue, lilac, rose
PALETTE = {"yellow": ("#fdf5d6", "#e2c24f"), "green": ("#e5f3ec", "#5aa985"), "orange": ("#fcebdc", "#e39b62"),
           "blue": ("#e4ecf8", "#5b83c6"), "lilac": ("#eee9f7", "#8d78c2"), "rose": ("#fbe6e6", "#cf6b6b")}
ACCENT = {"blue": "#2f67c4", "green": "#2e8a5f", "orange": "#d9762b", "red": "#c9463d", "grey": "#6b6b6b"}

W = 7.5          # figure width, inches
MARGIN = 0.28
PAD_X, PAD_TOP, PAD_BOTTOM = 0.22, 0.15, 0.17
ROUND = 0.12
GAP = 0.28       # vertical space between two steps (holds the arrow)

SIZE = {"title": 20, "sub": 11, "head": 14.5, "body": 11.2, "chip": 10.6, "small": 9.6, "foot": 12.5}
LEAD = 1.28      # line height as a multiple of the font size


def _pt(size):
    return size / 72.0


class _Measure:
    """Text widths in inches, from the Agg renderer."""

    def __init__(self):
        with matplotlib.rc_context(RC):     # not pyplot, so a notebook never shows this empty figure
            self.fig = Figure(figsize=(4, 4), dpi=200)
            self.r = FigureCanvasAgg(self.fig).get_renderer()
        self.cache = {}

    def width(self, s, size, weight="normal", style="normal"):
        key = (s, size, weight, style)
        if key not in self.cache:
            with matplotlib.rc_context(RC):
                t = self.fig.text(0, 0, s, size=size, weight=weight, style=style)
                self.cache[key] = t.get_window_extent(self.r).width / self.fig.dpi
                t.remove()
        return self.cache[key]


M = _Measure()


# ----------------------------------------------------------------------------------------------- rich text
def _runs(text, weight="normal", color=INK):
    """Rich text as runs [(string, weight, color)]; '**x**' marks bold inside a plain string."""
    if isinstance(text, list):
        return text
    out, bold = [], False
    for i, part in enumerate(text.split("**")):
        if part:
            out.append((part, "bold" if bold else weight, color))
        bold = not bold
    return out


def _space_w(size, weight):
    return M.width("a a", size, weight) - M.width("aa", size, weight)


def _wrap(runs, width, size):
    """Greedy line breaking of rich runs; returns lines of runs and the width of each line.

    Lines break only at spaces, so a bold run followed by ',' or ')' stays glued to it."""
    units, cur = [], []                   # a unit = the pieces between two spaces, possibly of several styles
    for s, w, c in runs:
        for k, piece in enumerate(s.split(" ")):
            if k > 0:
                if cur:
                    units.append(cur)
                cur = []
                units.append([(" ", w, c)])
            if piece:
                cur.append((piece, w, c))
    if cur:
        units.append(cur)
    lines, line, line_w = [], [], 0.0
    for u in units:
        if u[0][0] == " ":
            if line:
                line.append(u[0])
                line_w += _space_w(size, u[0][1])
            continue
        uw = sum(M.width(p, size, w) for p, w, _ in u)
        if line and line_w + uw > width:
            while line and line[-1][0] == " ":
                line_w -= _space_w(size, line[-1][1])
                line.pop()
            lines.append((line, line_w))
            line, line_w = [], 0.0
        line += u
        line_w += uw
    while line and line[-1][0] == " ":
        line_w -= _space_w(size, line[-1][1])
        line.pop()
    if line:
        lines.append((line, line_w))
    merged = []
    for ln, lw in lines:                  # join neighbouring pieces of the same style into one text object
        m = []
        for s, w, c in ln:
            if m and m[-1][1] == w and m[-1][2] == c:
                m[-1] = (m[-1][0] + s, w, c)
            else:
                m.append((s, w, c))
        merged.append((m, lw))
    return merged


def _draw_lines(ax, lines, x, y, width, size, align="left"):
    lh = _pt(size) * LEAD
    for i, (line, lw) in enumerate(lines):
        if align == "center":
            cx = x + (width - lw) / 2
        elif align == "right":
            cx = x + width - lw
        else:
            cx = x
        for s, w, c in line:
            ax.text(cx, y + i * lh + _pt(size) * 0.86, s, size=size, weight=w, color=c, va="baseline", ha="left")
            cx += M.width(s, size, w)
    return len(lines) * lh


# ----------------------------------------------------------------------------------------------- widgets
class Widget:
    def height(self, width):
        raise NotImplementedError

    def draw(self, ax, x, y, width):
        raise NotImplementedError


@dataclass
class Text(Widget):
    text: object
    size: float = None
    align: str = "left"
    weight: str = "normal"
    color: str = INK
    space_after: float = 0.06

    @property
    def sz(self):
        return self.size or SIZE["body"]

    def _lines(self, width):
        return _wrap(_runs(self.text, self.weight, self.color), width, self.sz)

    def height(self, width):
        return len(self._lines(width)) * _pt(self.sz) * LEAD + self.space_after

    def draw(self, ax, x, y, width):
        _draw_lines(ax, self._lines(width), x, y, width, self.sz, self.align)


@dataclass
class Chip:
    """A white rounded box holding one or more rich lines (first line can be a bold head)."""
    lines: list
    align: str = "center"
    edge: str = CHIP_EDGE
    fill: str = "white"
    size: float = None
    lw: float = 0.9
    math: bool = False      # True: the line is mathtext, drawn as one text object


@dataclass
class Chips(Widget):
    """A row of chips of equal width (or the given width shares), with optional arrows between them."""
    chips: list
    shares: list = None
    arrows: bool = False
    gap: float = 0.14
    pad: float = 0.09
    min_h: float = 0.0
    space_after: float = 0.1

    def _widths(self, width):
        n = len(self.chips)
        gap = self.gap + (0.2 if self.arrows else 0)
        free = width - gap * (n - 1)
        shares = self.shares or [1] * n
        return [free * s / sum(shares) for s in shares], gap

    def _chip_lines(self, chip, w):
        out = []
        for ln in chip.lines:
            if chip.math:
                out.append(("math", ln))
            else:
                out += [("text", l) for l in _wrap(_runs(ln), w - 2 * self.pad, (chip.size or SIZE["chip"]))]
        return out

    def _chip_h(self, chip, w):
        lh = _pt((chip.size or SIZE["chip"])) * LEAD
        n = 0
        for kind, _ in self._chip_lines(chip, w):
            n += 1.9 if kind == "math" else 1
        return n * lh + 2 * self.pad

    def height(self, width):
        ws, _ = self._widths(width)
        return max([self.min_h] + [self._chip_h(c, w) for c, w in zip(self.chips, ws)]) + self.space_after

    def draw(self, ax, x, y, width):
        ws, gap = self._widths(width)
        h = self.height(width) - self.space_after
        cx = x
        for i, (chip, w) in enumerate(zip(self.chips, ws)):
            _box(ax, cx, y, w, h, chip.fill, chip.edge, chip.lw, r=0.07)
            body = self._chip_lines(chip, w)
            lh = _pt((chip.size or SIZE["chip"])) * LEAD
            tot = sum((1.9 if k == "math" else 1) * lh for k, _ in body)
            ty = y + (h - tot) / 2
            for kind, ln in body:
                if kind == "math":
                    ax.text(cx + w / 2, ty + 0.95 * lh, ln, size=(chip.size or SIZE["chip"]) + 1, ha="center", va="center", color=INK)
                    ty += 1.9 * lh
                else:
                    _draw_lines(ax, [ln], cx + self.pad, ty, w - 2 * self.pad, (chip.size or SIZE["chip"]), chip.align)
                    ty += lh
            if self.arrows and i < len(self.chips) - 1:
                _arrow(ax, cx + w + 0.04, y + h / 2, cx + w + gap - 0.04, y + h / 2)
            cx += w + gap


@dataclass
class Row(Widget):
    """Widgets side by side; shares give the width split."""
    parts: list
    shares: list
    gap: float = 0.22
    valign: str = "center"

    def _ws(self, width):
        free = width - self.gap * (len(self.parts) - 1)
        return [free * s / sum(self.shares) for s in self.shares]

    def height(self, width):
        return max(p.height(w) for p, w in zip(self.parts, self._ws(width)))

    def draw(self, ax, x, y, width):
        H = self.height(width)
        cx = x
        for p, w in zip(self.parts, self._ws(width)):
            off = (H - p.height(w)) / 2 if self.valign == "center" else 0
            p.draw(ax, cx, y + off, w)
            cx += w + self.gap


@dataclass
class Stack(Widget):
    parts: list

    def height(self, width):
        return sum(p.height(width) for p in self.parts)

    def draw(self, ax, x, y, width):
        for p in self.parts:
            p.draw(ax, x, y, width)
            y += p.height(width)


@dataclass
class SeriesRows(Widget):
    """Rows of cells, one row per series: observed cells are filled, unknown future cells hold a '?'.

    rows: [(label, future_known, note)], with n_past and n_future cells per row.
    """
    rows: list
    n_past: int = 9
    n_future: int = 4
    cell: float = 0.19
    pitch: float = 0.28
    past_label: str = "past (context)"
    future_label: str = "future (horizon)"
    colors: list = None
    space_after: float = 0.0

    def height(self, width):
        return 0.24 + len(self.rows) * self.pitch - (self.pitch - self.cell) + self.space_after

    def draw(self, ax, x, y, width):
        c, g = self.cell, 0.032
        lab_w = max(M.width(r[0], SIZE["chip"], "bold") for r in self.rows) + 0.16
        gx = x + lab_w
        gap_mid = 0.2
        px_end = gx + self.n_past * (c + g) - g
        fx0 = px_end + gap_mid
        fx_end = fx0 + self.n_future * (c + g) - g
        top = y + 0.24
        ax.text((gx + px_end) / 2, y + 0.1, self.past_label, size=SIZE["small"], color=GREY, ha="center",
                va="center", style="italic")
        ax.text((fx0 + fx_end) / 2, y + 0.1, self.future_label, size=SIZE["small"], color=GREY, ha="center",
                va="center", style="italic")
        bottom = top + len(self.rows) * self.pitch - (self.pitch - c)
        mid = px_end + gap_mid / 2
        ax.plot([mid, mid], [top - 0.05, bottom + 0.05], color=LIGHT, lw=1.0, ls=(0, (3, 2)))
        for i, (label, known, note) in enumerate(self.rows):
            ry = top + i * self.pitch
            col = (self.colors or ["#3d3d3d"] * len(self.rows))[i]
            ax.text(x, ry + c / 2, label, size=SIZE["chip"], weight="bold", color=INK, va="center")
            for k in range(self.n_past):
                _cell(ax, gx + k * (c + g), ry, c, col)
            for k in range(self.n_future):
                cx = fx0 + k * (c + g)
                if known:
                    _cell(ax, cx, ry, c, col)
                else:
                    _cell(ax, cx, ry, c, None)
                    ax.text(cx + c / 2, ry + c / 2 + 0.006, "?", size=SIZE["small"] - 1, color=LIGHT,
                            ha="center", va="center", weight="bold")
            ax.text(fx_end + 0.14, ry + c / 2, note, size=SIZE["small"], color=GREY, va="center")


@dataclass
class PatchToToken(Widget):
    """A patch (values, time index, observed flags) -> MLP -> one token, then one row as a token sequence."""
    n_tokens_past: int = 4
    n_tokens_future: int = 2
    patch_len: int = 16
    rows: tuple = ("values", "time index", "observed?")
    token_label: str = "one token\n(768 numbers)"
    seq_label: str = "a row as tokens"
    reg: bool = True
    mlp_label: str = "MLP"
    sequence: bool = True
    space_after: float = 0.0

    def height(self, width):
        return 0.7 + self.space_after

    def draw(self, ax, x, y, width):
        c, g, rg = 0.082, 0.014, 0.06
        cols = ["#3d3d3d", "#8c8c8c", "#c4c4c4"]
        lab_w = max(M.width(r, SIZE["small"]) for r in self.rows) + 0.08
        px = x + lab_w
        for i, (lab, col) in enumerate(zip(self.rows, cols)):
            ry = y + 0.06 + i * (c + rg)
            ax.text(px - 0.07, ry + c / 2, lab, size=SIZE["small"], color=GREY, ha="right", va="center")
            for k in range(self.patch_len):
                _cell(ax, px + k * (c + g), ry, c, col, lw=0.4)
        pw = self.patch_len * (c + g) - g
        ph = 3 * c + 2 * rg
        ax.text(px + pw / 2, y + 0.06 + ph + 0.07, f"one patch: {self.patch_len} steps",
                size=SIZE["small"], color=GREY, ha="center", va="top", style="italic")
        mid_y = y + 0.06 + ph / 2
        a0 = px + pw + 0.1
        a1 = a0 + max(0.5, M.width(self.mlp_label, SIZE["small"]) + 0.12)
        _arrow(ax, a0, mid_y, a1, mid_y)
        ax.text((a0 + a1) / 2, mid_y - 0.06, self.mlp_label, size=SIZE["small"], color=GREY, ha="center",
                va="bottom")
        tw, th = 0.2, 0.38
        tx = a1 + 0.04
        _box(ax, tx, mid_y - th / 2, tw, th, "#d9e6f7", ACCENT["blue"], 1.2, r=0.04)
        ax.text(tx + tw / 2, mid_y + th / 2 + 0.05, self.token_label, size=SIZE["small"], color=GREY,
                ha="center", va="top", linespacing=1.1)
        if not self.sequence:
            return
        # right: one row as a sequence of tokens
        t, tg = 0.25, 0.055
        n = self.n_tokens_past + self.n_tokens_future
        seq_w = n * (t + tg) + (0.48 + tg if self.reg else 0) - tg
        sx = x + width - seq_w
        sy = mid_y - t / 2
        x0 = sx
        for k in range(self.n_tokens_past):
            _box(ax, sx, sy, t, t, "#d9e6f7", ACCENT["blue"], 1.0, r=0.04)
            sx += t + tg
        if self.reg:
            _box(ax, sx, sy, 0.48, t, "#fbe3cf", ACCENT["orange"], 1.0, r=0.04)
            ax.text(sx + 0.24, sy + t / 2, "REG", size=SIZE["small"], weight="bold", color=INK, ha="center",
                    va="center")
            reg_c = sx + 0.24
            sx += 0.48 + tg
        f0 = sx
        for k in range(self.n_tokens_future):
            _box(ax, sx, sy, t, t, "white", ACCENT["blue"], 1.0, r=0.04, ls=(0, (2.5, 1.5)))
            sx += t + tg
        ax.text((x0 + x0 + self.n_tokens_past * (t + tg) - tg) / 2, sy + t + 0.07, "past", size=SIZE["small"],
                color=GREY, ha="center", va="top", style="italic")
        if self.reg:
            ax.text(reg_c, sy + t + 0.07, "separator", size=SIZE["small"], color=GREY, ha="center", va="top",
                    style="italic")
        ax.text((f0 + sx - tg) / 2, sy + t + 0.07, "future", size=SIZE["small"], color=GREY, ha="center",
                va="top", style="italic")
        ax.text((x0 + sx - tg) / 2, sy - 0.07, self.seq_label, size=SIZE["small"], color=GREY, ha="center",
                va="bottom")


@dataclass
class AttentionGrid(Widget):
    """Token grid (rows = series, columns = patches): the dark token reads along its row and down its column."""
    row_labels: list
    n_past: int = 4
    n_future: int = 2
    reg: bool = True
    focus_row: int = 0
    cell: float = 0.22
    gap: float = 0.05
    space_after: float = 0.0

    def height(self, width):
        n = len(self.row_labels)
        return 0.27 + n * (self.cell + self.gap) - self.gap + 0.62 + self.space_after

    def draw(self, ax, x, y, width):
        c, g = self.cell, self.gap
        nrow = len(self.row_labels)
        cols = ["p"] * self.n_past + (["r"] if self.reg else []) + ["f"] * self.n_future
        ncol = len(cols)
        lab_w = max(M.width(l, SIZE["small"]) for l in self.row_labels) + 0.1
        gw = ncol * (c + g) - g
        gx = x + lab_w
        top = y + 0.27
        fc = cols.index("f")
        fr = self.focus_row
        bot = top + nrow * (c + g) - g
        _box(ax, gx - 0.05, top + fr * (c + g) - 0.05, gw + 0.1, c + 0.1, "#cfe0fa", "none", 0, r=0.06)
        _box(ax, gx + fc * (c + g) - 0.05, top - 0.05, c + 0.1, bot - top + 0.1, "#f8cfa9", "none", 0, r=0.06)
        for i, lab in enumerate(self.row_labels):
            ry = top + i * (c + g)
            ax.text(gx - 0.1, ry + c / 2, lab, size=SIZE["small"], color=GREY, ha="right", va="center")
            for k, kind in enumerate(cols):
                cx = gx + k * (c + g)
                if (i, k) == (fr, fc):
                    _box(ax, cx, ry, c, c, INK, INK, 1.0, r=0.035)
                elif kind == "r":
                    _box(ax, cx, ry, c, c, "#fbe3cf", ACCENT["orange"], 0.8, r=0.035)
                else:
                    _box(ax, cx, ry, c, c, "white", "#8f8f8f", 0.8, r=0.035,
                         ls=(0, (2.2, 1.4)) if kind == "f" else "-")
        ax.annotate("", xy=(gx + gw + 0.03, top - 0.15), xytext=(gx - 0.03, top - 0.15),
                    arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.18", color=ACCENT["blue"],
                                    lw=1.3, shrinkA=0, shrinkB=0))
        ax.text(gx + gw / 2, top - 0.2, "time attention", size=SIZE["small"], weight="bold", color=ACCENT["blue"],
                ha="center", va="bottom")
        ax.text(gx + self.n_past * (c + g) / 2 - g / 2, bot + 0.07, "past", size=SIZE["small"], color=GREY,
                ha="center", va="top", style="italic")
        if self.reg:
            ax.text(gx + self.n_past * (c + g) + c / 2, bot + 0.07, "REG", size=SIZE["small"], color=GREY,
                    ha="center", va="top", style="italic")
        ax.text(gx + fc * (c + g) + (self.n_future * (c + g) - g) / 2, bot + 0.07, "future", size=SIZE["small"],
                color=GREY, ha="center", va="top", style="italic")
        bx = gx + fc * (c + g) + c / 2
        ax.annotate("", xy=(bx, bot + 0.3), xytext=(bx, bot + 0.07 + 0.17),
                    arrowprops=dict(arrowstyle="-", color="none"))
        ax.text(bx, bot + 0.26, "group\nattention", size=SIZE["small"], weight="bold", color=ACCENT["orange"],
                ha="center", va="top", linespacing=1.05)


@dataclass
class Fan(Widget):
    """A small forecast picture: context line, median and two quantile bands, legend underneath."""
    h: float = 0.88
    labels: tuple = ("median", "80% band", "60% band")
    space_after: float = 0.0

    def height(self, width):
        return self.h + 0.27 + self.space_after

    def draw(self, ax, x, y, width):
        rng = np.random.default_rng(7)
        n, hz = 64, 22
        t = np.arange(n + hz)
        base = 0.62 * np.sin(2 * np.pi * t / 26) + 0.003 * t
        past = base[:n] + rng.normal(0, 0.07, n)
        fut = base[n - 1:]
        k = np.arange(len(fut))
        w60, w80 = 0.08 + 0.012 * k, 0.15 + 0.022 * k
        X = lambda i: x + i / (n + hz - 1) * width
        lo, hi = -1.35, 1.35
        Y = lambda v: y + 0.04 + (hi - v) / (hi - lo) * (self.h - 0.08)
        xs = np.array([X(i) for i in range(n + hz)])
        ax.fill_between(xs[n - 1:], [Y(v) for v in fut - w80], [Y(v) for v in fut + w80], color="#f6cdb0", lw=0)
        ax.fill_between(xs[n - 1:], [Y(v) for v in fut - w60], [Y(v) for v in fut + w60], color="#eea070", lw=0)
        ax.plot(xs[:n], [Y(v) for v in past], color="#4a4a4a", lw=1.1)
        ax.plot(xs[n - 1:], [Y(v) for v in fut], color="#b6501a", lw=1.5)
        ax.plot([xs[n - 1]] * 2, [y, y + self.h], color=LIGHT, lw=1.0, ls=(0, (3, 2)))
        ly = y + self.h + 0.15
        items = [("line", "#b6501a"), ("box", "#eea070"), ("box", "#f6cdb0")]
        widths = [0.26 + 0.07 + M.width(l, SIZE["small"]) for l in self.labels]
        lx = x + (width - sum(widths) - 0.2 * (len(widths) - 1)) / 2
        for (kind, col), lab, w in zip(items, self.labels, widths):
            if kind == "line":
                ax.plot([lx, lx + 0.26], [ly, ly], color=col, lw=2)
            else:
                ax.add_patch(Rectangle((lx, ly - 0.06), 0.26, 0.12, color=col, lw=0))
            ax.text(lx + 0.33, ly, lab, size=SIZE["small"], color=GREY, va="center")
            lx += w + 0.2


# ----------------------------------------------------------------------------------------------- primitives
def _box(ax, x, y, w, h, fill, edge, lw, r=ROUND, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fill, ec=edge, lw=lw, ls=ls,
                       joinstyle="round")
    ax.add_patch(p)
    return p


def _cell(ax, x, y, s, fill, lw=0.7):
    if fill is None:
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.025", fc="white",
                                    ec="#8f8f8f", lw=lw))
    else:
        ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.025", fc=fill, ec=fill,
                                    lw=lw))


def _arrow(ax, x0, y0, x1, y1, color=INK, lw=1.1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4.2,head_width=2.6",
                                 color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=1))


def _elbow(ax, x0, y0, x1, y1, color=INK, lw=1.3, r=None):
    """Connector from (x0, y0) down, across, and down to an arrow head at (x1, y1), with rounded corners (y grows
    downwards). The run across sits a little above the middle, so the last leg is long enough for the head to
    clear the corner. When the two ends nearly line up it is one straight arrow."""
    ym = y0 + 0.4 * (y1 - y0)
    r = min(CORNER if r is None else r, (ym - y0) * 0.8, (y1 - ym) * 0.8, abs(x1 - x0) / 2)
    if abs(x1 - x0) < 0.02 or r <= 0.005:
        _arrow(ax, x1, y0, x1, y1, color=color, lw=lw)
        return
    s = 1 if x1 > x0 else -1
    verts = [(x0, y0), (x0, ym - r), (x0, ym), (x0 + s * r, ym), (x1 - s * r, ym), (x1, ym), (x1, ym + r), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CURVE3,
             MplPath.CURVE3, MplPath.LINETO]
    ax.add_patch(FancyArrowPatch(path=MplPath(verts, codes), arrowstyle="-|>,head_length=4.2,head_width=2.6",
                                 color=color, lw=lw, mutation_scale=1, joinstyle="round", capstyle="round"))


# ----------------------------------------------------------------------------------------------- figure
@dataclass
class Step:
    title: str
    color: str
    parts: list
    highlight: bool = False      # red border, as in the reference (the one step that is new in this model)


@dataclass
class Explainer:
    title: str
    subtitle: str
    steps: list
    footer: list = field(default_factory=list)     # [(text, size, weight, color)]


def draw_explainer(ex: Explainer, path=None, show=False, dpi=200):
    with matplotlib.rc_context(RC):
        fig = plt.figure(figsize=(W, 30), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, W)
        ax.set_ylim(30, 0)
        ax.axis("off")
        y = 0.24
        inner = W - 2 * MARGIN - 2 * PAD_X
        y += Text(ex.title, SIZE["title"], "center", "bold", space_after=0).height(W - 2 * MARGIN)
        Text(ex.title, SIZE["title"], "center", "bold").draw(ax, MARGIN, 0.24, W - 2 * MARGIN)
        if ex.subtitle:
            st = Text(ex.subtitle, SIZE["sub"], "center", color=GREY, space_after=0)
            st.draw(ax, MARGIN, y + 0.02, W - 2 * MARGIN)
            y += st.height(W - 2 * MARGIN) + 0.02
        y += 0.16
        for i, step in enumerate(ex.steps):
            fill, edge = PALETTE[step.color]
            head = Text(step.title, SIZE["head"], weight="bold", space_after=0.07)
            h = PAD_TOP + head.height(inner) + sum(p.height(inner) for p in step.parts) + PAD_BOTTOM
            _box(ax, MARGIN, y, W - 2 * MARGIN, h, fill, ACCENT["red"] if step.highlight else edge,
                 1.8 if step.highlight else 1.25)
            cy = y + PAD_TOP
            head.draw(ax, MARGIN + PAD_X, cy, inner)
            cy += head.height(inner)
            for p in step.parts:
                p.draw(ax, MARGIN + PAD_X, cy, inner)
                cy += p.height(inner)
            y += h
            if i < len(ex.steps) - 1:
                _arrow(ax, W / 2, y + 0.06, W / 2, y + GAP - 0.04, lw=1.3)
                y += GAP
        y += 0.2
        for text, size, weight, color in ex.footer:
            t = Text(text, size, "center", weight, color, space_after=0.04)
            t.draw(ax, MARGIN, y, W - 2 * MARGIN)
            y += t.height(W - 2 * MARGIN)
        H = y + 0.22
        fig.set_size_inches(W, H)
        ax.set_ylim(H, 0)
        if path:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white")
            fig.savefig(path.with_suffix(".pdf"), facecolor="white")
        if not show:
            plt.close(fig)
        return fig


# ================================================================================================ landscape layout
# Sizes for the left-to-right layout: the figure is 9 in wide and is meant to print at the text width (about 6 in),
# so these are about 2/3 of the printed size: body text prints at about 7.3 pt.
SIZE_H = {"title": 17.5, "sub": 10.6, "head": 12.6, "body": 10.6, "chip": 10.2, "small": 9.6, "foot": 12.0}
W_H = 9.0
MARGIN_H = 0.2
GAP_H = 0.42          # horizontal space between two panels (holds the arrow)
GAP_2UP = 0.38        # the same inside a two-up grid, where panels are narrower
ARROW_PAD = 0.08      # clearance between a connector's ends and the panels it joins
CORNER = 0.09         # radius of a connector's rounded corners
PAD_H = 0.13          # inner padding of a panel


@dataclass
class SeriesRowsStacked(Widget):
    """Narrow version of SeriesRows: each row's label sits above its cells."""
    rows: list            # [(bold role, grey example, future_known)]
    n_past: int = 7
    n_future: int = 3
    colors: list = None
    space_after: float = 0.0

    def _cell(self, width):
        g, mid = 0.024, 0.12
        n = self.n_past + self.n_future
        return min(0.17, (width - mid - g * (n - 2)) / n), g, mid

    def height(self, width):
        c, _, _ = self._cell(width)
        return 0.2 + len(self.rows) * (0.2 + c + 0.1) - 0.1 + self.space_after

    def draw(self, ax, x, y, width):
        c, g, mid = self._cell(width)
        px_end = x + self.n_past * (c + g) - g
        fx0 = px_end + mid
        fx_end = fx0 + self.n_future * (c + g) - g
        ax.text((x + px_end) / 2, y + 0.08, "past", size=SIZE["small"], color=GREY, ha="center", va="center",
                style="italic")
        ax.text((fx0 + fx_end) / 2, y + 0.08, "future", size=SIZE["small"], color=GREY, ha="center", va="center",
                style="italic")
        top = y + 0.2
        for i, (role, example, known) in enumerate(self.rows):
            ry = top + i * (0.2 + c + 0.1)
            col = (self.colors or ["#3d3d3d"] * len(self.rows))[i]
            rs, es = SIZE["chip"], SIZE["small"]
            while M.width(role + " ", rs, "bold") + M.width(example, es) > width - 0.04 and es > 7.5:
                rs, es = rs - 0.3, es - 0.3          # shrink a little so the label fits a narrow panel
            ax.text(x, ry + 0.09, role, size=rs, weight="bold", color=INK, va="center")
            ax.text(x + M.width(role + " ", rs, "bold"), ry + 0.09, example, size=es, color=GREY, va="center")
            cy = ry + 0.2
            for k in range(self.n_past):
                _cell(ax, x + k * (c + g), cy, c, col)
            for k in range(self.n_future):
                cx = fx0 + k * (c + g)
                if known:
                    _cell(ax, cx, cy, c, col)
                else:
                    _cell(ax, cx, cy, c, None)
                    ax.text(cx + c / 2, cy + c / 2 + 0.005, "?", size=SIZE["small"] - 1.2, color=LIGHT,
                            ha="center", va="center", weight="bold")
        m = px_end + mid / 2
        for i in range(len(self.rows)):
            cy = top + i * (0.2 + c + 0.1) + 0.2
            ax.plot([m, m], [cy - 0.03, cy + c + 0.03], color=LIGHT, lw=0.9, ls=(0, (2, 1.6)))


@dataclass
class PatchToTokenV(Widget):
    """Narrow version of PatchToToken: a patch (values, time index, observed flags), an arrow down, one token."""
    patch_len: int = 16
    rows: tuple = ("value", "time", "seen?")
    caption: str = "one patch = 16 steps"
    mlp_label: str = "residual MLP"
    token_label: str = "one token: 768 numbers"
    space_after: float = 0.0

    def _c(self, width):
        lab_w = max(M.width(r, SIZE["small"]) for r in self.rows) + 0.07
        g = 0.012 if self.patch_len <= 16 else 0.005
        c = min(0.085, (width - lab_w - g * (self.patch_len - 1)) / self.patch_len)
        return lab_w, g, c

    def height(self, width):
        _, _, c = self._c(width)
        n = len(self.rows)
        return 0.02 + n * c + (n - 1) * max(0.045, 0.15 - c) + 0.05 + 0.2 + 0.3 + 0.15 + 0.24 + self.space_after

    def draw(self, ax, x, y, width):
        lab_w, g, c = self._c(width)
        rg = max(0.045, 0.15 - c)
        cols = ["#3d3d3d", "#8c8c8c", "#c4c4c4"] if len(self.rows) == 3 else ["#3d3d3d", "#c4c4c4"]
        pw = self.patch_len * (c + g) - g
        px = x + lab_w + max(0.0, (width - lab_w - pw) / 2)
        for i, (lab, col) in enumerate(zip(self.rows, cols)):
            ry = y + 0.02 + i * (c + rg)
            ax.text(px - 0.06, ry + c / 2, lab, size=SIZE["small"], color=GREY, ha="right", va="center")
            for k in range(self.patch_len):
                _cell(ax, px + k * (c + g), ry, c, col, lw=0.4)
        n = len(self.rows)
        cy = y + 0.02 + n * c + (n - 1) * rg + 0.05
        ax.text(x + width / 2, cy, self.caption, size=SIZE["small"], color=GREY, ha="center", va="top",
                style="italic")
        a0 = cy + 0.2
        ax_x = x + 0.3
        _arrow(ax, ax_x, a0, ax_x, a0 + 0.27)
        ax.text(ax_x + 0.09, a0 + 0.135, self.mlp_label, size=SIZE["small"], color=GREY, ha="left", va="center")
        ty = a0 + 0.3
        _box(ax, x + 0.05, ty, width - 0.1, 0.15, "#d9e6f7", ACCENT["blue"], 1.1, r=0.04)
        ax.text(x + width / 2, ty + 0.2, self.token_label, size=SIZE["small"] - 0.4, color=INK, ha="center",
                va="top", weight="bold")


@dataclass
class TokenSequence(Widget):
    """One row as tokens: past tokens, the REG separator, future tokens (dashed)."""
    n_past: int = 3
    n_future: int = 2
    label: str = "REG separates past from future"
    space_after: float = 0.0

    def height(self, width):
        return 0.2 + 0.2 + 0.22 + self.space_after

    def draw(self, ax, x, y, width):
        t, tg, rw = 0.19, 0.05, 0.38
        total = (self.n_past + self.n_future) * (t + tg) + rw
        sx = x + max(0.0, (width - total) / 2)
        ax.text(x + width / 2, y + 0.08, self.label, size=SIZE["small"], color=GREY, ha="center", va="center")
        ty = y + 0.2
        p0 = sx
        for _ in range(self.n_past):
            _box(ax, sx, ty, t, t, "#d9e6f7", ACCENT["blue"], 1.0, r=0.035)
            sx += t + tg
        _box(ax, sx, ty, rw, t, "#fbe3cf", ACCENT["orange"], 1.0, r=0.035)
        ax.text(sx + rw / 2, ty + t / 2, "REG", size=SIZE["small"] - 0.8, weight="bold", color=INK, ha="center",
                va="center")
        reg_c = sx + rw / 2
        sx += rw + tg
        f0 = sx
        for _ in range(self.n_future):
            _box(ax, sx, ty, t, t, "white", ACCENT["blue"], 1.0, r=0.035, ls=(0, (2.5, 1.5)))
            sx += t + tg
        ly = ty + t + 0.04
        ax.text((p0 + reg_c - rw / 2 - tg) / 2, ly, "past", size=SIZE["small"], color=GREY, ha="center", va="top",
                style="italic")
        ax.text((f0 + sx - tg) / 2, ly, "future", size=SIZE["small"], color=GREY, ha="center", va="top",
                style="italic")


@dataclass
class FanV(Widget):
    """Fan chart with the legend stacked under it (for a narrow panel)."""
    h: float = 0.8
    labels: tuple = ("median", "60% band", "80% band")
    space_after: float = 0.04
    bands: bool = True            # False: a point forecast only (one line, one legend entry)
    samples: int = 0              # >0: draw that many sample paths instead of bands

    def height(self, width):
        n = 3 if self.bands and not self.samples else (2 if self.samples else 1)
        return self.h + 0.1 + n * 0.19 + self.space_after

    def draw(self, ax, x, y, width):
        rng = np.random.default_rng(7)
        n, hz = 52, 20
        t = np.arange(n + hz)
        base = 0.62 * np.sin(2 * np.pi * t / 24) + 0.003 * t
        past = base[:n] + rng.normal(0, 0.07, n)
        fut = base[n - 1:]
        k = np.arange(len(fut))
        w60, w80 = 0.08 + 0.012 * k, 0.15 + 0.022 * k
        X = lambda i: x + i / (n + hz - 1) * width
        lo, hi = -1.3, 1.3
        Y = lambda v: y + 0.03 + (hi - v) / (hi - lo) * (self.h - 0.06)
        xs = np.array([X(i) for i in range(n + hz)])
        if self.samples:
            rs = np.random.default_rng(3)
            for _ in range(self.samples):
                path = fut + np.cumsum(rs.normal(0, 0.05, len(fut)))
                ax.plot(xs[n - 1:], [Y(v) for v in path], color="#f0b48a", lw=0.7, alpha=0.9)
        elif self.bands:
            ax.fill_between(xs[n - 1:], [Y(v) for v in fut - w80], [Y(v) for v in fut + w80], color="#f6cdb0", lw=0)
            ax.fill_between(xs[n - 1:], [Y(v) for v in fut - w60], [Y(v) for v in fut + w60], color="#eea070", lw=0)
        ax.plot(xs[:n], [Y(v) for v in past], color="#4a4a4a", lw=1.0)
        ax.plot(xs[n - 1:], [Y(v) for v in fut], color="#b6501a", lw=1.4)
        ax.plot([xs[n - 1]] * 2, [y, y + self.h], color=LIGHT, lw=0.9, ls=(0, (3, 2)))
        if self.samples:
            items = [("line", "#b6501a"), ("thin", "#f0b48a")]
        elif self.bands:
            items = [("line", "#b6501a"), ("box", "#eea070"), ("box", "#f6cdb0")]
        else:
            items = [("line", "#b6501a")]
        for i, ((kind, col), lab) in enumerate(zip(items, self.labels)):
            ly = y + self.h + 0.1 + i * 0.19 + 0.08
            if kind == "line":
                ax.plot([x, x + 0.24], [ly, ly], color=col, lw=2)
            elif kind == "thin":
                for dy in (-0.035, 0.0, 0.035):
                    ax.plot([x, x + 0.24], [ly + dy, ly + dy], color=col, lw=0.8)
            else:
                ax.add_patch(Rectangle((x, ly - 0.055), 0.24, 0.11, color=col, lw=0))
            ax.text(x + 0.31, ly, lab, size=SIZE["small"], color=GREY, va="center")


def draw_explainer_h(ex: Explainer, shares, path=None, show=False, dpi=200):
    """The same steps as draw_explainer, set side by side from left to right."""
    saved = dict(SIZE)
    SIZE.update(SIZE_H)
    try:
        with matplotlib.rc_context(RC):
            fig = plt.figure(figsize=(W_H, 20), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, W_H)
            ax.set_ylim(20, 0)
            ax.axis("off")
            full = W_H - 2 * MARGIN_H
            y = 0.16
            t = Text(ex.title, SIZE["title"], "center", "bold", space_after=0)
            t.draw(ax, MARGIN_H, y, full)
            y += t.height(full)
            if ex.subtitle:
                st = Text(ex.subtitle, SIZE["sub"], "center", color=GREY, space_after=0)
                st.draw(ax, MARGIN_H, y + 0.01, full)
                y += st.height(full) + 0.01
            y += 0.12
            free = full - GAP_H * (len(ex.steps) - 1)
            ws = [free * s / sum(shares) for s in shares]
            heads = [Text(st.title, SIZE["head"], weight="bold", space_after=0.06) for st in ex.steps]
            hh = max(h.height(w - 2 * PAD_H) for h, w in zip(heads, ws))      # headings share one height
            body = [sum(p.height(w - 2 * PAD_H) for p in st.parts) for st, w in zip(ex.steps, ws)]
            H = PAD_H + hh + max(body) + PAD_H + 0.02
            x = MARGIN_H
            rects = []
            for i, (step, w, head) in enumerate(zip(ex.steps, ws, heads)):
                fill, edge = PALETTE[step.color]
                _box(ax, x, y, w, H, fill, ACCENT["red"] if step.highlight else edge, 1.8 if step.highlight else 1.2,
                     r=0.1)
                head.draw(ax, x + PAD_H, y + PAD_H, w - 2 * PAD_H)
                cy = y + PAD_H + hh
                for p in step.parts:
                    p.draw(ax, x + PAD_H, cy, w - 2 * PAD_H)
                    cy += p.height(w - 2 * PAD_H)
                if i < len(ex.steps) - 1:
                    _arrow(ax, x + w + ARROW_PAD, y + H / 2, x + w + GAP_H - ARROW_PAD, y + H / 2, lw=1.3)
                rects.append((x, y, w, H))
                x += w + GAP_H
            y += H + 0.14
            for text, size, weight, color in ex.footer:
                tt = Text(text, size, "center", weight, color, space_after=0.03)
                tt.draw(ax, MARGIN_H, y, full)
                y += tt.height(full)
            Ht = y + 0.12
            fig.set_size_inches(W_H, Ht)
            ax.set_ylim(Ht, 0)
            fig._panel_rects = rects
            fig._layout_issues = check_layout(fig, ax, rects)
            for msg in fig._layout_issues:
                print("LAYOUT:", msg)
            if path:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white")
                fig.savefig(path.with_suffix(".pdf"), facecolor="white")
            if not show:
                plt.close(fig)
            return fig
    finally:
        SIZE.clear()
        SIZE.update(saved)


@dataclass
class AttentionGridC(Widget):
    """Compact token grid: column labels on top, a blue band along the focus row (time attention) and a rose band
    down the focus column (group attention); the chips underneath carry the same colours and explain them."""
    row_labels: list
    n_past: int = 3
    n_future: int = 2
    reg: bool = True
    focus_row: int = 0
    cell: float = 0.17
    gap: float = 0.04
    row_band: str = "#cfe0fa"
    col_band: str = "#f6c9c4"
    space_after: float = 0.08
    causal: bool = False          # the blue band stops at the focus cell (time attention looks back only)
    col_rows: list = None         # rows the focus token reads down its column (None = all)
    fill_rows: list = None        # rows whose future cells are known (drawn solid, not dashed)
    row_colors: list = None       # optional label colour per row
    focus_col: int = None         # column of the black focus token (None = first future column)

    def height(self, width):
        n = len(self.row_labels)
        return 0.2 + n * (self.cell + self.gap) - self.gap + 0.06 + self.space_after

    def draw(self, ax, x, y, width):
        c, g = self.cell, self.gap
        cols = ["p"] * self.n_past + (["r"] if self.reg else []) + ["f"] * self.n_future
        lab_w = max(M.width(l, SIZE["small"]) for l in self.row_labels) + 0.08
        gw = len(cols) * (c + g) - g
        gx = x + lab_w + max(0.0, (width - lab_w - gw) / 2)
        top = y + 0.2
        fc, fr = (cols.index("f") if self.focus_col is None else self.focus_col), self.focus_row
        f0 = cols.index("f")
        nrow = len(self.row_labels)
        bot = top + nrow * (c + g) - g
        if self.row_band:
            rw = (fc + 1) * (c + g) - g if self.causal else gw
            _box(ax, gx - 0.04, top + fr * (c + g) - 0.04, rw + 0.08, c + 0.08, self.row_band, "none", 0, r=0.05)
        if self.col_band:
            if self.col_rows is None:
                _box(ax, gx + fc * (c + g) - 0.04, top - 0.04, c + 0.08, bot - top + 0.08, self.col_band, "none", 0,
                     r=0.05)
            else:
                for rr in self.col_rows:
                    _box(ax, gx + fc * (c + g) - 0.04, top + rr * (c + g) - 0.04, c + 0.08, c + 0.08, self.col_band,
                         "none", 0, r=0.04)
        for i, lab in enumerate(self.row_labels):
            ry = top + i * (c + g)
            ax.text(gx - 0.08, ry + c / 2, lab, size=SIZE["small"],
                    color=(self.row_colors[i] if self.row_colors else GREY), ha="right", va="center")
            known = self.fill_rows is not None and i in self.fill_rows
            for k, kind in enumerate(cols):
                cx = gx + k * (c + g)
                if (i, k) == (fr, fc):
                    _box(ax, cx, ry, c, c, INK, INK, 1.0, r=0.03)
                elif kind == "r":
                    _box(ax, cx, ry, c, c, "#fbe3cf", ACCENT["orange"], 0.8, r=0.03)
                elif kind == "f" and known:
                    _box(ax, cx, ry, c, c, "#dcefe4", ACCENT["green"], 0.8, r=0.03)
                else:
                    _box(ax, cx, ry, c, c, "white", "#8f8f8f", 0.8, r=0.03, ls=(0, (2.2, 1.4)) if kind == "f" else "-")
        ly = y + 0.09
        ax.text(gx + (self.n_past * (c + g) - g) / 2, ly, "past", size=SIZE["small"], color=GREY, ha="center",
                va="center", style="italic")
        ax.text(gx + f0 * (c + g) + (self.n_future * (c + g) - g) / 2, ly, "future", size=SIZE["small"],
                color=GREY, ha="center", va="center", style="italic")


@dataclass
class EncDec(Widget):
    """Encoder-decoder picture: context tokens (+ REG) on top, cross-attention lines down to one decoder token."""
    n_ctx: int = 4
    reg: bool = True
    top_label: str = "context tokens (encoder)"
    dec_label: str = "start"
    bottom_label: str = "one decoder token"
    line_color: str = ACCENT["red"]
    space_after: float = 0.08
    dec_labels: tuple = None      # several decoder tokens side by side (e.g. Kairos's two forecast tokens)

    def height(self, width):
        return 0.2 + 0.19 + 0.36 + 0.22 + 0.22 + self.space_after

    def draw(self, ax, x, y, width):
        t, tg, rw = 0.19, 0.05, 0.38
        n = self.n_ctx
        total = n * (t + tg) + (rw if self.reg else -tg)
        sx = x + max(0.0, (width - total) / 2)
        ax.text(x + width / 2, y + 0.08, self.top_label, size=SIZE["small"], color=GREY, ha="center", va="center")
        ty = y + 0.2
        centres = []
        for _ in range(n):
            _box(ax, sx, ty, t, t, "#d9e6f7", ACCENT["blue"], 1.0, r=0.035)
            centres.append(sx + t / 2)
            sx += t + tg
        if self.reg:
            _box(ax, sx, ty, rw, t, "#fbe3cf", ACCENT["orange"], 1.0, r=0.035)
            ax.text(sx + rw / 2, ty + t / 2, "REG", size=SIZE["small"] - 0.8, weight="bold", color=INK, ha="center",
                    va="center")
            centres.append(sx + rw / 2)
        labels = self.dec_labels or (self.dec_label,)
        dh, dg = 0.22, 0.12
        dw = 0.62 if len(labels) == 1 else max(M.width(l, SIZE["small"] - 0.6) + 0.14 for l in labels)
        tot = len(labels) * (dw + dg) - dg
        dy = ty + t + 0.36
        for k, lab in enumerate(labels):
            dx = x + width / 2 - tot / 2 + k * (dw + dg)
            for cx in centres:
                ax.plot([cx, dx + dw / 2], [ty + t + 0.02, dy - 0.02], color=self.line_color, lw=0.9, alpha=0.8)
            _box(ax, dx, dy, dw, dh, "#e5f3ec", ACCENT["green"], 1.1, r=0.04)
            ax.text(dx + dw / 2, dy + dh / 2, lab, size=SIZE["small"] - (0 if len(labels) == 1 else 0.6), color=INK,
                    ha="center", va="center", weight="bold")
        ax.text(x + width / 2, dy + dh + 0.05, self.bottom_label, size=SIZE["small"], color=GREY, ha="center",
                va="top", style="italic")


# ----------------------------------------------------------------------------------------------- two-up grid
def _panel_row(ax, steps, shares, x, y, width, arrows=True, gap=GAP_2UP):
    """Draw steps side by side in one row of equal-height panels; return (height, panel rects)."""
    free = width - gap * (len(steps) - 1)
    ws = [free * s / sum(shares) for s in shares]
    heads = [Text(st.title, SIZE["head"], weight="bold", space_after=0.06) for st in steps]
    hh = max(h.height(w - 2 * PAD_H) for h, w in zip(heads, ws))
    body = [sum(p.height(w - 2 * PAD_H) for p in st.parts) for st, w in zip(steps, ws)]
    H = PAD_H + hh + max(body) + PAD_H + 0.02
    rects = []
    for i, (step, w, head) in enumerate(zip(steps, ws, heads)):
        fill, edge = PALETTE[step.color]
        _box(ax, x, y, w, H, fill, ACCENT["red"] if step.highlight else edge, 1.8 if step.highlight else 1.2, r=0.1)
        head.draw(ax, x + PAD_H, y + PAD_H, w - 2 * PAD_H)
        cy = y + PAD_H + hh
        for p in step.parts:
            p.draw(ax, x + PAD_H, cy, w - 2 * PAD_H)
            cy += p.height(w - 2 * PAD_H)
        rects.append((x, y, w, H))
        if arrows and i < len(steps) - 1:
            _arrow(ax, x + w + ARROW_PAD, y + H / 2, x + w + gap - ARROW_PAD, y + H / 2, lw=1.3)
        x += w + gap
    return H, rects


def _model_grid(ax, ex, shares, x, y, width, title_size=None, rects=None):
    """One model as a 2 x 2 grid of panels (1 2 / 3 4) with its title above and its footer below."""
    t = Text(ex.title, title_size or SIZE["head"] + 2.5, "center", "bold", space_after=0)
    t.draw(ax, x, y, width)
    y += t.height(width)
    if ex.subtitle:
        st = Text(ex.subtitle, SIZE["small"] + 0.4, "center", color=GREY, space_after=0)
        st.draw(ax, x, y + 0.01, width)
        y += st.height(width) + 0.01
    y += 0.1
    h1, r1 = _panel_row(ax, ex.steps[:2], shares[:2], x, y, width)
    gap_v = 0.56
    y2 = y + h1 + gap_v
    h2, r2 = _panel_row(ax, ex.steps[2:], shares[2:], x, y2, width)
    if rects is not None:
        rects.extend(r1 + r2)
    # rounded elbow from the bottom of panel 2 to the top of panel 3, with room for both legs and the head
    x2 = r1[1][0] + r1[1][2] / 2
    x3 = r2[0][0] + r2[0][2] / 2
    _elbow(ax, x2, y + h1 + ARROW_PAD, x3, y2 - ARROW_PAD)
    y = y2 + h2 + 0.1
    for text, size, weight, color in ex.footer:
        tt = Text(text, size, "center", weight, color, space_after=0.03)
        tt.draw(ax, x, y, width)
        y += tt.height(width)
    return y


def draw_two_up(left, right, shares_l, shares_r, path=None, dpi=200, width=W_H, gap=0.3):
    """Two models side by side, each as a 2 x 2 grid; same fonts as the landscape figures."""
    saved = dict(SIZE)
    SIZE.update(SIZE_H)
    try:
        with matplotlib.rc_context(RC):
            fig = plt.figure(figsize=(width, 30), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, width)
            ax.set_ylim(30, 0)
            ax.axis("off")
            half = (width - 2 * MARGIN_H - gap) / 2
            rects = []
            yl = _model_grid(ax, left, shares_l, MARGIN_H, 0.14, half, rects=rects)
            yr = _model_grid(ax, right, shares_r, MARGIN_H + half + gap, 0.14, half, rects=rects)
            ax.plot([width / 2] * 2, [0.2, max(yl, yr) - 0.05], color="#dddddd", lw=0.8)
            Ht = max(yl, yr) + 0.1
            fig.set_size_inches(width, Ht)
            ax.set_ylim(Ht, 0)
            fig._layout_issues = check_layout(fig, ax, rects)
            for msg in fig._layout_issues:
                print("LAYOUT:", msg)
            if path:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path.with_suffix(".png"), dpi=dpi, facecolor="white")
                fig.savefig(path.with_suffix(".pdf"), facecolor="white")
            plt.close(fig)
            return fig
    finally:
        SIZE.clear()
        SIZE.update(saved)


@dataclass
class CausalMask(Widget):
    """Who-reads-whom grid for causal attention: row i (a token) reads columns 0..i (itself and earlier tokens)."""
    n: int = 5
    cell: float = 0.17
    gap: float = 0.035
    top_label: str = "earlier  →  later"
    caption: str = "the last patch reads every earlier one"
    color: str = "#9ebcf0"
    space_after: float = 0.08

    def height(self, width):
        return 0.2 + self.n * (self.cell + self.gap) - self.gap + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        c, g, n = self.cell, self.gap, self.n
        gw = n * (c + g) - g
        gx = x + (width - gw) / 2
        ax.text(x + width / 2, y + 0.08, self.top_label, size=SIZE["small"], color=GREY, ha="center", va="center",
                style="italic")
        top = y + 0.2
        for i in range(n):
            for j in range(n):
                cx, cy = gx + j * (c + g), top + i * (c + g)
                if j <= i:
                    fill = INK if (i == n - 1 and j == n - 1) else (ACCENT["blue"] if i == n - 1 else self.color)
                    _box(ax, cx, cy, c, c, fill, fill, 0.6, r=0.03)
                else:
                    _box(ax, cx, cy, c, c, "white", "#c4c4c4", 0.6, r=0.03)
        ax.text(x + width / 2, top + n * (c + g) - g + 0.08, self.caption, size=SIZE["small"], color=GREY,
                ha="center", va="top", style="italic")


@dataclass
class StateChain(Widget):
    """Recurrent picture: one token per patch, a state circle under each, arrows carrying the state forward.
    Future tokens are dashed (fed in as missing values)."""
    n_past: int = 4
    n_future: int = 2
    top_label: str = "patch tokens"
    state_label: str = "state carried forward"
    future_label: str = "missing → forecast"
    reverse_row: bool = False     # add a second chain running right to left (TiRex-2 known-future rows)
    reverse_label: str = "reverse pass over known-ahead rows"
    space_after: float = 0.08

    def height(self, width):
        return 0.2 + 0.19 + 0.12 + 0.2 + 0.24 + (0.36 if self.reverse_row else 0) + self.space_after

    def draw(self, ax, x, y, width):
        t, n = 0.19, self.n_past + self.n_future
        step = min(0.36, (width - 0.1) / n)
        sx = x + (width - (n - 1) * step - t) / 2
        ax.text(x + width / 2, y + 0.08, self.top_label, size=SIZE["small"], color=GREY, ha="center", va="center")
        ty = y + 0.2
        cy = ty + t + 0.12 + 0.1
        r = 0.075
        cs = []
        for k in range(n):
            bx = sx + k * step
            fut = k >= self.n_past
            if fut:
                _box(ax, bx, ty, t, t, "white", ACCENT["blue"], 1.0, r=0.035, ls=(0, (2.5, 1.5)))
            else:
                _box(ax, bx, ty, t, t, "#d9e6f7", ACCENT["blue"], 1.0, r=0.035)
            ccx = bx + t / 2
            cs.append(ccx)
            ax.plot([ccx, ccx], [ty + t + 0.01, cy - r - 0.01], color="#8f8f8f", lw=0.8)
            ax.add_patch(matplotlib.patches.Circle((ccx, cy), r, fc="#cfe6da" if not fut else "#f6dcc8",
                                                   ec=ACCENT["green"] if not fut else ACCENT["orange"], lw=1.0))
        for a, b in zip(cs[:-1], cs[1:]):
            _arrow(ax, a + r + 0.01, cy, b - r - 0.01, cy, color=ACCENT["green"], lw=1.1)
        ly = cy + r + 0.05
        ax.text((cs[0] + cs[self.n_past - 1]) / 2, ly, self.state_label, size=SIZE["small"], color=GREY,
                ha="center", va="top", style="italic")
        if self.n_future:
            ax.text((cs[self.n_past] + cs[-1]) / 2, ty - 0.02, "", size=1)
            ax.text((cs[self.n_past] + cs[-1]) / 2, ly, self.future_label, size=SIZE["small"], color=ACCENT["orange"],
                    ha="center", va="top", style="italic")
        if self.reverse_row:
            ry = ly + 0.3
            for a, b in zip(cs[::-1][:-1], cs[::-1][1:]):
                _arrow(ax, a - 0.03, ry, b + 0.03, ry, color=ACCENT["green"], lw=1.0)
            ax.text(x + width / 2, ry + 0.06, self.reverse_label, size=SIZE["small"], color=GREY, ha="center",
                    va="top", style="italic")


@dataclass
class PatchLookahead(Widget):
    """TimesFM-3 token input: the row's own patch plus the next 64 steps, visible only for known-ahead rows."""
    rows: tuple = (("target", False), ("known", True))
    n_patch: int = 4                  # drawn cells for the 32-step patch (8 steps per cell)
    n_ahead: int = 8                  # drawn cells for the next 64 steps
    caption: str = "patch (32 steps) + next 64 steps"
    token_label: str = "one token: 1,280 numbers"
    space_after: float = 0.0

    def height(self, width):
        return 0.02 + len(self.rows) * 0.2 + 0.2 + 0.2 + 0.3 + 0.15 + 0.24 + self.space_after

    def draw(self, ax, x, y, width):
        lab_w = max(M.width(r, SIZE["small"]) for r, _ in self.rows) + 0.07
        n = self.n_patch + self.n_ahead
        g = 0.008
        c = min(0.1, (width - lab_w - 0.06 - g * (n - 1)) / n)
        px = x + lab_w
        for i, (lab, known) in enumerate(self.rows):
            ry = y + 0.02 + i * 0.2
            ax.text(px - 0.06, ry + c / 2, lab, size=SIZE["small"], color=GREY, ha="right", va="center")
            for k in range(n):
                cx = px + k * (c + g) + (0.06 if k >= self.n_patch else 0)
                if k < self.n_patch:
                    _cell(ax, cx, ry, c, "#3d3d3d" if not known else ACCENT["green"], lw=0.4)
                elif known:
                    _cell(ax, cx, ry, c, "#a9d4bd", lw=0.4)
                else:
                    ax.add_patch(FancyBboxPatch((cx, ry), c, c, boxstyle="round,pad=0,rounding_size=0.015",
                                                fc="white", ec="#a0a0a0", lw=0.5, ls=(0, (1.5, 1))))
        cy = y + 0.02 + len(self.rows) * 0.2 + 0.0
        p_mid = px + (self.n_patch * (c + g) - g) / 2
        a_mid = px + self.n_patch * (c + g) + 0.06 + (self.n_ahead * (c + g) - g) / 2
        ax.text(p_mid, cy, "patch", size=SIZE["small"], color=GREY, ha="center", va="top", style="italic")
        ax.text(a_mid, cy, "next 64 steps", size=SIZE["small"], color=GREY, ha="center", va="top", style="italic")
        a0 = cy + 0.22
        ax_x = x + 0.3
        _arrow(ax, ax_x, a0, ax_x, a0 + 0.27)
        ax.text(ax_x + 0.09, a0 + 0.135, "residual MLP", size=SIZE["small"], color=GREY, ha="left", va="center")
        ty = a0 + 0.3
        _box(ax, x + 0.05, ty, width - 0.1, 0.15, "#d9e6f7", ACCENT["blue"], 1.1, r=0.04)
        ax.text(x + width / 2, ty + 0.2, self.token_label, size=SIZE["small"] - 0.4, color=INK, ha="center",
                va="top", weight="bold")



def check_layout(fig, ax, rects, tol=0.012, inner=0.05):
    """Return a list of problems: text running out of its panel (or the figure) and text boxes that overlap.

    Works in inches (data units). A text belongs to the panel that holds its anchor point."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    W, H = fig.get_size_inches()
    boxes = []
    for t in ax.texts:
        if not t.get_text().strip() or not t.get_visible():
            continue
        bb = t.get_window_extent(r)
        (x0, y1), (x1, y0) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        ax_, ay_ = t.get_position()
        boxes.append((t.get_text(), x0, y0, x1, y1, ax_, ay_))
    issues = []
    for text, x0, y0, x1, y1, px, py in boxes:
        home = [rc for rc in rects if rc[0] <= px <= rc[0] + rc[2] and rc[1] <= py <= rc[1] + rc[3]]
        if home:
            rx, ry, rw, rh = home[0]
            if x0 < rx + inner or x1 > rx + rw - inner or y0 < ry + inner or y1 > ry + rh - inner:
                issues.append(f"outside its panel: {text[:50]!r}")
        elif x0 < -tol or x1 > W + tol or y0 < -tol or y1 > H + tol:
            issues.append(f"outside the figure: {text[:50]!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            ox = min(a[3], b[3]) - max(a[1], b[1])
            oy = min(a[4], b[4]) - max(a[2], b[2])
            if ox > 0.02 and oy > 0.03:
                issues.append(f"overlap: {a[0][:30]!r} / {b[0][:30]!r}")
    return issues


@dataclass
class QueryTokens(Widget):
    """Context patch tokens followed by learned forecast-query tokens (one learned vector, repeated)."""
    n_ctx: int = 4
    n_q: int = 3
    left_label: str = "context"
    right_label: str = "forecast queries"
    q_text: str = "Q"
    space_after: float = 0.08

    def height(self, width):
        return 0.19 + 0.06 + 0.2 + self.space_after

    def draw(self, ax, x, y, width):
        t, tg = 0.19, 0.05
        n = self.n_ctx + self.n_q
        total = n * (t + tg) - tg + 0.12
        sx = x + max(0.0, (width - total) / 2)
        c0 = sx
        for _ in range(self.n_ctx):
            _box(ax, sx, y, t, t, "#d9e6f7", ACCENT["blue"], 1.0, r=0.035)
            sx += t + tg
        c1 = sx - tg
        sx += 0.12
        q0 = sx
        for _ in range(self.n_q):
            _box(ax, sx, y, t, t, "#fbe3cf", ACCENT["orange"], 1.0, r=0.035)
            ax.text(sx + t / 2, y + t / 2, self.q_text, size=SIZE["small"] - 1.2, weight="bold", color=INK,
                    ha="center", va="center")
            sx += t + tg
        q1 = sx - tg
        ly = y + t + 0.05
        ax.text((c0 + c1) / 2, ly, self.left_label, size=SIZE["small"] - 0.4, color=GREY, ha="center", va="top",
                style="italic")
        ax.text((q0 + q1) / 2, ly, self.right_label, size=SIZE["small"] - 0.4, color=ACCENT["orange"],
                ha="center", va="top", style="italic")


@dataclass
class PatchBar(Widget):
    """A context bar cut into n patches, an arrow, then n tokens (optionally a frequency token in front)."""
    n: int = 8
    bar_label: str = "512 steps = 8 patches of 64"
    arrow_label: str = "linear layer"
    prefix: str = "freq"
    token_label: str = "8 tokens of 192 numbers"
    space_after: float = 0.0

    def height(self, width):
        return 0.18 + 0.16 + 0.22 + 0.3 + 0.2 + 0.24 + self.space_after

    def draw(self, ax, x, y, width):
        ax.text(x + width / 2, y + 0.07, self.bar_label, size=SIZE["small"], color=GREY, ha="center", va="center",
                style="italic")
        by = y + 0.18
        g = 0.03
        w = (width - 0.1 - g * (self.n - 1)) / self.n
        for k in range(self.n):
            ax.add_patch(FancyBboxPatch((x + 0.05 + k * (w + g), by), w, 0.16, boxstyle="round,pad=0,rounding_size=0.02",
                                        fc="#3d3d3d", ec="#3d3d3d", lw=0.5))
        a0 = by + 0.22
        ax_x = x + 0.3
        _arrow(ax, ax_x, a0, ax_x, a0 + 0.26)
        ax.text(ax_x + 0.09, a0 + 0.13, self.arrow_label, size=SIZE["small"], color=GREY, ha="left", va="center")
        ty = a0 + 0.3
        n_tok = self.n + (1 if self.prefix else 0)
        tw = min(0.17, (width - 0.1 - 0.035 * (n_tok - 1)) / n_tok)
        total = n_tok * (tw + 0.035) - 0.035
        sx = x + (width - total) / 2
        for k in range(n_tok):
            if self.prefix and k == 0:
                _box(ax, sx, ty, tw, tw, "#fbe3cf", ACCENT["orange"], 1.0, r=0.03)
            else:
                _box(ax, sx, ty, tw, tw, "#d9e6f7", ACCENT["blue"], 1.0, r=0.03)
            sx += tw + 0.035
        ax.text(x + width / 2, ty + tw + 0.05, self.token_label, size=SIZE["small"] - 0.4, color=INK, ha="center",
                va="top", weight="bold")


@dataclass
class MixerGrid(Widget):
    """MLP-mixer picture: a patches x features grid with a time-mixing band (row) and feature-mixing band (column)."""
    n_patch: int = 6
    n_feat: int = 4
    cell: float = 0.15
    gap: float = 0.035
    space_after: float = 0.08

    def height(self, width):
        return 0.2 + self.n_feat * (self.cell + self.gap) - self.gap + 0.08 + self.space_after

    def draw(self, ax, x, y, width):
        c, g = self.cell, self.gap
        gw = self.n_patch * (c + g) - g
        gx = x + (width - gw) / 2 + 0.2
        top = y + 0.2
        fr, fc = 1, 3
        gh = self.n_feat * (c + g) - g
        _box(ax, gx - 0.035, top + fr * (c + g) - 0.035, gw + 0.07, c + 0.07, "#cfe0fa", "none", 0, r=0.04)
        _box(ax, gx + fc * (c + g) - 0.035, top - 0.035, c + 0.07, gh + 0.07, "#f6c9c4", "none", 0, r=0.04)
        for i in range(self.n_feat):
            for j in range(self.n_patch):
                _box(ax, gx + j * (c + g), top + i * (c + g), c, c, "white", "#8f8f8f", 0.7, r=0.025)
        ax.text(gx + gw / 2, y + 0.08, "patches  →", size=SIZE["small"], color=GREY, ha="center", va="center",
                style="italic")
        ax.text(gx - 0.08, top + gh / 2, "features", size=SIZE["small"], color=GREY, ha="right", va="center",
                style="italic")


@dataclass
class BasisCurves(Widget):
    """Functional decoder picture: a few Legendre polynomials, weighted and summed into a smooth forecast curve."""
    h: float = 0.62
    caption: str = "weighted sum of curves"
    space_after: float = 0.06

    def height(self, width):
        return self.h + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        from numpy.polynomial import legendre as L
        t = np.linspace(-1, 1, 120)
        cols = ["#9ebcf0", "#a9d4bd", "#f0c59e", "#c9b8e6"]
        w1 = width * 0.55
        for k, col in enumerate(cols):
            cvec = np.zeros(k + 1); cvec[k] = 1
            v = L.legval(t, cvec)
            ax.plot(x + (t + 1) / 2 * w1, y + self.h / 2 - v * self.h * 0.42, color=col, lw=1.4)
        ax.text(x + w1 + 0.06, y + self.h / 2, "→", size=SIZE["body"], color=INK, ha="left", va="center")
        coef = [0.1, 0.45, -0.3, 0.2, 0.12]
        v = L.legval(t, coef)
        v = v / np.abs(v).max()
        x2 = x + w1 + 0.25
        ax.plot(x2 + (t + 1) / 2 * (width - (x2 - x)), y + self.h / 2 - v * self.h * 0.42, color="#b6501a", lw=2.0)
        ax.text(x + width / 2, y + self.h + 0.08, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


@dataclass
class OverlapPatches(Widget):
    """Overlapping patches (length 16, stride 8): context patches solid, future patches masked (dashed)."""
    n_ctx: int = 4
    n_fut: int = 2
    space_after: float = 0.08
    caption: str = "patches of 16, every 8 steps"

    def height(self, width):
        return 2 * 0.17 + 0.06 + 0.2 + self.space_after

    def draw(self, ax, x, y, width):
        n = self.n_ctx + self.n_fut
        unit = (width - 0.05) / (n + 1)        # one stride; a patch spans two strides
        for k in range(n):
            px = x + k * unit
            py = y + (k % 2) * 0.17
            fut = k >= self.n_ctx
            _box(ax, px, py, 2 * unit - 0.02, 0.13, "white" if fut else "#d9e6f7",
                 ACCENT["orange"] if fut else ACCENT["blue"], 1.0, r=0.03, ls=(0, (2.5, 1.5)) if fut else "-")
        ax.text(x + width / 2, y + 2 * 0.17 + 0.03, self.caption, size=SIZE["small"] - 0.4, color=GREY,
                ha="center", va="top", style="italic")


@dataclass
class TableGrid(Widget):
    """Forecasting as a table: one row per time step, feature columns, the target column; horizon rows have '?'."""
    cols: tuple = ("index", "sin", "cos", "day len.", "y")
    n_ctx: int = 4
    n_fut: int = 2
    space_after: float = 0.08

    def height(self, width):
        return 0.2 + (self.n_ctx + self.n_fut) * 0.16 + 0.05 + self.space_after

    def draw(self, ax, x, y, width):
        nc = len(self.cols)
        cw = min(0.42, (width - 0.05) / nc)
        gx = x + (width - nc * cw) / 2
        for j, c in enumerate(self.cols):
            ax.text(gx + j * cw + cw / 2, y + 0.08, c, size=SIZE["small"] - 1.0, color=GREY, ha="center",
                    va="center", weight="bold" if c == "y" else "normal")
        top = y + 0.18
        for i in range(self.n_ctx + self.n_fut):
            fut = i >= self.n_ctx
            ry = top + i * 0.16 + (0.05 if fut else 0)
            for j in range(nc):
                is_y = j == nc - 1
                if is_y and fut:
                    fc, ec = "white", ACCENT["orange"]
                elif is_y:
                    fc, ec = "#3d3d3d", "#3d3d3d"
                else:
                    fc, ec = ("#dcefe4", ACCENT["green"]) if fut else ("#eef3fb", "#9ab0d6")
                ax.add_patch(Rectangle((gx + j * cw + 0.01, ry), cw - 0.02, 0.13, fc=fc, ec=ec, lw=0.7))
                if is_y and fut:
                    ax.text(gx + j * cw + cw / 2, ry + 0.065, "?", size=SIZE["small"] - 1.2, color=ACCENT["orange"],
                            ha="center", va="center", weight="bold")
        ax.plot([gx - 0.02, gx + nc * cw + 0.02], [top + self.n_ctx * 0.16 + 0.015] * 2, color=LIGHT, lw=0.9,
                ls=(0, (3, 2)))


ROW_STYLE = {"target": ("#e3e3e3", "#2f2f2f"), "known": ("#dcefe4", ACCENT["green"]),
             "past": ("#f1e8d2", "#9a7b33")}


@dataclass
class FlatSequence(Widget):
    """Moirai input: every row's patch tokens laid end to end in one sequence. Target future = [mask] tokens,
    known-ahead future = real tokens, past-only rows stop at the forecast origin."""
    segments: tuple = (("target", 2, 1), ("known", 2, 1), ("past", 2, 0))
    labels: tuple = ("target", "known", "past")
    caption: str = "joined into one sequence"
    future_text: str = "M"          # what an unknown target patch shows: the mask token, or "?" (decoder-only)
    space_after: float = 0.08

    def height(self, width):
        return 0.17 + 0.19 + 0.06 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        n = sum(p + f for _, p, f in self.segments)
        sg, g = 0.09, 0.03
        t = min(0.19, (width - sg * (len(self.segments) - 1) - g * (n - 1)) / n)
        total = n * (t + g) - g + sg * (len(self.segments) - 1) - g * (len(self.segments) - 1)
        sx = x + (width - total) / 2
        ty = y + 0.17
        for (kind, n_p, n_f), lab in zip(self.segments, self.labels):
            fill, edge = ROW_STYLE[kind]
            s0 = sx
            for k in range(n_p + n_f):
                fut = k >= n_p
                if fut and kind == "target":
                    _box(ax, sx, ty, t, t, "white", ACCENT["orange"], 1.0, r=0.03, ls=(0, (2.5, 1.5)))
                    ax.text(sx + t / 2, ty + t / 2, self.future_text, size=SIZE["small"] - 2.2, weight="bold",
                            color=ACCENT["orange"], ha="center", va="center")
                else:
                    _box(ax, sx, ty, t, t, fill, edge, 1.0, r=0.03)
                sx += t + g
            ax.text((s0 + sx - g) / 2, y + 0.07, lab, size=SIZE["small"] - 0.8, color=edge, ha="center",
                    va="center", weight="bold")
            sx += sg - g
        ax.text(x + width / 2, ty + t + 0.06, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


@dataclass
class BlockAttention(Widget):
    """Who-reads-whom over a flattened sequence of several rows: cells of the same row in blue, of other rows in
    rose (two learned biases); with causal=True a token reads only tokens at the same or an earlier time."""
    n_rows: int = 3
    n_tok: int = 3
    causal: bool = False
    row_kinds: tuple = ("target", "known", "past")
    labels: tuple = ("same row", "other row")
    space_after: float = 0.08
    max_cell: float = 0.105

    def _cell(self, width):
        n = self.n_rows * self.n_tok
        return min(self.max_cell, (width - 0.3) / (n + 0.5))

    def height(self, width):
        c = self._cell(width)
        n = self.n_rows * self.n_tok
        return 0.08 + n * c + (self.n_rows - 1) * 0.03 + 0.1 + 0.19 + self.space_after

    def draw(self, ax, x, y, width):
        c = self._cell(width)
        n, bg = self.n_rows * self.n_tok, 0.03
        gw = n * c + (self.n_rows - 1) * bg
        gx = x + (width - gw) / 2 + 0.05
        gy = y + 0.08
        pos = lambda i: i * c + (i // self.n_tok) * bg
        for v in range(self.n_rows):           # row colour ticks on the left and on top
            fill, edge = ROW_STYLE[self.row_kinds[v % len(self.row_kinds)]]
            a, b = pos(v * self.n_tok), pos(v * self.n_tok + self.n_tok - 1) + c
            ax.plot([gx - 0.07] * 2, [gy + a + 0.01, gy + b - 0.01], color=edge, lw=2.6, solid_capstyle="butt")
            ax.plot([gx + a + 0.01, gx + b - 0.01], [gy - 0.05] * 2, color=edge, lw=2.6, solid_capstyle="butt")
        for i in range(n):
            vi, ti = divmod(i, self.n_tok)
            for j in range(n):
                vj, tj = divmod(j, self.n_tok)
                ok = (not self.causal) or tj <= ti
                cx, cy = gx + pos(j), gy + pos(i)
                if not ok:
                    _box(ax, cx + 0.008, cy + 0.008, c - 0.016, c - 0.016, "white", "#d0d0d0", 0.5, r=0.015)
                else:
                    col = "#9ebcf0" if vi == vj else "#f2b8b1"
                    _box(ax, cx + 0.008, cy + 0.008, c - 0.016, c - 0.016, col, col, 0.5, r=0.015)
        ly = gy + gw + 0.1 + 0.09
        items = [("#9ebcf0", self.labels[0]), ("#f2b8b1", self.labels[1])]
        tw = [0.2 + M.width(l, SIZE["small"] - 0.6) for _, l in items]
        lx = x + (width - sum(tw) - 0.15) / 2
        for (col, lab), w_ in zip(items, tw):
            _box(ax, lx, ly - 0.05, 0.11, 0.11, col, col, 0.5, r=0.02)
            ax.text(lx + 0.16, ly, lab, size=SIZE["small"] - 0.6, color=GREY, ha="left", va="center")
            lx += w_ + 0.15


@dataclass
class ExpertRouter(Widget):
    """Sparse mixture of experts: a token, a gate, a row of expert boxes of which k light up."""
    n_show: int = 7
    picked: tuple = (1, 4)
    gate_label: str = "gate"
    caption: str = "2 of 32 experts run for this token"
    space_after: float = 0.08

    def height(self, width):
        return 0.19 + 0.22 + 0.17 + 0.06 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        t = 0.19
        tx = x + width / 2 - t / 2
        _box(ax, tx, y, t, t, "#d9e6f7", ACCENT["blue"], 1.0, r=0.035)
        gy = y + t + 0.22
        gw = M.width(self.gate_label, SIZE["small"] - 0.6) + 0.16
        e = min(0.17, (width - 0.05 - 0.03 * (self.n_show - 1)) / self.n_show)
        ex = x + (width - (self.n_show * (e + 0.03) - 0.03)) / 2
        _arrow(ax, x + width / 2, y + t + 0.02, x + width / 2, gy - 0.02, lw=1.0)
        ax.text(x + width / 2 + 0.06, y + t + 0.11, self.gate_label, size=SIZE["small"] - 0.6, color=GREY,
                ha="left", va="center", style="italic")
        for k in range(self.n_show):
            bx = ex + k * (e + 0.03)
            if k == self.n_show - 2:
                ax.text(bx + e / 2, gy + e / 2, "…", size=SIZE["small"], color=GREY, ha="center", va="center")
                continue
            on = k in self.picked
            _box(ax, bx, gy, e, e, "#fbe3cf" if on else "white", ACCENT["orange"] if on else "#bdbdbd",
                 1.2 if on else 0.7, r=0.03)
        ax.text(x + width / 2, gy + e + 0.06, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")
        del gw


@dataclass
class MixtureCurve(Widget):
    """A predictive density drawn as a few thin component curves and their weighted sum (thick)."""
    h: float = 0.55
    n: int = 3
    caption: str = "a weighted mix of curves"
    space_after: float = 0.06

    def height(self, width):
        return self.h + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        z = np.linspace(-3.2, 4.2, 200)
        comps = [(0.45, -0.6, 0.6), (0.35, 0.7, 0.8), (0.2, 2.1, 0.9), (0.15, -1.6, 1.1)][: self.n]
        dens = [w * np.exp(-0.5 * ((z - m) / s) ** 2) / s for w, m, s in comps]
        tot = np.sum(dens, axis=0)
        top = tot.max()
        X = x + 0.05 + (z - z[0]) / (z[-1] - z[0]) * (width - 0.1)
        Y = lambda d: y + self.h - d / top * (self.h - 0.04)
        for d, col in zip(dens, ["#9ebcf0", "#a9d4bd", "#c9b8e6", "#f0c59e"]):
            ax.plot(X, Y(d), color=col, lw=1.2)
        ax.plot(X, Y(tot), color="#b6501a", lw=1.9)
        ax.plot([X[0], X[-1]], [y + self.h] * 2, color=LIGHT, lw=0.8)
        ax.text(x + width / 2, y + self.h + 0.08, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


@dataclass
class BranchFan(Widget):
    """Quantile feedback for long horizons: each quantile path of the first block is fed back and branches again;
    the quantiles are then taken across all branches."""
    h: float = 0.7
    k: int = 3
    caption: str = "9 paths × 9 = 81 → 9 quantiles"
    space_after: float = 0.06

    def height(self, width):
        return self.h + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        rng = np.random.default_rng(5)
        n = 24
        tp = np.arange(n)
        past = 0.45 * np.sin(2 * np.pi * tp / 12) + rng.normal(0, 0.06, n)
        x0, x1, x2 = x + 0.05, x + width * 0.36, x + width * 0.66
        lo, hi = -1.5, 1.5
        Y = lambda v: y + 0.03 + (hi - v) / (hi - lo) * (self.h - 0.06)
        xs = x0 + tp / (n - 1) * (x1 - x0)
        ax.plot(xs, [Y(v) for v in past], color="#4a4a4a", lw=1.0)
        ax.plot([x1, x1], [y, y + self.h], color=LIGHT, lw=0.9, ls=(0, (3, 2)))
        p0 = past[-1]
        offs = np.linspace(-0.75, 0.75, self.k)
        ends = []
        for o in offs:
            e = p0 + o
            ax.plot([x1, x2], [Y(p0), Y(e)], color="#b6501a" if o == 0 else "#eea070", lw=1.4 if o == 0 else 1.1)
            ends.append(e)
        x3 = x + width - 0.05
        for e in ends:
            for o in offs * 0.5:
                ax.plot([x2, x3], [Y(e), Y(e + o)], color="#f0b48a", lw=0.8)
        ax.text(x + width / 2, y + self.h + 0.08, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


@dataclass
class FlowSteps(Widget):
    """Flow matching at a glance: random noise, a few in-between curves, the final forecast path."""
    n: int = 4
    h: float = 0.36
    caption: str = "noise → path, step by step"
    space_after: float = 0.06

    def height(self, width):
        return self.h + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        rng = np.random.default_rng(11)
        m = 40
        tt = np.linspace(0, 1, m)
        target = 0.7 * np.sin(2 * np.pi * tt * 1.3) + 0.2 * tt
        noise = rng.normal(0, 0.7, m)
        aw = 0.16
        w = (width - (self.n - 1) * aw) / self.n
        for k in range(self.n):
            a = k / (self.n - 1)
            v = a * target + (1 - a) * noise
            bx = x + k * (w + aw)
            _box(ax, bx, y, w, self.h, "white", CHIP_EDGE, 0.7, r=0.03)
            X = bx + 0.03 + tt * (w - 0.06)
            Y = y + self.h / 2 - np.clip(v, -1.3, 1.3) / 1.3 * (self.h / 2 - 0.04)
            ax.plot(X, Y, color="#b6501a" if k == self.n - 1 else "#8f8f8f", lw=1.3 if k == self.n - 1 else 0.9)
            if k < self.n - 1:
                _arrow(ax, bx + w + 0.02, y + self.h / 2, bx + w + aw - 0.02, y + self.h / 2, lw=0.9)
        ax.text(x + width / 2, y + self.h + 0.08, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


@dataclass
class SizeRouter(Widget):
    """Kairos mixture-of-size patching: one 128-step segment, three patch sizes and two null experts; the router
    keeps three of the five and the kept sizes are summed."""
    sizes: tuple = (("128", 1), ("64", 2), ("32", 4))
    n_null: int = 2
    picked: tuple = (0, 2, 3)       # indices into sizes + nulls
    caption: str = "router keeps 3 of 5 per segment"
    top_label: str = "one 128-step segment"
    space_after: float = 0.08

    def height(self, width):
        return 0.17 + 0.15 + 0.08 + len(self.sizes) * 0.2 + 0.24 + 0.06 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        lab_w = max(M.width(s_, SIZE["small"] - 0.6) for s_, _ in self.sizes) + 0.08
        bx, bw = x + lab_w, width - lab_w - 0.02
        ax.text(bx + bw / 2, y + 0.07, self.top_label, size=SIZE["small"] - 0.6, color=GREY, ha="center",
                va="center", style="italic")
        y += 0.17
        ax.add_patch(FancyBboxPatch((bx, y), bw, 0.15, boxstyle="round,pad=0,rounding_size=0.02", fc="#3d3d3d",
                                    ec="#3d3d3d", lw=0.5))
        ry = y + 0.15 + 0.08
        for i, (lab, n) in enumerate(self.sizes):
            on = i in self.picked
            g = 0.025
            w = (bw - g * (n - 1)) / n
            for k in range(n):
                _box(ax, bx + k * (w + g), ry, w, 0.14, "#fbe3cf" if on else "white",
                     ACCENT["orange"] if on else "#bdbdbd", 1.1 if on else 0.7, r=0.02)
            ax.text(bx - 0.06, ry + 0.07, lab, size=SIZE["small"] - 0.6, color=INK if on else LIGHT, ha="right",
                    va="center", weight="bold" if on else "normal")
            ry += 0.2
        nw = (bw - 0.06 * (self.n_null - 1)) / self.n_null
        for k in range(self.n_null):
            on = len(self.sizes) + k in self.picked
            _box(ax, bx + k * (nw + 0.06), ry, nw, 0.17, "#f3f3f3" if on else "white", ACCENT["orange"] if on else
                 "#bdbdbd", 1.1 if on else 0.7, r=0.03, ls=(0, (2.5, 1.5)))
            ax.text(bx + k * (nw + 0.06) + nw / 2, ry + 0.085, "null", size=SIZE["small"] - 1.0,
                    color=INK if on else LIGHT, ha="center", va="center", style="italic")
        ax.text(x + width / 2, ry + 0.17 + 0.07, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


@dataclass
class LagToken(Widget):
    """Lag-Llama token: the value at step t is predicted from a token made of lagged values (arrows from past
    steps), date features and two scale statistics."""
    lags: tuple = (1, 3, 7, 14)
    parts: tuple = (("84 lags", 0.5, "#d9e6f7", ACCENT["blue"]), ("6 dates", 0.27, "#dcefe4", ACCENT["green"]),
                    ("2 stats", 0.23, "#fbe3cf", ACCENT["orange"]))
    caption: str = "one token for step t"
    space_after: float = 0.08

    def height(self, width):
        return 0.5 + 0.28 + 0.22 + 0.06 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        rng = np.random.default_rng(2)
        n = 18
        c, g = (width - 0.1) / n - 0.02, 0.02
        vals = 0.5 + 0.35 * np.sin(np.arange(n) / 2.2) + rng.normal(0, 0.06, n)
        base = y + 0.5
        t_idx = n - 1
        for k in range(n):
            bx = x + 0.05 + k * (c + g)
            hh = 0.12 + 0.3 * vals[k]
            fut = k == t_idx
            col = "#f6dcc8" if fut else ("#9ebcf0" if (t_idx - k) in self.lags else "#d6d6d6")
            ax.add_patch(Rectangle((bx, base - hh), c, hh, fc=col, ec=ACCENT["orange"] if fut else "none",
                                   lw=1.0, ls=(0, (2, 1.2)) if fut else "-"))
        ax.text(x + 0.05 + t_idx * (c + g) + c / 2, y + 0.02, "t", size=SIZE["small"] - 0.4, color=ACCENT["orange"],
                ha="center", va="top", weight="bold")
        ty = base + 0.28
        tw = width - 0.1
        tx = x + 0.05
        for lag in self.lags:
            cx = x + 0.05 + (t_idx - lag) * (c + g) + c / 2
            ax.plot([cx, tx + tw * 0.25], [base + 0.02, ty - 0.02], color=ACCENT["blue"], lw=0.7, alpha=0.8)
        for lab, frac, fill, edge in self.parts:
            w = tw * frac
            _box(ax, tx, ty, w - 0.02, 0.22, fill, edge, 1.0, r=0.03)
            ax.text(tx + (w - 0.02) / 2, ty + 0.11, lab, size=SIZE["small"] - 1.2, color=INK, ha="center",
                    va="center", weight="bold")
            tx += w
        ax.text(x + width / 2, ty + 0.22 + 0.06, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


def _mini_series(n=60, seed=4):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.012 * t + 0.25 * np.sin(2 * np.pi * t / 60)
    season = 0.32 * np.sin(2 * np.pi * t / 9)
    return trend, season + rng.normal(0, 0.07, n)


def _line(ax, x, y, w, h, v, color, lw=1.1, lo=None, hi=None):
    lo = v.min() if lo is None else lo
    hi = v.max() if hi is None else hi
    X = x + np.linspace(0, w, len(v))
    Y = y + h - (v - lo) / (hi - lo + 1e-9) * h
    ax.plot(X, Y, color=color, lw=lw)


@dataclass
class Decomp(Widget):
    """DLinear's split: the window (top), its moving-average trend and the remainder."""
    labels: tuple = ("window", "trend (moving average)", "remainder")
    row_h: float = 0.26
    space_after: float = 0.06

    def height(self, width):
        return 3 * (self.row_h + 0.2) + self.space_after

    def draw(self, ax, x, y, width):
        trend, rest = _mini_series()
        rows = [(trend + rest, "#3d3d3d"), (trend, ACCENT["blue"]), (rest, ACCENT["orange"])]
        for k, ((v, col), lab) in enumerate(zip(rows, self.labels)):
            ry = y + k * (self.row_h + 0.2)
            ax.text(x, ry + 0.07, lab, size=SIZE["small"] - 0.6, color=col if k else GREY, ha="left", va="center",
                    style="italic")
            _line(ax, x + 0.02, ry + 0.15, width - 0.04, self.row_h - 0.04, v, col, lw=1.2)


@dataclass
class DenseMap(Widget):
    """A linear (or dense) layer drawn as two columns of nodes joined all to all."""
    n_in: int = 7
    n_out: int = 4
    left_label: str = "window"
    right_label: str = "horizon"
    h: float = 0.7
    space_after: float = 0.06

    def height(self, width):
        return self.h + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        x0, x1 = x + width * 0.28, x + width * 0.72
        yi = [y + self.h * (k + 0.5) / self.n_in for k in range(self.n_in)]
        yo = [y + self.h * (k + 0.5) / self.n_out for k in range(self.n_out)]
        for a in yi:
            for b in yo:
                ax.plot([x0, x1], [a, b], color="#b9c9e6", lw=0.6)
        r = min(0.045, self.h / self.n_in / 2.4)
        for a in yi:
            ax.add_patch(matplotlib.patches.Circle((x0, a), r, fc="#d9e6f7", ec=ACCENT["blue"], lw=0.9))
        for b in yo:
            ax.add_patch(matplotlib.patches.Circle((x1, b), r * 1.15, fc="#fbe3cf", ec=ACCENT["orange"], lw=0.9))
        ax.text(x0, y + self.h + 0.08, self.left_label, size=SIZE["small"] - 0.4, color=GREY, ha="center", va="top",
                style="italic")
        ax.text(x1, y + self.h + 0.08, self.right_label, size=SIZE["small"] - 0.4, color=ACCENT["orange"],
                ha="center", va="top", style="italic")


@dataclass
class PooledRows(Widget):
    """N-HiTS: the same window max-pooled with a large, a medium and no kernel, one row per stack."""
    labels: tuple = ("stack 1", "stack 2", "stack 3")
    kernels: tuple = (20, 5, 1)
    row_h: float = 0.24
    space_after: float = 0.06

    def height(self, width):
        return len(self.kernels) * (self.row_h + 0.1) + self.space_after

    def draw(self, ax, x, y, width):
        trend, rest = _mini_series(80, 6)
        v = trend + rest
        lab_w = max(M.width(l, SIZE["small"] - 0.6) for l in self.labels) + 0.08
        lo, hi = v.min(), v.max()
        for k, (kern, lab) in enumerate(zip(self.kernels, self.labels)):
            ry = y + k * (self.row_h + 0.1)
            m = int(np.ceil(len(v) / kern))
            pooled = np.array([v[i * kern:(i + 1) * kern].max() for i in range(m)])
            ax.text(x, ry + self.row_h / 2, lab, size=SIZE["small"] - 0.6, color=GREY, ha="left", va="center")
            if kern == 1:
                _line(ax, x + lab_w, ry, width - lab_w, self.row_h, pooled, "#3d3d3d", lo=lo, hi=hi)
            else:
                xs = x + lab_w + np.linspace(0, width - lab_w, m + 1)
                for i in range(m):
                    yy = ry + self.row_h - (pooled[i] - lo) / (hi - lo) * self.row_h
                    ax.plot([xs[i], xs[i + 1]], [yy, yy], color=ACCENT["blue"], lw=1.6)


@dataclass
class CoarseInterp(Widget):
    """A few forecast points (dots) and the smooth curve interpolated through them."""
    n_pts: int = 5
    h: float = 0.5
    caption: str = "few points → interpolated forecast"
    space_after: float = 0.06

    def height(self, width):
        return self.h + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        t = np.linspace(0, 1, 100)
        f = lambda s: 0.5 + 0.35 * np.sin(2 * np.pi * s * 1.1) + 0.1 * s
        tp = np.linspace(0, 1, self.n_pts)
        X = lambda s: x + 0.05 + s * (width - 0.1)
        Y = lambda v: y + self.h - v * self.h
        ax.plot(X(t), Y(np.interp(t, tp, f(tp))), color="#b6501a", lw=1.5)
        ax.scatter(X(tp), Y(f(tp)), s=16, color=ACCENT["blue"], zorder=3)
        ax.text(x + width / 2, y + self.h + 0.08, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")


@dataclass
class VarWeights(Widget):
    """Variable selection: one bar per input, its learned weight at this step."""
    items: tuple = (("temperature", 0.5, "#2f2f2f"), ("day length", 0.3, ACCENT["green"]),
                    ("NAO index", 0.2, "#9a7b33"))
    caption: str = "weights at one step (sum to 1)"
    space_after: float = 0.06

    def height(self, width):
        return len(self.items) * 0.22 + 0.06 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        lab_w = max(M.width(l, SIZE["small"] - 0.6) for l, _, _ in self.items) + 0.1
        for k, (lab, w, col) in enumerate(self.items):
            ry = y + k * 0.22
            ax.text(x + lab_w - 0.06, ry + 0.08, lab, size=SIZE["small"] - 0.6, color=GREY, ha="right", va="center")
            ax.add_patch(Rectangle((x + lab_w, ry + 0.02), (width - lab_w - 0.05) * w / 0.55, 0.13, fc=col,
                                   ec="none", alpha=0.75))
        ax.text(x + width / 2, y + len(self.items) * 0.22 + 0.06, self.caption, size=SIZE["small"] - 0.4,
                color=GREY, ha="center", va="top", style="italic")


@dataclass
class TreeRow(Widget):
    """Boosting: small trees in a row joined by plus signs; each one fits what the earlier ones missed."""
    n: int = 3
    caption: str = "each tree corrects the ones before"
    space_after: float = 0.06

    def height(self, width):
        return 0.5 + 0.08 + 0.18 + self.space_after

    def draw(self, ax, x, y, width):
        pw = 0.18
        tw = (width - (self.n - 1) * pw - 0.3) / self.n
        cx = x
        for k in range(self.n):
            mid = cx + tw / 2
            lv = [(mid, y + 0.05)]
            l1 = [(mid - tw * 0.25, y + 0.24), (mid + tw * 0.25, y + 0.24)]
            l2 = [(mid - tw * 0.38, y + 0.44), (mid - tw * 0.12, y + 0.44), (mid + tw * 0.12, y + 0.44),
                  (mid + tw * 0.38, y + 0.44)]
            for a in l1:
                ax.plot([lv[0][0], a[0]], [lv[0][1], a[1]], color="#8f8f8f", lw=0.9)
            for i, b in enumerate(l2):
                a = l1[i // 2]
                ax.plot([a[0], b[0]], [a[1], b[1]], color="#8f8f8f", lw=0.9)
            for (px, py) in lv + l1:
                ax.add_patch(matplotlib.patches.Circle((px, py), 0.035, fc="#d9e6f7", ec=ACCENT["blue"], lw=0.9))
            for (px, py) in l2:
                ax.add_patch(matplotlib.patches.Circle((px, py), 0.035, fc="#dcefe4", ec=ACCENT["green"], lw=0.9))
            cx += tw
            if k < self.n - 1:
                ax.text(cx + pw / 2, y + 0.25, "+", size=SIZE["body"], color=INK, ha="center", va="center",
                        weight="bold")
                cx += pw
        ax.text(cx + 0.15, y + 0.25, "…", size=SIZE["body"], color=GREY, ha="center", va="center")
        ax.text(x + width / 2, y + 0.5 + 0.08, self.caption, size=SIZE["small"] - 0.4, color=GREY, ha="center",
                va="top", style="italic")
