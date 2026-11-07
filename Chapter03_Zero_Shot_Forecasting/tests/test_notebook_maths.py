"""Formulas in the notebooks must render in every viewer (Colab, Jupyter, Cursor/VS Code, GitHub).

MathJax builds a tall square root, and the braces of \\underbrace and \\overbrace, from private-use glyphs of its
size fonts. Viewers that cannot draw those glyphs show empty boxes and a broken root, so the notebooks keep every
root short (s = \\sqrt{s^2}, \\sqrt{a/b}) and use no horizontal braces. They also avoid commands that KaTeX
knows but plain MathJax does not, such as \\thickspace, and calligraphic, blackboard and bold letters, which
some systems draw as empty boxes. A root inside a fraction is set in the smaller text style and stays short, so it
is allowed."""
import json
import re

import pytest
from conftest import ROOT

NOTEBOOKS = sorted(ROOT.glob("*.ipynb")) + sorted((ROOT / "bonus").glob("*.ipynb"))
TALL = ("\\frac", "\\dfrac", "\\tfrac", "\\sum", "\\prod", "\\int")
MISSING = ("\\thickspace", "\\medspace")   # KaTeX knows them, plain MathJax 2 and 3 do not
# letter styles that viewers draw from special font tables or Unicode's mathematical alphabets, which some systems
# lack (empty boxes); plain, upright (\\mathrm) and \\text letters are safe
SPECIAL = ("\\mathcal", "\\mathbb", "\\mathfrak", "\\mathscr", "\\mathbf", "\\boldsymbol", "\\bm", "\\imath", "\\jmath")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.S)
_MATH = re.compile(r"\$\$(.+?)\$\$|(?<![\\$])\$(?!\$)(.+?)(?<![\\])\$", re.S)


def _argument(tex, i):
    """End index of the TeX argument that starts at i (a {...} group or a single token)."""
    while i < len(tex) and tex[i] == " ":
        i += 1
    if i >= len(tex):
        return i
    if tex[i] == "{":
        depth = 0
        for j in range(i, len(tex)):
            depth += {"{": 1, "}": -1}.get(tex[j], 0)
            if depth == 0:
                return j + 1
        return len(tex)
    m = re.match(r"\\[A-Za-z]+|\\.|.", tex[i:], re.S)
    return i + len(m.group(0))


def problems(tex):
    out = []
    if "\\underbrace" in tex or "\\overbrace" in tex:
        out.append("horizontal brace")
    out += [f"{m} is not defined in MathJax" for m in MISSING if re.search(re.escape(m) + r"(?![A-Za-z])", tex)]
    out += [f"{m} needs a special font" for m in SPECIAL if re.search(re.escape(m) + r"(?![A-Za-z])", tex)]
    inside_frac = []
    for m in re.finditer(r"\\[dt]?frac(?![A-Za-z])", tex):
        a = _argument(tex, m.end())
        b = _argument(tex, a)
        inside_frac.append((m.end(), b))
    for m in re.finditer(r"\\sqrt(?![A-Za-z])", tex):
        start = m.end()
        if tex[start:start + 1] == "[":
            start = tex.index("]", start) + 1
        end = _argument(tex, start)
        arg = tex[start:end]
        if any(t in arg for t in TALL) and not any(a <= m.start() < b for a, b in inside_frac):
            out.append(f"tall root \\sqrt{arg[:60]}")
    return out


def math_spans(path):
    nb = json.loads(path.read_text())
    for k, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        text = _CODE.sub(" ", "".join(cell["source"]))
        for m in _MATH.finditer(text):
            yield k, m.group(1) or m.group(2)


def test_checker_flags_what_breaks():
    assert problems(r"s = \sqrt{\frac{1}{n-1}\sum_{t=1}^{n}(x_t-\bar x)^2}")
    assert problems(r"H = \underbrace{a + b}_{c}")
    assert problems(r"a \thickspace b")
    assert problems(r"\mathcal C(x)") and problems(r"\mathbb E[x]") and problems(r"\mathbf y_t")
    assert not problems(r"s = \sqrt{s^2}, \quad CD = q_\alpha \sqrt{M(M+1)/(6N)}, \ a \negthickspace b")
    assert not problems(r"U = \frac{\sqrt{\frac1H\sum_t e_t^2}}{\sqrt{\frac1H\sum_t y_t^2}}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_formulas_need_no_assembled_glyphs(path):
    bad = [(cell, p) for cell, tex in math_spans(path) for p in problems(tex)]
    assert not bad, bad
