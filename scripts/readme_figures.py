"""
Regenerate the README figures from the README's own code.

Runs the Python code blocks of README.md in order, in one namespace, and saves the
figure shown by each `plt.show()` to images/, under the names in FIGURES, in order.
The figure style block of the README must be the one of the documentation
(docs/_figure-style.qmd), so that both produce the same figures.

Usage, from any directory:
    python scripts/readme_figures.py
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ["example1_hotelling_ellipse.png", "grouped_ellipses.png", "3d_ellipsoids.png"]
DPI = 300

readme_blocks = re.findall(
    r"```python\n(.*?)```", (ROOT / "README.md").read_text(encoding="utf-8"), re.S
)
style = re.search(
    r"```\{python\}\n(.*?)```",
    (ROOT / "docs" / "_figure-style.qmd").read_text(encoding="utf-8"),
    re.S,
).group(1)
style = "".join(line for line in style.splitlines(keepends=True) if not line.startswith("#|"))
if style not in readme_blocks:
    raise SystemExit("The figure style in README.md differs from docs/_figure-style.qmd.")

saved = []


def show(*args, **kwargs):
    """Save the current figure under the next name of FIGURES instead of showing it."""
    fig = plt.gcf()
    path = ROOT / "images" / FIGURES[len(saved)]
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    saved.append(path)
    print(f"Saved {path.relative_to(ROOT)}")


plt.show = show
namespace = {"__name__": "__main__"}
for block in readme_blocks:
    exec(compile(block, "README.md", "exec"), namespace)

if len(saved) != len(FIGURES):
    raise SystemExit(f"Expected {len(FIGURES)} figures, saved {len(saved)}.")
