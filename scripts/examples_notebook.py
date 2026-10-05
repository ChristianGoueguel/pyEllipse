"""
Regenerate examples.ipynb from the "Usage Examples" section of README.md.

Each Python code block of the section becomes a code cell and the text between them a
markdown cell (images are left out, since the executed cells show the figures). The
notebook is then executed, so it contains the outputs of the current version.

Usage, from any directory, with pyEllipse, scikit-learn, nbformat, nbclient and
ipykernel installed:
    python scripts/examples_notebook.py
"""
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent

readme = (ROOT / "README.md").read_text(encoding="utf-8")
section = readme[readme.index("## Usage Examples"):readme.index("## Key Differences")]

intro = (
    "# pyEllipse examples\n\n"
    "The examples of the README, with their outputs. The "
    "[documentation](https://christiangoueguel.com/pyEllipse) has more, with the theory "
    "behind each function.\n\n"
    "This notebook is generated from README.md by `scripts/examples_notebook.py`."
)
cells = [nbformat.v4.new_markdown_cell(intro)]
for i, part in enumerate(re.split(r"```python\n(.*?)```", section, flags=re.S)):
    if i % 2:
        cells.append(nbformat.v4.new_code_cell(part.rstrip("\n")))
        continue
    lines = [
        line for line in part.splitlines()
        if not line.startswith(("## Usage Examples", "![", "<details>", "</details>", "<summary>"))
    ]
    text = "\n".join(lines).strip()
    if text:
        cells.append(nbformat.v4.new_markdown_cell(text))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
NotebookClient(nb, timeout=600, kernel_name="python3",
               resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.write(nb, ROOT / "examples.ipynb")
print(f"Wrote examples.ipynb ({len(cells)} cells)")
