"""
Quarto pre-render step.

1. Writes _variables.yml with the package version and the git commit the site is
   built from, shown in the footer of every page.
2. Generates the API reference pages from the docstrings with quartodoc.
"""
import re
import subprocess
from pathlib import Path

from quartodoc import Builder

DOCS = Path(__file__).resolve().parent.parent
REPO = "https://github.com/ChristianGoueguel/pyEllipse"


def package_version() -> str:
    init = (DOCS.parent / "pyEllipse" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__ = "([^"]+)"', init, re.MULTILINE).group(1)


def git_commit() -> str:
    """Short hash of HEAD, with a -dirty suffix when tracked files have uncommitted changes."""
    try:
        return subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7", "--exclude", "*"],
            cwd=DOCS, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


commit = git_commit()
sha = commit.removesuffix("-dirty")
(DOCS / "_variables.yml").write_text(
    f'version: "{package_version()}"\n'
    f'commit: "{commit}"\n'
    f'commit_url: "{REPO}/commit/{sha}"\n',
    encoding="utf-8",
)

Builder.from_quarto_config(str(DOCS / "_quarto.yml")).build()
