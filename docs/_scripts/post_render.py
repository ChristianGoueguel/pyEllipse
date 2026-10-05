"""
Quarto post-render step.

Redirects the URLs of the former pdoc documentation (pyEllipse <= 0.1.5) to the pages
that replace them. GitHub Pages has no server-side redirects, so each former URL gets a
small HTML page that forwards to the new one.
"""
import os
from pathlib import Path

SITE_URL = "https://christiangoueguel.com/pyEllipse/"

# Former page -> new page, relative to the root of the site
REDIRECTS = {
    "pyEllipse.html": "index.html",
    "pyEllipse/hotelling_parameters.html": "reference/hotelling_parameters.html",
    "pyEllipse/hotelling_coordinates.html": "reference/hotelling_coordinates.html",
    "pyEllipse/confidence_ellipse.html": "reference/confidence_ellipse.html",
}

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Page moved</title>
<meta name="robots" content="noindex">
<link rel="canonical" href="{url}">
<meta http-equiv="refresh" content="0; url={url}">
</head>
<body>
<p>This page has moved to <a href="{url}">{url}</a>.</p>
</body>
</html>
"""

site = Path(os.environ.get("QUARTO_PROJECT_OUTPUT_DIR", "_site"))
for old, new in REDIRECTS.items():
    page = site / old
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text(TEMPLATE.format(url=SITE_URL + new), encoding="utf-8")
