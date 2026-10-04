"""
app/report/generate_pdf.py
==========================
Generates 1-page situation report (English and Nepali) via Jinja2 and WeasyPrint.
Every single number rendered in the template is pulled strictly from results.json.
"""
from __future__ import annotations

import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

from pipeline.schema import SpaceTrackResults

logger = logging.getLogger("spacetrack.report")

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


def generate_situation_report(
    results: SpaceTrackResults,
    output_dir: Path,
    languages: tuple[str, ...] = ("en", "ne"),
) -> dict[str, Path]:
    """
    Renders reports for requested languages.
    Generates HTML and PDF versions.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = {}

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))

    for lang in languages:
        template_name = f"{lang}/report.html"
        try:
            template = env.get_template(template_name)
        except Exception as e:
            logger.warning(f"Could not load template {template_name}: {e}")
            continue

        html_out = template.render(res=results)

        html_path = output_dir / f"situation_report_{lang}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_out)
        generated_files[f"{lang}_html"] = html_path

        pdf_path = output_dir / f"situation_report_{lang}.pdf"
        try:
            from weasyprint import HTML
            HTML(string=html_out).write_pdf(str(pdf_path))
            generated_files[f"{lang}_pdf"] = pdf_path
            logger.info(f"Generated PDF situation report: {pdf_path}")
        except Exception as e:
            logger.warning(
                f"WeasyPrint PDF rendering unavailable in current host ({e}). "
                f"HTML situation report saved at {html_path}."
            )

    return generated_files
