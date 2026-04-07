from __future__ import annotations

import shutil
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parent
SOURCE_PATH = ROOT / "docs" / "MIRA-Production-Guide.md"
MARKDOWN_OUTPUT_PATH = ROOT / "static" / "docs" / "MIRA-Production-Guide.md"
OUTPUT_PATH = ROOT / "static" / "docs" / "mira-production-guide.pdf"

PAGE_SIZE = (8.27, 11.69)
LEFT = 0.08
TOP = 0.95
BOTTOM = 0.06
LINE_HEIGHT = 0.023
BODY_FONT = 10
HEADING_FONT = 17
SUBHEADING_FONT = 12
WRAP_WIDTH = 95


def build_blocks(markdown_text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.rstrip()
        if not line:
            blocks.append(("blank", ""))
            continue
        if line.startswith("# "):
            blocks.append(("h1", line[2:].strip()))
            continue
        if line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            continue
        if line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            continue
        blocks.append(("body", line))
    return blocks


def wrap_line(kind: str, text: str) -> list[tuple[str, str]]:
    if kind == "blank":
        return [(kind, "")]
    width = 70 if kind == "h1" else 85 if kind.startswith("h") else WRAP_WIDTH
    wrapped = textwrap.wrap(text, width=width, replace_whitespace=False, drop_whitespace=False)
    return [(kind, line.strip()) for line in wrapped] or [(kind, text)]


def render_pdf(blocks: list[tuple[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(OUTPUT_PATH) as pdf:
        fig = None
        ax = None
        y = TOP

        def new_page() -> None:
            nonlocal fig, ax, y
            if fig is not None:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            fig, ax = plt.subplots(figsize=PAGE_SIZE)
            ax.set_axis_off()
            y = TOP

        def ensure_space(lines_needed: int) -> None:
            nonlocal y
            required = lines_needed * LINE_HEIGHT
            if y - required < BOTTOM:
                new_page()

        new_page()

        for kind, text in blocks:
            wrapped_lines = wrap_line(kind, text)
            ensure_space(len(wrapped_lines) + (1 if kind in {"h1", "h2", "h3"} else 0))

            for wrapped_kind, wrapped_text in wrapped_lines:
                if wrapped_kind == "blank":
                    y -= LINE_HEIGHT * 0.65
                    continue

                fontsize = BODY_FONT
                weight = "normal"
                color = "#12313c"

                if wrapped_kind == "h1":
                    fontsize = HEADING_FONT
                    weight = "bold"
                    color = "#0b7285"
                elif wrapped_kind == "h2":
                    fontsize = SUBHEADING_FONT
                    weight = "bold"
                    color = "#0a5c6b"
                elif wrapped_kind == "h3":
                    fontsize = 11
                    weight = "bold"
                    color = "#1f4955"

                ax.text(
                    LEFT,
                    y,
                    wrapped_text,
                    fontsize=fontsize,
                    fontweight=weight,
                    color=color,
                    va="top",
                    ha="left",
                    family="DejaVu Sans",
                )
                y -= LINE_HEIGHT * (1.25 if wrapped_kind == "h1" else 1.0)

            if kind in {"h1", "h2", "h3"}:
                y -= LINE_HEIGHT * 0.2

        if fig is not None:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    MARKDOWN_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_PATH, MARKDOWN_OUTPUT_PATH)
    markdown_text = SOURCE_PATH.read_text(encoding="utf-8")
    render_pdf(build_blocks(markdown_text))
    print(f"Generated {OUTPUT_PATH}")

