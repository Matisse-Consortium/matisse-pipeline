"""
MATISSE visualisation interface based on mat_showOiData.py.
"""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio
import typer

from matisse_pipeline.core.utils.oifits_reader import open_oifits
from matisse_pipeline.viewer import viewer_plotly


def show(
    file: str = typer.Argument(..., help="Path to the OIFITS file"),
    save: str | None = typer.Option(
        None,
        "--save",
        help="Output filename (.png or .pdf)",
    ),
):
    """
    Display MATISSE OIFITS data (Step 1: fake data; Step 2: legacy parser integration).
    """
    typer.echo("🔭 Launching MATISSE OIFITS viewer...")

    data = open_oifits(file)

    # Static mode
    fig = viewer_plotly.make_static_matisse_plot(data)
    if save:
        path = Path(save)
        ext = path.suffix.lower()
        if ext not in {".png", ".pdf"}:
            typer.echo(f"❌ Unsupported output format: {ext}. Use .png or .pdf.")
            raise typer.Exit(code=1)
        pio.write_image(fig, path, engine="kaleido")
        typer.echo(f"💾 Figure saved as {save}")
    else:
        typer.echo("👁️ Opening static figure...")
        viewer_plotly.show_plot(fig)
