import textwrap

import typer

from matisse_pipeline.cli import calibrate, format_results, reduce, show

app = typer.Typer(help="MATISSE Data Reduction CLI")

app.command(name="reduce")(reduce.reduce)
app.command(name="calibrate")(calibrate.calibrate)
app.command(name="show")(show.show)

doc = textwrap.dedent(format_results.format_results.__doc__ or "").strip()
doc_wrapped = textwrap.fill(doc, width=88)
app.command(
    name="format",
    help="Format reduced data into OIFITS files using FITS metadata.",
    epilog=doc_wrapped,
)(format_results.format_results)


# -------------------------
# Main entrypoint
# -------------------------
def main():
    """CLI entrypoint for MATISSE pipeline."""
    app()


if __name__ == "__main__":
    main()
