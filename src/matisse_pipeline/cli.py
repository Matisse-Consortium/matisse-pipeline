import typer

app = typer.Typer(help="MATISSE data-reduction toolkit")


# -------------------------
# Subcommand: reduce
# -------------------------
@app.command()
def reduce(raw_dir: str, calib_dir: str | None = None):
    """
    Run the automatic reduction pipeline.

    raw_dir: Path to the directory containing raw MATISSE data.
    calib_dir: Optional path to calibration data (raw or processed).
    """
    typer.echo(f"Running MATISSE pipeline on {raw_dir} (calib: {calib_dir})")


# -------------------------
# Subcommand: calibrate
# -------------------------
@app.command()
def calibrate(red_dir: str):
    """
    Apply the transfer function to the data (use of calibrator(s))

    red_dir: Path to the directory containing reduced MATISSE data.
    """
    typer.echo(f"Running calibration on {red_dir}")


# -------------------------
# Main entrypoint
# -------------------------
def main():
    """CLI entrypoint for the MATISSE pipeline."""
    app()  # Do not call reduce() directly — just invoke the app.


if __name__ == "__main__":
    main()
