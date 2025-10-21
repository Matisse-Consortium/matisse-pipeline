import subprocess

from typer.testing import CliRunner

from matisse_pipeline.cli.main import app

runner = CliRunner()


def test_cli_help():
    """Ensure the CLI responds correctly to --help."""
    result = subprocess.run(
        ["matisse", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Usage" in result.stdout


def test_reduce_empty_directory(tmp_path, caplog):
    """
    Ensure 'matisse reduce' exits cleanly with an error
    when executed in an empty directory (no raw data files).
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(
        app, ["reduce", "--datadir", str(empty_dir)], catch_exceptions=False
    )

    caplog.set_level("ERROR", logger="matisse")

    # Should terminate with a non-zero exit code
    assert result.exit_code in (1, 2), f"Unexpected exit code: {result.exit_code}"

    # Collect all error messages
    error_messages = [
        rec.message.lower() for rec in caplog.records if rec.levelname == "ERROR"
    ]
    assert any(
        "no fits files found in the provided raw" in msg for msg in error_messages
    ), f"Expected error not found in logs: {error_messages}"


def test_reduce_with_one_file(tmp_path):
    """
    Ensure 'matisse reduce' runs successfully when the directory contains one file.
    In this case, the file is a fake MATISSE one and therefore, no esorex command are
    proceed (as expected).
    """
    # --- Create a fake FITS file in the directory ---
    datadir = tmp_path / "data"
    datadir.mkdir()
    fits_file = datadir / "MATIS_test_file.fits"
    fits_file.write_text("FAKE DATA")  # content doesn't matter for the test

    result = runner.invoke(
        app, ["reduce", "--datadir", str(datadir)], catch_exceptions=False
    )

    # --- Assertions ---
    assert result.exit_code == 0, f"Unexpected exit code: {result.exit_code}"
    assert "[SUCCESS]" in result.stdout, f"Missing success message: {result.stdout}"
    assert "results" in result.stdout.lower()
