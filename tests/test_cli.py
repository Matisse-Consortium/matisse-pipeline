import re
import subprocess

import pytest
from typer.testing import CliRunner

from matisse_pipeline.cli import format_results as format_module, show as show_module
from matisse_pipeline.cli.main import app
from matisse_pipeline.core.utils.oifits_reader import OIFitsReader

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
        app, ["reduce", "--data-dir", str(empty_dir)], catch_exceptions=False
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


def test_reduce_with_one_file(tmp_path, caplog):
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

    with caplog.at_level("INFO"):
        result = runner.invoke(
            app, ["reduce", "--data-dir", str(datadir)], catch_exceptions=False
        )

    # --- Assertions ---
    assert result.exit_code == 0, f"Unexpected exit code: {result.exit_code}"
    assert any("[SUCCESS]" in rec.message for rec in caplog.records), (
        f"Missing success message in logs: {[r.message for r in caplog.records]}"
    )


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes for consistent test output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_reduce_no_good_res(tmp_path):
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
        app,
        ["reduce", "--data-dir", str(datadir), "--resol", "bad_res"],
        catch_exceptions=False,
        color=False,
    )
    clean_output = strip_ansi(result.output)
    assert result.exit_code != 0
    assert "Invalid value for '--resol'" in clean_output


def test_matisse_format_with_fake_file(tmp_path, caplog):
    """
    Ensure `matisse format` runs successfully on a directory containing a fake FITS file.

    The fake file should not trigger any real FITS parsing logic,
    but the command should complete gracefully and log a success message.
    """
    # --- Setup a temporary fake dataset ---
    datadir = tmp_path / "Iter1"
    datadir.mkdir()
    fake_fits = datadir / "fake_science_file.fits"
    fake_fits.write_text("FAKE DATA")  # content doesn't matter for this test

    # --- Run the CLI command ---
    with caplog.at_level("INFO"):
        result = runner.invoke(
            app,
            ["format", str(datadir)],
            catch_exceptions=False,
        )

    # --- Assertions ---
    assert result.exit_code == 0, (
        f"Unexpected exit code: {result.exit_code}\n{result.stdout}"
    )
    assert any("Tidyup complete" in rec.message for rec in caplog.records), (
        f"Missing success message in logs: {[r.message for r in caplog.records]}"
    )


def test_format_cli_missing_directory(tmp_path, caplog):
    missing_dir = tmp_path / "does_not_exist"

    with caplog.at_level("ERROR"):
        result = runner.invoke(
            app,
            ["format", str(missing_dir)],
            catch_exceptions=False,
        )

    assert result.exit_code == 1
    assert any(str(missing_dir) in record.message for record in caplog.records), (
        "Expected error log referencing missing directory."
    )


def test_format_cli_invokes_tidyup_with_verbose_flag(tmp_path, monkeypatch):
    target_dir = tmp_path / "Iter3"
    target_dir.mkdir()
    # File content is irrelevant; command should only forward the path.
    (target_dir / "placeholder.fits").write_text("DATA")

    calls = {"tidyup": None, "verbosity": []}

    def fake_tidyup(path):
        calls["tidyup"] = path

    def fake_set_verbosity(logger, verbose):
        calls["verbosity"].append(verbose)

    monkeypatch.setattr(format_module, "tidyup_path", fake_tidyup)
    monkeypatch.setattr(format_module, "set_verbosity", fake_set_verbosity)

    result = runner.invoke(
        app,
        ["format", "-v", str(target_dir)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert calls["tidyup"] == target_dir
    assert calls["verbosity"] == [True]


def test_show_cli_opens_viewer(monkeypatch, real_oifits):
    captured = {}

    def fake_show(fig):
        captured["fig"] = fig
        return fig

    monkeypatch.setattr(show_module.viewer_plotly, "show_plot", fake_show)

    result = runner.invoke(app, ["show", str(real_oifits)], catch_exceptions=False)

    assert result.exit_code == 0
    assert "fig" in captured

    fig = captured["fig"]
    assert len(fig.data) > 0
    spectrum_traces = [
        trace for trace in fig.data if getattr(trace, "legendgroup", None) == "spectre"
    ]
    assert spectrum_traces, "Expected at least one spectrum trace"
    assert all(len(getattr(trace, "x", [])) for trace in spectrum_traces)

    closure_traces = [
        trace for trace in fig.data if getattr(trace, "legendgroup", None) == "CPHASE"
    ]
    assert closure_traces, "Expected closure phase traces"
    unique_closures = {getattr(trace, "name", None) for trace in closure_traces}
    unique_closures.discard(None)
    assert len(unique_closures) >= 4, "Expected at least four closure phase triplets"
    assert all(len(getattr(trace, "x", [])) for trace in closure_traces)


def test_show_cli_save_option(monkeypatch, real_oifits, tmp_path):
    calls = {"saved": None, "show": False}
    monkeypatch.setattr(
        show_module.viewer_plotly, "make_static_matisse_plot", lambda data: "figure"
    )
    monkeypatch.setattr(
        show_module.viewer_plotly,
        "show_plot",
        lambda fig: calls.__setitem__("show", True),
    )

    def fake_write(fig, path, engine):
        calls["saved"] = (fig, path, engine)

    monkeypatch.setattr(show_module.pio, "write_image", fake_write)

    save_path = tmp_path / "out.png"
    result = runner.invoke(
        app,
        ["show", str(real_oifits), "--save", str(save_path)],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert calls["saved"] == ("figure", save_path, "kaleido")
    assert calls["show"] is False


def test_show_cli_rejects_bad_extension(monkeypatch, real_oifits, tmp_path):
    monkeypatch.setattr(
        show_module.viewer_plotly, "make_static_matisse_plot", lambda data: "figure"
    )
    bad = tmp_path / "out.txt"

    result = runner.invoke(
        app,
        ["show", str(real_oifits), "--save", str(bad)],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "Unsupported output format" in result.stdout


@pytest.mark.parametrize("flag", ["--macports-probe", "--no-macports-probe"])
def test_doctor_command_runs(flag):
    """Ensure 'matisse doctor' command runs without crashing."""
    result = runner.invoke(
        app,
        ["doctor", flag],
        catch_exceptions=False,
    )
    # Exit code can be 0 (success) or 2 (esorex not found/configured)
    # This test just ensures the command runs
    assert result.exit_code in (0, 2), f"Unexpected exit code: {result.exit_code}"


def test_doctor_command_with_verbose():
    """Ensure 'matisse doctor' with verbose flag runs."""
    result = runner.invoke(
        app,
        ["doctor", "--verbose", "--no-macports-probe"],
        catch_exceptions=False,
    )
    assert result.exit_code in (0, 2), f"Unexpected exit code: {result.exit_code}"


def test_doctor_command_with_no_require_any():
    """Ensure 'matisse doctor' with --no-require-any flag runs."""
    result = runner.invoke(
        app,
        ["doctor", "--no-require-any", "--no-macports-probe"],
        catch_exceptions=False,
    )
    assert result.exit_code in (0, 2), f"Unexpected exit code: {result.exit_code}"


def test_doctor_command_esorex_not_found(monkeypatch):
    """Ensure 'matisse doctor' handles missing esorex gracefully."""
    from matisse_pipeline.cli import doctor as doctor_module

    # Mock shutil.which to return None (esorex not found)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda cmd: None)

    result = runner.invoke(
        app,
        ["doctor"],
        catch_exceptions=False,
    )
    # Should exit with code 2 (fatal error)
    assert result.exit_code == 2, f"Unexpected exit code: {result.exit_code}"
    # Should mention esorex not found
    assert "esorex" in result.output.lower()


def test_calibrate_command(data_dir, tmp_path):
    """Ensure 'matisse calibrate' run on testing data."""

    resultdir = tmp_path / "calibration_results"

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--data-dir",
            str(data_dir),
            "--result-dir",
            str(resultdir),
            "--bands",
            "LM",
        ],
        catch_exceptions=False,
    )

    list_oifits = list(resultdir.glob("*.fits"))
    data_merged = OIFitsReader(list_oifits[0]).read()
    assert result.exit_code == 0
    assert len(data_merged.wavelength) == 118  # Based on test data setup


def test_calibrate_invalid_band(data_dir, tmp_path, capfd):
    """Test that calibrate rejects invalid band names."""
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--data-dir",
            str(data_dir),
            "--result-dir",
            str(tmp_path),
            "--bands",
            "INVALID",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    # Check both result output and captured output
    output = result.output + capfd.readouterr().out
    assert "Invalid bands" in output or result.exit_code == 1


def test_calibrate_with_exception(data_dir, tmp_path, monkeypatch):
    """Test that calibrate handles exceptions properly."""

    # Mock run_calibration to raise an exception
    def mock_run_calibration(*args, **kwargs):
        raise RuntimeError("Simulated calibration error")

    monkeypatch.setattr(
        "matisse_pipeline.cli.calibrate.run_calibration",
        mock_run_calibration,
    )

    result = runner.invoke(
        app,
        [
            "calibrate",
            "--data-dir",
            str(data_dir),
            "--result-dir",
            str(tmp_path),
            "--bands",
            "LM",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 1
    assert "Calibration failed" in result.output
