import re
import subprocess

import pytest
from astropy.io import fits
from typer.testing import CliRunner

from matisse.cli import format_results as format_module, show as show_module
from matisse.cli.main import app
from matisse.core.utils.oifits_reader import OIFitsReader

runner = CliRunner()


@pytest.fixture
def skip_without_esorex():
    """Skip test if esorex is not available."""
    try:
        result = subprocess.run(
            ["which", "esorex"],
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            pytest.skip("esorex not available")
    except Exception:
        pytest.skip("esorex not available")


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


def test_show_cli_interactive(monkeypatch, viewer_dir):
    list_file_bcd = list(viewer_dir.glob("*.fits"))

    captured = {}

    def fake_show(fig, filename="matisse_view.html", auto_open=True, post_script=""):
        captured["fig"] = fig
        captured["post_script"] = post_script
        return fig

    monkeypatch.setattr(show_module.viewer_plotly, "show_plot", fake_show)

    result = runner.invoke(
        app, ["show", str(list_file_bcd[0]), "--interactive"], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert "fig" in captured
    assert "post_script" in captured
    post_script = captured["post_script"]
    assert post_script, "Expected interactive post_script to be injected"
    # UI now uses HTML <select> controls (single-band: BCD only; multi-band: band + BCD)
    assert "document.createElement('select')" in post_script
    assert ("BCD : " in post_script) and ("Band : " in post_script)


def test_show_cli_single_band_interactive(monkeypatch, viewer_dir, tmp_path):
    """
    Test the interactive mode of the `show` CLI command for single-band cases.
    Ensure only BCD files are present by removing IR-N files temporarily.
    """
    import shutil

    # Copy viewer_dir to a temporary path
    shutil.copytree(viewer_dir, tmp_path, dirs_exist_ok=True)  # py>=3.8

    # Remove all IR-N files to simulate a single-band scenario
    for f in tmp_path.rglob("*_IR-N*"):
        if f.is_file():
            f.unlink()

    # Collect remaining BCD files
    list_file_bcd = list(tmp_path.rglob("*.fits"))

    captured = {}

    def fake_show(fig, filename="matisse_view.html", auto_open=True, post_script=""):
        captured["fig"] = fig
        captured["post_script"] = post_script
        return fig

    monkeypatch.setattr(show_module.viewer_plotly, "show_plot", fake_show)

    result = runner.invoke(
        app, ["show", str(list_file_bcd[0]), "--interactive"], catch_exceptions=False
    )

    assert result.exit_code == 0
    assert "fig" in captured
    assert "post_script" in captured
    post_script = captured["post_script"]
    assert post_script, "Expected interactive post_script to be injected"
    # UI now uses HTML <select> controls (single-band: BCD only; multi-band: band + BCD)
    assert "document.createElement('select')" in post_script
    assert ("BCD : " in post_script) and ("Band : " not in post_script)


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


def test_calibrate_command(data_dir, tmp_path, skip_without_esorex):
    """Ensure 'matisse calibrate' run on testing data (requires esorex)."""

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


def test_calibrate_invalid_band(data_dir, tmp_path, capfd, skip_without_esorex):
    """Test that calibrate rejects invalid band names (requires esorex)."""
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


def test_calibrate_with_exception(data_dir, tmp_path, monkeypatch, skip_without_esorex):
    """Test that calibrate handles exceptions properly (requires esorex)."""

    # Mock run_calibration to raise an exception
    def mock_run_calibration(*args, **kwargs):
        raise RuntimeError("Simulated calibration error")

    monkeypatch.setattr(
        "matisse.cli.calibrate.run_calibration",
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


def test_calibrate_command_mocked(data_dir, tmp_path, monkeypatch):
    """Test calibrate CLI with mocked esorex (for CI without esorex)."""

    # Mock run_calibration to succeed without running esorex
    def mock_run_calibration(*args, **kwargs):
        # Just create dummy output directory
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "matisse.cli.calibrate.run_calibration",
        mock_run_calibration,
    )

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

    # Should succeed
    assert result.exit_code == 0
    assert resultdir.exists()


def test_bcd_compute_command(bcd_dir, tmp_path, caplog):
    """Ensure 'matisse bcd compute' runs on testing data."""

    # Capture all logs at DEBUG level
    caplog.set_level("DEBUG")

    result = runner.invoke(
        app,
        [
            "bcd",
            "compute",
            str(bcd_dir),
            "--output-dir",
            str(tmp_path),
            "--bcd-mode",
            "IN_IN",
        ],
        catch_exceptions=False,
    )

    # Print output and logs for debugging
    if result.exit_code != 0:
        print(f"\n[COMMAND OUTPUT]\n{result.output}")
        print("\n[CAPTURED LOGS]\n")
        for record in caplog.records:
            print(f"  {record.levelname}: {record.message}")

    assert result.exit_code == 0, (
        f"Command failed with exit code {result.exit_code}:\n{result.output}"
    )
    # Check that some correction files were created
    correction_files = list(tmp_path.glob("*.csv"))
    n_file = len(correction_files)

    assert n_file == 2, (
        f"Expected 2 correction files, found {n_file}:\n{correction_files}"
    )

    # Verify that both CSV files contain "IN_IN" (the BCD mode)
    for csv_file in correction_files:
        assert "IN_IN" in csv_file.name, (
            f"Expected 'IN_IN' in filename: {csv_file.name}"
        )
        # Also verify the CSV content contains IN_IN data
        csv_content = csv_file.read_text()
        assert len(csv_content) > 0, f"CSV file is empty: {csv_file.name}"


@pytest.mark.parametrize("bcd_mode", ["IN_IN", "OUT_IN", "IN_OUT"])
def test_bcd_compute_multiple_modes(bcd_dir, tmp_path, bcd_mode):
    """Test BCD compute for multiple BCD modes."""

    result = runner.invoke(
        app,
        [
            "bcd",
            "compute",
            str(bcd_dir),
            "--output-dir",
            str(tmp_path / bcd_mode),
            "--bcd-mode",
            bcd_mode,
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, (
        f"BCD compute failed for mode {bcd_mode}:\n{result.output}"
    )

    # Check that correction files were created for this BCD mode
    output_dir = tmp_path / bcd_mode
    correction_files = list(output_dir.glob("*.csv"))

    assert len(correction_files) == 2, (
        f"Expected 2 CSV files for {bcd_mode}, found {len(correction_files)}"
    )

    # Verify files contain the correct BCD mode
    for csv_file in correction_files:
        assert bcd_mode in csv_file.name, (
            f"Expected '{bcd_mode}' in filename: {csv_file.name}"
        )
        csv_content = csv_file.read_text()
        assert len(csv_content) > 0, f"CSV file is empty: {csv_file.name}"


def test_bcd_compute_all(bcd_dir, tmp_path):
    """Test BCD compute for ALL BCD modes."""

    result = runner.invoke(
        app,
        [
            "bcd",
            "compute",
            str(bcd_dir),
            "--output-dir",
            str(tmp_path),
            "--bcd-mode",
            "ALL",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"BCD compute failed for mode ALL:\n{result.output}"

    # Check that correction files were created for this BCD mode
    correction_files = list(tmp_path.glob("*.csv"))
    assert len(correction_files) == 6, (
        f"Expected 6 CSV files for ALL BCD modes, found {len(correction_files)}"
    )


def test_bcd_apply(bcd_apply_outputs):
    """Test BCD correction application."""
    corrections_dir = bcd_apply_outputs["corrections_dir"]
    corrected_dir = bcd_apply_outputs["corrected_dir"]

    correction_files = list(corrections_dir.glob("*.csv"))
    assert len(correction_files) == 6, (
        f"Expected 6 CSV files for ALL BCD modes, found {len(correction_files)}"
    )

    assert corrected_dir.exists(), (
        f"Corrected output directory was not created: {corrected_dir}"
    )
    corrected_files = list(sorted(corrected_dir.glob("*_bcd_corr.fits")))
    assert len(corrected_files) == 5, (
        f"No corrected FITS files were created in {corrected_dir}"
    )


def test_bcd_apply_expected_values(bcd_apply_outputs):
    """Test BCD correction application."""
    corrected_dir = bcd_apply_outputs["corrected_dir"]
    corrected_files = list(sorted(corrected_dir.glob("*_bcd_corr.fits")))

    # Check that the merged file was created and has the expected content
    merged = corrected_files[4]  # The last file is the merged one

    assert merged.exists(), f"Merged corrected file was not created: {merged}"

    data_merged = OIFitsReader(merged).read()

    # Check that the merged file has the expected number of wavelength points and baselines
    blname = data_merged.blname
    vis2 = data_merged.vis2["VIS2"]
    n_bl = vis2.shape[0]

    assert len(data_merged.wavelength) == 118, (
        f"Expected 118 wavelength points in merged file, found {len(data_merged.wavelength)}"
    )
    assert n_bl == 6, f"Expected 6 baselines in merged file, found {n_bl}"

    # Check that the baseline names are in expected order (OUT-OUT)
    expected_blname = [
        "U3-U4",
        "U1-U2",
        "U2-U3",
        "U2-U4",
        "U1-U3",
        "U1-U4",
    ]
    for i, bl in enumerate(expected_blname):
        assert blname[i] == bl, (
            f"Expected baseline {bl} at index {i}, but got {blname[i]}"
        )

    # Check that the VIS2 values are within expected ranges (based on test data)
    expected_value = [
        0.75,  # U3-U4
        0.70,  # U1-U2
        0.57,  # U2-U3
        0.49,  # U2-U4
        0.42,  # U1-U3
        0.32,  # U1-U4
    ]
    # Mask to select the wavelength range around 3.45 microns
    # where we expect the values to be close to the expected ones
    mask = (data_merged.wavelength * 1e6 >= 3.4) & (data_merged.wavelength * 1e6 <= 3.5)
    for i in range(n_bl):
        vis2_values = vis2[i][mask].mean()
        assert vis2_values == pytest.approx(expected_value[i], abs=0.01), (
            f"Expected {expected_value[i]}, got {vis2_values}"
        )


def test_bcd_apply_verbose(bcd_dir, tmp_path):
    """Apply bcd corrections with verbose output for testing purposes."""
    import shutil

    temp_data_dir = tmp_path / "temp_bcd_data"
    shutil.copytree(bcd_dir, temp_data_dir)

    corrections_dir = tmp_path / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)

    result_compute = runner.invoke(
        app,
        [
            "bcd",
            "compute",
            str(temp_data_dir),
            "--output-dir",
            str(corrections_dir),
            "--bcd-mode",
            "ALL",
        ],
        catch_exceptions=False,
    )

    assert result_compute.exit_code == 0, result_compute.output

    result_apply = runner.invoke(
        app,
        [
            "bcd",
            "apply",
            str(temp_data_dir),
            str(corrections_dir),
            "--verbose",
        ],
        catch_exceptions=False,
    )

    assert result_apply.exit_code == 0, result_apply.output


@pytest.fixture(scope="module")
def bcd_apply_outputs(bcd_dir, tmp_path_factory):
    """Prepare corrections and corrected directory for BCD tests."""
    import shutil

    base_dir = tmp_path_factory.mktemp("bcd_apply")
    temp_data_dir = base_dir / "temp_bcd_data"
    shutil.copytree(bcd_dir, temp_data_dir)

    corrections_dir = base_dir / "corrections"
    corrections_dir.mkdir(parents=True, exist_ok=True)

    result_compute = runner.invoke(
        app,
        [
            "bcd",
            "compute",
            str(temp_data_dir),
            "--output-dir",
            str(corrections_dir),
            "--bcd-mode",
            "ALL",
            "--tau0-min",
            "5",
        ],
        catch_exceptions=False,
    )

    assert result_compute.exit_code == 0, result_compute.output

    result_apply = runner.invoke(
        app,
        [
            "bcd",
            "apply",
            str(temp_data_dir),
            str(corrections_dir),
            "--merge",
        ],
        catch_exceptions=False,
    )

    assert result_apply.exit_code == 0, result_apply.output

    corrected_dir = temp_data_dir.parent / "bcd_corrected/"

    return {
        "temp_data_dir": temp_data_dir,
        "corrections_dir": corrections_dir,
        "corrected_dir": corrected_dir,
    }


def test_bcd_compare_minimal(bcd_apply_outputs):
    """Minimal test for `matisse bcd compare` figure generation."""
    import matplotlib.pyplot as plt

    result_compare = runner.invoke(
        app,
        [
            "bcd",
            "compare",
            str(bcd_apply_outputs["corrected_dir"]),
        ],
        catch_exceptions=False,
    )

    assert result_compare.exit_code == 0, result_compare.output
    assert len(plt.get_fignums()) == 1, (
        f"Expected 1 figure to be generated, but found {len(plt.get_fignums())}"
    )


def test_bcd_compare_cli_no_files(tmp_path):
    """Test `matisse bcd compare` on empty directory exits with error."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    result = runner.invoke(
        app,
        ["bcd", "compare", str(empty_dir)],
        catch_exceptions=False,
    )

    assert result.exit_code == 1


def test_bcd_remove(bcd_dir, tmp_path):
    """Test `matisse bcd remove` functionality."""
    import shutil
    from pathlib import Path

    temp_data_dir = tmp_path / "temp_bcd_data"
    shutil.copytree(bcd_dir, temp_data_dir)

    result = runner.invoke(
        app,
        ["bcd", "remove", str(temp_data_dir), "--cal"],
        catch_exceptions=True,
    )

    output_dir = temp_data_dir.parent / f"{temp_data_dir.name}_noBCD/"

    n_original = len(list(bcd_dir.glob("*.fits")))
    n_removed = len(list(Path(output_dir).glob("*_noBCD.fits")))

    # 4 files in test data, expect 3 to be processed (OUT_OUT is not affected)
    assert n_removed == n_original - 1, (
        f"Expected {n_original - 1} files to be processed, but found {n_removed} in output directory."
    )
    assert result.exit_code == 0


def test_bcd_merge(bcd_dir, tmp_path):
    """Test `matisse bcd merge` functionality."""
    import shutil

    temp_data_dir = tmp_path / "temp_bcd_data"
    shutil.copytree(bcd_dir, temp_data_dir)

    n_original = len(list(temp_data_dir.glob("*.fits")))

    result = runner.invoke(
        app,
        ["bcd", "merge", str(temp_data_dir)],
        catch_exceptions=True,
    )

    n_processed = len(list(temp_data_dir.glob("*.fits")))
    n_merge = len(list(temp_data_dir.glob("*LOW_noChop.fits")))

    # 4 files in test data, expect 3 to be processed (OUT_OUT is not affected)
    assert n_processed == n_original + 1, (
        f"Expected {n_original + 1} files to be processed, but found {n_processed} in output directory."
    )
    assert n_merge == 1, (
        f"Expected 1 merged file to be created, but found {n_merge} in output directory."
    )
    assert result.exit_code == 0


def test_flux_calibrate(flux_dir, tmp_path):
    """Test `matisse flux_calibrate` functionality."""
    import shutil

    temp_data_dir = tmp_path / "temp_flux_data"
    shutil.copytree(flux_dir, temp_data_dir)

    result = runner.invoke(
        app,
        ["flux_calibrate", "-d", str(temp_data_dir), "-f", tmp_path],
        catch_exceptions=True,
    )

    resultdir = temp_data_dir / "calflux"

    assert result.exit_code == 0, f"Flux calibration failed:\n{result.output}"
    assert resultdir.exists(), "Flux calibration output directory was not created."
    calibrated_files = list(resultdir.glob("*_calflux.fits"))
    assert len(calibrated_files) == 1, (
        f"Expected 1 calibrated file, found {len(calibrated_files)}:\n{calibrated_files}"
    )


def test_flux_calibrate_show(flux_dir, tmp_path):
    """Test `matisse flux_calibrate` functionality with figures."""
    import shutil

    from matplotlib import pyplot as plt

    temp_data_dir = tmp_path / "temp_flux_data"
    shutil.copytree(flux_dir, temp_data_dir)

    result = runner.invoke(
        app,
        ["flux_calibrate", "-d", str(temp_data_dir), "--show", "--sf", "-f", tmp_path],
        catch_exceptions=True,
    )

    resultdir = temp_data_dir / "calflux"

    assert result.exit_code == 0, f"Flux calibration failed:\n{result.output}"
    assert resultdir.exists(), "Flux calibration output directory was not created."
    calibrated_files = list(resultdir.glob("*_calflux.fits"))
    assert len(calibrated_files) == 1, (
        f"Expected 1 calibrated file, found {len(calibrated_files)}:\n{calibrated_files}"
    )
    assert len(plt.get_fignums()) == 2, (
        f"Expected 1 figure to be generated, but found {len(plt.get_fignums())}"
    )


def test_flux_calibrate_airmass_correction(flux_dir, tmp_path):
    """Test `matisse flux_calibrate` functionality with airmass correction."""
    import shutil

    temp_data_dir = tmp_path / "temp_flux_data"
    shutil.copytree(flux_dir, temp_data_dir)

    result = runner.invoke(
        app,
        ["flux_calibrate", "-d", str(temp_data_dir), "--airmass-corr", "-f", tmp_path],
        catch_exceptions=True,
    )

    resultdir = temp_data_dir / "calflux"

    assert result.exit_code == 0, f"Flux calibration failed:\n{result.output}"
    assert resultdir.exists(), "Flux calibration output directory was not created."
    calibrated_files = list(resultdir.glob("*_calflux.fits"))
    assert len(calibrated_files) == 1, (
        f"Expected 1 calibrated file, found {len(calibrated_files)}:\n{calibrated_files}"
    )


def test_flux_calibrate_Nband(flux_dir, tmp_path):
    """Test `matisse flux_calibrate` functionality with correlated flux."""
    import shutil

    temp_data_dir = tmp_path / "temp_flux_data"
    shutil.copytree(flux_dir, temp_data_dir)

    result = runner.invoke(
        app,
        [
            "flux_calibrate",
            "-d",
            str(temp_data_dir),
            "--no-airmass-corr",
            "--mode",
            "both",
            "-b",
            "N",
            "-f",
            tmp_path,
        ],
        catch_exceptions=True,
    )

    resultdir = temp_data_dir / "calcorrflux"

    assert result.exit_code == 0, f"Flux calibration failed:\n{result.output}"
    assert resultdir.exists(), "Flux calibration output directory was not created."
    calibrated_files = list(resultdir.glob("*_calcorrflux.fits"))
    assert len(calibrated_files) == 1, (
        f"Expected 1 calibrated file, found {len(calibrated_files)}:\n{calibrated_files}"
    )


def test_flux_calibrate_invalid_band(flux_dir, tmp_path):
    """Typer should reject invalid --band values before command logic runs."""
    import shutil

    temp_data_dir = tmp_path / "temp_flux_data"
    shutil.copytree(flux_dir, temp_data_dir)

    result = runner.invoke(
        app,
        [
            "flux_calibrate",
            "-d",
            str(temp_data_dir),
            "--band",
            "INVALID",
            "-f",
            tmp_path,
        ],
        catch_exceptions=True,
    )

    assert result.exit_code != 0
    assert "Invalid value for '--band'" in strip_ansi(result.output)


def test_flux_calibrate_invalid_mode(flux_dir, tmp_path):
    """Typer should reject invalid --mode values before command logic runs."""
    import shutil

    temp_data_dir = tmp_path / "temp_flux_data"
    shutil.copytree(flux_dir, temp_data_dir)

    result = runner.invoke(
        app,
        [
            "flux_calibrate",
            "-d",
            str(temp_data_dir),
            "--mode",
            "INVALID",
            "-f",
            tmp_path,
        ],
        catch_exceptions=True,
    )

    assert result.exit_code != 0
    assert "Invalid value for '--mode'" in strip_ansi(result.output)


def test_flux_calibrate_run_failure_is_reported(monkeypatch, tmp_path):
    """Ensure flux_calibrate reports exceptions raised by run_flux_calibration."""
    from matisse.cli import flux_calibrate as flux_calibrate_module

    def fail_run(_config):
        raise RuntimeError("simulated calibration crash")

    monkeypatch.setattr(flux_calibrate_module, "run_flux_calibration", fail_run)

    result = runner.invoke(
        app,
        ["flux_calibrate", "-d", str(tmp_path), "-f", tmp_path],
        catch_exceptions=True,
    )

    output = strip_ansi(result.output)
    assert result.exit_code == 1
    assert "Flux calibration failed" in output
    assert "simulated calibration crash" in output


def test_reduce_inventory(bcd_dir):
    """Test that --check-files calls show_files_inventory with the correct dir."""
    from matisse.core.utils.log_utils import show_files_inventory

    result = runner.invoke(
        app,
        [
            "reduce",
            "--data-dir",
            str(bcd_dir),
            "--check-files",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, f"Command failed:\n{result.output}"

    # Verify the inventory function returns a table with 4 rows (one per FITS file)
    n_fits = len(list(bcd_dir.glob("*.fits")))
    assert n_fits == 4, f"Expected 4 FITS files in bcd_dir, found {n_fits}"

    table = show_files_inventory(str(bcd_dir))
    assert table.row_count == n_fits, (
        f"Expected {n_fits} rows in inventory table, got {table.row_count}"
    )


def test_show_files_inventory_ignores_appledouble_sidecar(tmp_path):
    from matisse.core.utils.log_utils import show_files_inventory

    valid = tmp_path / "MATIS_valid.fits"
    sidecar = tmp_path / "._MATIS_valid.fits"

    fits.PrimaryHDU().writeto(valid)
    sidecar.write_text("not a fits file")

    table = show_files_inventory(str(tmp_path))
    assert table.row_count == 1
