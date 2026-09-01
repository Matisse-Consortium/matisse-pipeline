import logging
from pathlib import Path
from unittest import mock

import pytest
from astropy.io import fits

from matisse.core import lib_auto_calib as lac
from matisse.core.auto_calib import run_calibration
from matisse.core.utils.oifits_reader import OIFitsReader


@pytest.fixture
def mock_esorex(monkeypatch):
    """Mock esorex os.system calls for CI environments without esorex."""

    def mock_system(cmd):
        # Return 0 for success on any esorex command
        if "esorex" in cmd:
            return 0
        return 1

    monkeypatch.setattr("os.system", mock_system)
    return mock_system


@pytest.fixture
def skip_without_esorex():
    """Skip test if esorex is not available."""
    import subprocess

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


def test_generate_calibration_sof_files(data_dir, tmp_path):
    # Generate SOF files
    arbitraty_large_timespan = 300  # hours
    sof_files = lac.generate_sof_files(
        input_dir=data_dir,
        band="-LM",
        output_dir=tmp_path,
        timespan=arbitraty_large_timespan,
    )

    # Check that SOF files are created correctly
    output_sof = sof_files[0]
    assert output_sof.exists()

    with open(output_sof) as f:
        content = f.read()
        assert "CALIB_RAW_INT" in content
        assert "TARGET_RAW_INT" in content


def test_generate_sof_files_force_sci(data_dir, tmp_path):
    """A CAL target name passed via force_sci_names is promoted to SCI."""
    arbitrary_large_timespan = 300  # hours
    # ome01_Tau is the calibrator in tests/data/MATIS_calibrator-LM_reduced.fits
    sof_files = lac.generate_sof_files(
        input_dir=data_dir,
        band="-LM",
        output_dir=tmp_path,
        timespan=arbitrary_large_timespan,
        force_sci_names=["ome01_Tau"],
    )

    forced_sof = next((s for s in sof_files if "calibrator" in s.name), None)
    assert forced_sof is not None, "Expected a SOF for the forced ome01_Tau target"

    content = forced_sof.read_text()
    assert "MATIS_calibrator-LM_reduced.fits\tTARGET_RAW_INT" in content
    assert "CALIB_RAW_INT" not in content


def test_run_calibration_pipeline(data_dir, tmp_path, skip_without_esorex):
    """Test complete calibration pipeline (requires esorex)."""
    # Run the complete calibration pipeline
    resultdir = tmp_path / "calibration_results"
    run_calibration(
        input_dir=data_dir,
        output_dir=resultdir,
        bands=["LM", "N"],
        timespan=1,  # hours
        cumul_block=False,
    )

    n_file_expected = 4  # Based on test data setup
    n_oifits_expected = 1  # Based on test data setup
    expected_bcd_mode = "OUT_OUT"
    list_oifits = list(resultdir.glob("*.fits"))
    data_to_be_check = OIFitsReader(list_oifits[0]).read()

    expected_spectral_channel = 118

    assert resultdir.exists()
    assert len(list(resultdir.iterdir())) == n_file_expected
    assert len(list_oifits) == n_oifits_expected
    assert expected_bcd_mode in list_oifits[0].name
    assert len(data_to_be_check.wavelength) == expected_spectral_channel


def test_run_calibration_pipeline_cumul(data_dir, tmp_path, skip_without_esorex):
    """Test calibration pipeline with cumul_block (requires esorex)."""
    # Run the complete calibration pipeline
    resultdir = tmp_path / "calibration_results"
    run_calibration(
        input_dir=data_dir,
        output_dir=resultdir,
        bands=["LM", "N"],
        timespan=1,  # hours
        cumul_block=True,
    )

    n_file_expected = 5  # Based on test data setup
    n_oifits_expected = 2  # Based on test data setup
    expected_bcd_mode = "OUT_OUT"

    # Get list of generated OIFITS files
    list_oifits = list(resultdir.glob("*.fits"))
    data_merged = OIFitsReader(list_oifits[0]).read()
    data_bcd = OIFitsReader(list_oifits[1]).read()

    expected_spectral_channel = 118  # Based on test data setup

    assert resultdir.exists()
    assert len(list(resultdir.iterdir())) == n_file_expected
    assert len(list_oifits) == n_oifits_expected
    assert expected_bcd_mode in list_oifits[1].name
    assert len(data_bcd.wavelength) == expected_spectral_channel
    assert len(data_merged.wavelength) == expected_spectral_channel


def test_run_esorex_calibration_error(tmp_path, skip_without_esorex):
    """Test that run_esorex_calibration handles esorex errors correctly.

    This test creates a SOF file with missing input files, causing esorex to fail.
    It verifies that the function returns False and that the error is logged.
    """
    # Create output and input directories
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    output_dir.mkdir()
    input_dir.mkdir()

    # Create a SOF file with references to non-existent files
    sof_file = output_dir / "test_missing_files.sof"
    with open(sof_file, "w") as f:
        f.write("../input/nonexistent_target.fits\tTARGET_RAW_INT\n")
        f.write("../input/nonexistent_calib.fits\tCALIB_RAW_INT\n")

    # Call run_esorex_calibration - this will fail because files don't exist
    result = lac.run_esorex_calibration(
        sof_path=sof_file,
        output_dir=output_dir,
        cumul_block=True,
    )

    # Verify it returns False on error
    assert result is False

    # Verify log file was created and contains error information
    log_file = output_dir / "calibration.log"
    assert log_file.exists()
    log_content = log_file.read_text()
    assert "ERROR" in log_content or "Could not open" in log_content


def test_run_esorex_calibration_error_with_mock(tmp_path, skip_without_esorex):
    """Test that run_esorex_calibration handles esorex errors with mocking.

    This is a faster unit test that mocks os.system to simulate an error.
    """
    # Create a minimal SOF file
    sof_file = tmp_path / "test.sof"
    sof_file.write_text("dummy_target.fits\tTARGET_RAW_INT\n")

    # Mock os.system to return a non-zero exit code (error)
    with mock.patch("matisse.core.lib_auto_calib.os.system") as mock_system:
        mock_system.return_value = 2304  # Simulated esorex error code

        # Call run_esorex_calibration
        result = lac.run_esorex_calibration(
            sof_path=sof_file,
            output_dir=tmp_path,
            cumul_block=True,
        )

        # Verify it returns False on error
        assert result is False

        # Verify os.system was called with the correct command
        assert mock_system.called


def test_run_esorex_calibration_success(tmp_path):
    """Test that run_esorex_calibration succeeds with exit code 0."""
    # Create a minimal SOF file
    sof_file = tmp_path / "test.sof"
    sof_file.write_text("dummy_target.fits\tTARGET_RAW_INT\n")

    # Mock os.system to return 0 (success)
    with mock.patch("matisse.core.lib_auto_calib.os.system") as mock_system:
        mock_system.return_value = 0

        # Call run_esorex_calibration
        result = lac.run_esorex_calibration(
            sof_path=sof_file,
            output_dir=tmp_path,
            cumul_block=False,
        )

        # Verify it returns True on success
        assert result is True

        # Verify os.system was called
        assert mock_system.called


def test_run_calibration_no_sof_generated(data_dir, tmp_path, monkeypatch):
    """Test run_calibration when no SOF files are generated (no matching data)."""
    # Mock generate_sof_files to return empty list
    monkeypatch.setattr(
        "matisse.core.auto_calib.generate_sof_files",
        lambda **kwargs: [],
    )

    resultdir = tmp_path / "calibration_results"

    # Should complete without error even if no SOF files are generated
    run_calibration(
        input_dir=data_dir,
        output_dir=resultdir,
        bands=["N"],
        timespan=0.02,
        cumul_block=False,
    )

    # Directory should exist but be mostly empty
    assert resultdir.exists()


def test_run_calibration_esorex_failure(data_dir, tmp_path, monkeypatch):
    """Test run_calibration when esorex fails."""
    # Mock run_esorex_calibration to always fail
    monkeypatch.setattr(
        "matisse.core.auto_calib.run_esorex_calibration",
        lambda **kwargs: False,
    )

    resultdir = tmp_path / "calibration_results"

    # Should complete without raising exception even if esorex fails
    run_calibration(
        input_dir=data_dir,
        output_dir=resultdir,
        bands=["LM"],
        timespan=0.02,
        cumul_block=False,
    )

    assert resultdir.exists()


def test_run_calibration_pipeline_mocked(data_dir, tmp_path):
    """Test calibration pipeline with mocked esorex (for CI without esorex)."""
    # Mock run_esorex_calibration to return success without actually running esorex
    with mock.patch("matisse.core.auto_calib.run_esorex_calibration") as mock_esorex:
        mock_esorex.return_value = True

        resultdir = tmp_path / "calibration_results"

        # Should complete successfully even without esorex
        run_calibration(
            input_dir=data_dir,
            output_dir=resultdir,
            bands=["LM"],
            timespan=0.02,
            cumul_block=False,
        )

        # Verify run_esorex_calibration was called
        assert mock_esorex.called
        # Verify output directory was created
        assert resultdir.exists()


# ---------------------------------------------------------------------------
# run_esorex_calibration — failure paths (no real esorex needed)
# ---------------------------------------------------------------------------


def test_run_esorex_calibration_reports_log_tail_on_failure(
    tmp_path, monkeypatch, caplog
):
    """A non-zero esorex exit code is reported with the last 10 log lines."""
    sof_file = tmp_path / "test.sof"
    sof_file.write_text("dummy_target.fits\tTARGET_RAW_INT\n")
    (tmp_path / "calibration.log").write_text(
        "\n".join(f"log line {i}" for i in range(20)), encoding="utf-8"
    )
    monkeypatch.setattr(lac.os, "system", lambda cmd: 256)

    with caplog.at_level(logging.ERROR, logger="matisse"):
        assert lac.run_esorex_calibration(sof_file, tmp_path) is False

    assert "exit code: 256" in caplog.text
    assert "Last lines of calibration.log" in caplog.text
    # Only the final 10 lines are echoed back.
    assert "log line 19" in caplog.text
    assert "log line 10" in caplog.text
    assert "log line 9" not in caplog.text


def test_run_esorex_calibration_failure_without_log_file(tmp_path, monkeypatch):
    """A failure with no log file on disk still returns False without raising."""
    sof_file = tmp_path / "test.sof"
    sof_file.write_text("dummy_target.fits\tTARGET_RAW_INT\n")
    monkeypatch.setattr(lac.os, "system", lambda cmd: 1)

    assert lac.run_esorex_calibration(sof_file, tmp_path) is False


def test_run_esorex_calibration_returns_false_on_unexpected_error(
    tmp_path, monkeypatch
):
    """An exception raised while launching esorex is swallowed into False."""
    sof_file = tmp_path / "test.sof"
    sof_file.write_text("dummy_target.fits\tTARGET_RAW_INT\n")

    def boom(cmd):
        raise OSError("cannot spawn")

    monkeypatch.setattr(lac.os, "system", boom)

    assert lac.run_esorex_calibration(sof_file, tmp_path) is False


def test_run_esorex_calibration_passes_custom_recipe_dir(tmp_path, monkeypatch):
    """A custom recipe directory is forwarded to esorex as --recipe-dir."""
    sof_file = tmp_path / "test.sof"
    sof_file.write_text("dummy_target.fits\tTARGET_RAW_INT\n")
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    captured: list[str] = []

    monkeypatch.setattr(lac.os, "system", lambda cmd: captured.append(cmd) or 0)

    assert (
        lac.run_esorex_calibration(sof_file, tmp_path, custom_recipes_dir=recipes)
        is True
    )
    assert f"--recipe-dir {recipes}" in captured[0]


# ---------------------------------------------------------------------------
# rename_calibrated_outputs
# ---------------------------------------------------------------------------


def _unreachable():
    """Fail loudly if a code path we expect to be skipped is taken."""
    raise AssertionError("fits.open should not have been called")


def _calibrated_output(path, **cards):
    """Write a header-only FITS standing in for an esorex calibration product."""
    header = fits.Header()
    for key, value in cards.items():
        header[key] = value
    fits.PrimaryHDU(header=header).writeto(path, overwrite=True)


@pytest.mark.parametrize(
    ("chop_status", "expected_suffix"),
    [("F", "noChop"), ("T", "Chop")],
)
def test_rename_calibrated_outputs_uses_bcd_and_chop(
    tmp_path, chop_status, expected_suffix
):
    """BCD mode and chopping status drive the canonical output file name."""
    _calibrated_output(
        tmp_path / "TARGET_CAL_INT_0001.fits",
        **{"ESO CFG BCD MODE": "OUT-OUT", "ESO ISS CHOP ST": chop_status},
    )

    lac.rename_calibrated_outputs(tmp_path, "MYTARGET")

    assert (tmp_path / f"MYTARGET_OUT_OUT_{expected_suffix}.fits").exists()
    assert not (tmp_path / "TARGET_CAL_INT_0001.fits").exists()


def test_rename_calibrated_outputs_ignores_unexpected_extensions(tmp_path, monkeypatch):
    """A matching stem with a foreign extension is skipped before being opened."""
    (tmp_path / "TARGET_CAL_INT_0009.fits.bak").touch()
    opened: list = []
    monkeypatch.setattr(
        lac.fits, "open", lambda path, *a, **k: opened.append(path) or _unreachable()
    )

    lac.rename_calibrated_outputs(tmp_path, "MYTARGET")

    assert opened == []
    assert (tmp_path / "TARGET_CAL_INT_0009.fits.bak").exists()


def test_rename_calibrated_outputs_keeps_file_with_missing_keywords(tmp_path, caplog):
    """A product lacking the BCD keywords is left in place, and the failure is named."""
    _calibrated_output(tmp_path / "TARGET_CAL_INT_0002.fits")

    with caplog.at_level(logging.WARNING, logger="matisse"):
        lac.rename_calibrated_outputs(tmp_path, "MYTARGET")

    assert "Failed to rename TARGET_CAL_INT_0002.fits" in caplog.text
    assert (tmp_path / "TARGET_CAL_INT_0002.fits").exists()


def test_rename_calibrated_outputs_renames_nobcd_product(tmp_path):
    """The noBCD product is renamed to the plain base name."""
    _calibrated_output(tmp_path / "TARGET_CAL_INT_noBCD.fits")

    lac.rename_calibrated_outputs(tmp_path, "MYTARGET")

    assert (tmp_path / "MYTARGET.fits").exists()


def test_rename_calibrated_outputs_appends_extra_suffix(tmp_path):
    """An added suffix is appended before the extension."""
    _calibrated_output(
        tmp_path / "TARGET_CAL_INT_0003.fits",
        **{"ESO CFG BCD MODE": "IN-IN", "ESO ISS CHOP ST": "F"},
    )

    lac.rename_calibrated_outputs(tmp_path, "MYTARGET", added_suffix="_cumul")

    assert (tmp_path / "MYTARGET_IN_IN_noChop_cumul.fits").exists()


# ---------------------------------------------------------------------------
# cleanup_intermediate_files
# ---------------------------------------------------------------------------


def test_cleanup_intermediate_files_removes_only_intermediates(tmp_path):
    """CALCPHASE/CALDPHASE/CALVIS products are removed, others are kept."""
    for name in ("CALCPHASE_0001.fits", "CALDPHASE_0001.fits", "CALVIS_0001.fits"):
        (tmp_path / name).touch()
    keeper = tmp_path / "TARGET_CAL_INT_noBCD.fits"
    keeper.touch()

    lac.cleanup_intermediate_files(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [keeper.name]


def test_cleanup_intermediate_files_warns_when_removal_fails(
    tmp_path, monkeypatch, caplog
):
    """A file that cannot be unlinked is named in a warning; the others still go."""
    (tmp_path / "CALVIS_0001.fits").touch()
    (tmp_path / "CALCPHASE_0001.fits").touch()
    real_unlink = Path.unlink

    def refuse_one(self, *args, **kwargs):
        if self.name == "CALVIS_0001.fits":
            raise OSError("permission denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_one)

    with caplog.at_level(logging.WARNING, logger="matisse"):
        lac.cleanup_intermediate_files(tmp_path)

    assert "Failed to remove CALVIS_0001.fits" in caplog.text
    assert (tmp_path / "CALVIS_0001.fits").exists()
    assert not (tmp_path / "CALCPHASE_0001.fits").exists()


# ---------------------------------------------------------------------------
# generate_sof_files — nothing to reduce
# ---------------------------------------------------------------------------


def test_generate_sof_files_without_targets_hints_at_force_sci(tmp_path, caplog):
    """Calibrators but no science target produces a --force-sci hint."""
    _calibrated_output(
        tmp_path / "cal_IR-LM_LOW.fits",
        **{
            "ESO PRO CATG": "CALIB_RAW_INT",
            "ESO OBS TARG NAME": "HD1234",
            "MJD-OBS": 60000.0,
            "ESO DET CHIP NAME": "HAWAII-2RG",
            "ESO DET SEQ1 DIT": 0.075,
        },
    )

    with caplog.at_level(logging.WARNING, logger="matisse"):
        sof_files = lac.generate_sof_files(
            input_dir=tmp_path, output_dir=tmp_path, band="IR-LM", timespan=1.0
        )

    assert sof_files == []
    assert "No target files found for band IRLM" in caplog.text
    assert "HD1234" in caplog.text
    assert "--force-sci" in caplog.text


def test_generate_sof_files_without_any_file_warns(tmp_path):
    """An input directory with no matching band yields no SOF file."""
    assert (
        lac.generate_sof_files(
            input_dir=tmp_path, output_dir=tmp_path, band="IR-N", timespan=1.0
        )
        == []
    )
