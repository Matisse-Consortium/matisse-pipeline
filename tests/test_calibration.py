from unittest import mock

import pytest

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
    arbitraty_large_timespan = 0.02  # days
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


def test_run_calibration_pipeline(data_dir, tmp_path, skip_without_esorex):
    """Test complete calibration pipeline (requires esorex)."""
    # Run the complete calibration pipeline
    resultdir = tmp_path / "calibration_results"
    run_calibration(
        input_dir=data_dir,
        output_dir=resultdir,
        bands=["LM", "N"],
        timespan=0.02,  # days
        cumul_block=False,
    )

    n_file_expected = 4  # Based on test data setup
    n_oifits_expected = 1  # Based on test data setup
    expected_bcd_mode = "OUT-OUT"
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
        timespan=0.02,  # days
        cumul_block=True,
    )

    n_file_expected = 5  # Based on test data setup
    n_oifits_expected = 2  # Based on test data setup
    expected_bcd_mode = "OUT-OUT"

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
