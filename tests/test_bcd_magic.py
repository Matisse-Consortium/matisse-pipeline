import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from typer.testing import CliRunner

from matisse.cli.main import app
from matisse.core.bcd import (
    correction as correction_module,
    visualization as visualization_module,
)
from matisse.core.bcd.config import BCDConfig
from matisse.core.bcd.visualization import (
    plot_corrections,
    plot_poly_corrections_results,
)


def test_magic_num_cli_runs_with_real_lamp(
    tmp_path, real_lamp_outout, real_lamp_inin, monkeypatch
):
    monkeypatch.setattr(plt, "show", lambda: None)
    runner = CliRunner()
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    for src in (real_lamp_outout, real_lamp_inin):
        shutil.copy(src, dataset_dir / Path(src).name)

    result = runner.invoke(
        app,
        [
            "bcd",
            "compute",
            str(dataset_dir),
            "--output-dir",
            str(tmp_path),
            "--plot",
        ],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    csv_file = tmp_path / "bcd_IN_IN_spectral_corrections.csv"
    assert csv_file.exists()

    png_files = sorted((tmp_path / "diagnostic_plot").glob("bcd_diagnostic_*.png"))
    assert len(png_files) == 4


def test_plot_corrections_returns_four_figures(monkeypatch, tmp_path):
    monkeypatch.setattr(plt, "show", lambda: None)
    config = BCDConfig(
        output_dir=tmp_path,
        spectral_binlen=2,
        wavelength_low=3.3e-6,
        wavelength_high=3.4e-6,
        poly_order=1,
        band="LM",
    )
    wavelengths = np.array([3.3e-6, 3.4e-6], dtype=float)
    corrections_mean = np.ones((1, 6))
    corrections_spectral = np.ones((1, 6, 2))
    combined = [
        np.ones((1, len(wavelengths))),
        np.ones((1, len(wavelengths))),
        np.ones((1, len(wavelengths))),
    ]
    poly_coef = np.ones((2, 2, config.poly_order + 1))

    figures = plot_corrections(
        wavelengths=wavelengths,
        corrections_mean=corrections_mean,
        corrections_spectral=corrections_spectral,
        combined_spectral=combined,
        poly_coef=poly_coef,
        config=config,
        save_plots=False,
    )

    assert len(figures) == 4
    for fig in figures:
        assert fig is not None
        plt.close(fig)


def test_compute_bcd_corrections_raises_when_no_pairs(tmp_path, monkeypatch):
    config = BCDConfig(output_dir=tmp_path)

    monkeypatch.setattr(
        correction_module,
        "_find_bcd_file_pairs",
        lambda **kwargs: [],
    )

    with pytest.raises(FileNotFoundError):
        correction_module.compute_bcd_corrections(
            folders=[str(tmp_path)],
            config=config,
            chopping=False,
            show_plots=False,
        )


def test_validate_file_filters_correlated_flux(monkeypatch):
    config = BCDConfig()
    config.correlated_flux = True

    class FakeExt:
        def __init__(self, header):
            self.header = header

    class FakeTable:
        def __init__(self, data):
            self.data = data

    class FakeHDUL(dict):
        def __getitem__(self, key):
            if key == 0:
                return FakeExt({"OBJECT": "STD"})
            if key == 3:
                return FakeTable({"eff_wave": np.arange(config.spectral_binlen)})
            if key == config.extension:
                return FakeExt({"AMPTYP": "correlated flux"})
            raise KeyError(key)

    hdul = FakeHDUL()
    assert correction_module._validate_file(hdul, config)


def test_magic_num_nofile(tmp_path, caplog):
    runner = CliRunner()
    tmp_path.mkdir(exist_ok=True)
    result = runner.invoke(app, ["bcd", "compute", str(tmp_path)])

    assert "File not found: No valid file" in caplog.text
    assert result.exit_code == 1


def test_plot_poly_corrections_results_from_csv(tmp_path):
    wavelengths = np.array([3.3e-6, 3.4e-6])
    corrections = pd.DataFrame(
        {
            "wavelength": wavelengths,
            "B0": [1.0, 1.1],
            "B0_std": [0.05, 0.05],
            "B1": [0.9, 1.0],
            "B1_std": [0.04, 0.04],
        }
    )
    corrections.to_csv(tmp_path / "bcd_IN_IN_spectral_corrections.csv", index=False)

    poly_rows = [
        {
            "baseline_idx1": 0,
            "baseline_idx2": 1,
            "window": 0,
            "wl_start_um": 3.3,
            "wl_end_um": 3.4,
            "coef_x0": 1.0,
            "coef_x1": 0.0,
            "coef_x2": 0.0,
            "coef_x3": 0.0,
        },
        {
            "baseline_idx1": 1,
            "baseline_idx2": 0,
            "window": 1,
            "wl_start_um": 3.3,
            "wl_end_um": 3.4,
            "coef_x0": 1.0,
            "coef_x1": 0.1,
            "coef_x2": 0.0,
            "coef_x3": 0.0,
        },
    ]
    pd.DataFrame(poly_rows).to_csv(tmp_path / "bcd_IN_IN_poly_coeffs.csv", index=False)

    fig = plot_poly_corrections_results(tmp_path, bcd_mode="IN_IN")

    assert fig is not None
    axes = fig.get_axes()
    assert len(axes) == 6
    assert axes[0].get_visible() is True
    assert axes[2].get_visible() is False
    plt.close(fig)


def test_plot_poly_corrections_results_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_poly_corrections_results(tmp_path, bcd_mode="IN_IN")


def test_magic_cli_plots_existing_results(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    runner = CliRunner()

    wavelengths = np.array([3.3e-6, 3.4e-6])
    corrections = pd.DataFrame(
        {
            "wavelength": wavelengths,
            "B0": [1.0, 1.1],
            "B0_std": [0.05, 0.05],
            "B1": [0.9, 1.0],
            "B1_std": [0.04, 0.04],
        }
    )
    corrections.to_csv(tmp_path / "bcd_IN_IN_spectral_corrections.csv", index=False)

    poly_rows = [
        {
            "baseline_idx1": 0,
            "baseline_idx2": 1,
            "window": 0,
            "wl_start_um": 3.3,
            "wl_end_um": 3.4,
            "coef_x0": 1.0,
            "coef_x1": 0.0,
            "coef_x2": 0.0,
        }
    ]
    pd.DataFrame(poly_rows).to_csv(tmp_path / "bcd_IN_IN_poly_coeffs.csv", index=False)

    tau0_min_value = 5
    result = runner.invoke(
        app,
        [
            "bcd",
            "compute",
            "--results-dir",
            str(tmp_path),
            "--tau0-min",
            str(tau0_min_value),
        ],
    )

    assert result.exit_code == 0, result.output


def test_magic_cli_plots_existing_results_missing_csv(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["bcd", "compute", "--results-dir", str(tmp_path)])

    assert result.exit_code != 0


def test_plot_corrections_saves_plots(tmp_path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    config = BCDConfig(
        output_dir=tmp_path,
        spectral_binlen=2,
        wavelength_low=3.3e-6,
        wavelength_high=3.4e-6,
        poly_order=1,
        band="LM",
    )
    wavelengths = np.array([3.3e-6, 3.4e-6], dtype=float)
    corrections_mean = np.ones((1, 6))
    corrections_spectral = np.ones((1, 6, 2))
    combined = [
        np.ones((1, len(wavelengths))),
        np.ones((1, len(wavelengths))),
        np.ones((1, len(wavelengths))),
    ]
    poly_coef = np.ones((2, 2, config.poly_order + 1))

    figs = plot_corrections(
        wavelengths=wavelengths,
        corrections_mean=corrections_mean,
        corrections_spectral=corrections_spectral,
        combined_spectral=combined,
        poly_coef=poly_coef,
        config=config,
        save_plots=True,
        bcd_mode="IN_IN",
    )

    png_files = sorted((tmp_path / "diagnostic_plot").glob("bcd_diagnostic*.png"))
    assert len(png_files) == 4
    for fig in figs:
        plt.close(fig)


def test_plot_poly_corrections_results_poly3(tmp_path):
    wavelengths = np.array([3.3e-6, 3.4e-6, 3.5e-6])
    corrections = pd.DataFrame(
        {
            "wavelength": wavelengths,
            "B0": [1.0, 1.1, 1.2],
            "B0_std": [0.05, 0.05, 0.05],
        }
    )
    corrections.to_csv(tmp_path / "bcd_IN_IN_spectral_corrections.csv", index=False)

    poly_rows = [
        {
            "baseline_idx1": 0,
            "baseline_idx2": 0,
            "window": 0,
            "wl_start_um": 3.3,
            "wl_end_um": 3.5,
            "coef_x0": 1.0,
            "coef_x1": 0.1,
            "coef_x2": 0.01,
            "coef_x3": -0.001,
        }
    ]
    pd.DataFrame(poly_rows).to_csv(tmp_path / "bcd_IN_IN_poly_coeffs.csv", index=False)

    fig = plot_poly_corrections_results(tmp_path, bcd_mode="IN_IN")

    axes = fig.get_axes()
    handles, labels = axes[0].get_legend_handles_labels()
    assert "Polynomial fit" in labels
    plt.close(fig)


def test_plot_mean_corrections_with_mplcursors(tmp_path, monkeypatch):
    """Test that _plot_mean_corrections uses mplcursors when available."""
    monkeypatch.setattr(plt, "show", lambda: None)

    config = BCDConfig(
        output_dir=tmp_path,
        wavelength_low=3.3e-6,
        wavelength_high=3.4e-6,
        bcd_mode="IN_IN",
    )

    corrections_mean = np.array(
        [
            [1.0, 1.1, 0.9, 1.05, 0.95, 1.02],
            [1.1, 1.0, 1.0, 0.98, 1.03, 0.97],
        ]
    )

    file_labels = ["file1.fits", "file2.fits"]
    baseline_names = ["B0", "B1", "B2", "B3", "B4", "B5"]
    target_names = ["HD123456", "HD789012"]
    tau0_values = [4.5, 2.0]  # Good and medium quality

    from matisse.core.bcd.visualization import _plot_mean_corrections

    fig = _plot_mean_corrections(
        corrections_mean,
        config,
        file_labels=file_labels,
        baseline_names=baseline_names,
        target_names=target_names,
        tau0_values=tau0_values,
    )

    # Check that figure was created
    assert fig is not None
    ax = fig.get_axes()[0]

    # Check that data lines were plotted (2 data lines + 1 mean line with error bars)
    lines = ax.get_lines()
    assert len(lines) >= 2  # At least the 2 data lines

    # Check that labels are set correctly
    labels = [line.get_label() for line in lines]
    assert "file1.fits" in labels
    assert "file2.fits" in labels

    plt.close(fig)


def test_plot_mean_corrections_without_mplcursors(tmp_path, monkeypatch):
    """Test that _plot_mean_corrections works without mplcursors."""
    monkeypatch.setattr(plt, "show", lambda: None)

    config = BCDConfig(
        output_dir=tmp_path,
        wavelength_low=3.3e-6,
        wavelength_high=3.4e-6,
        bcd_mode="IN_IN",
    )

    corrections_mean = np.array(
        [
            [1.0, 1.1, 0.9, 1.05, 0.95, 1.02],
            [1.1, 1.0, 1.0, 0.98, 1.03, 0.97],
        ]
    )

    from matisse.core.bcd.visualization import _plot_mean_corrections

    # Test without optional parameters
    fig = _plot_mean_corrections(corrections_mean, config)

    # Check that figure was created even without mplcursors
    assert fig is not None
    ax = fig.get_axes()[0]

    # Check that data was plotted
    lines = ax.get_lines()
    assert len(lines) >= 2  # At least 2 data lines

    # Check default labels
    labels = [line.get_label() for line in lines]
    assert "File 1" in labels
    assert "File 2" in labels

    plt.close(fig)


def test_compute_poly_correction_missing_coef_columns(tmp_path):
    """Test that missing polynomial coefficient columns raise ValueError."""
    from matisse.core.bcd.correction import _compute_poly_correction

    wl = np.linspace(3, 5, 50)
    # DataFrame without coef_x* columns
    df_bad = pd.DataFrame(
        {
            "wl_start_um": [3.2, 4.55],
            "wl_end_um": [3.8, 4.9],
        }
    )

    with pytest.raises(
        ValueError,
        match="No polynomial coefficient columns found",
    ):
        _compute_poly_correction(wl, df_bad, 0)


def test_compare_bcd_no_corrected_files(tmp_path):
    """Test that compare fails gracefully with no corrected files."""
    from matisse.core.bcd.visualization import compare_bcd_corrections

    empty_dir = tmp_path / "empty_corr"
    empty_dir.mkdir()

    with pytest.raises(
        ValueError,
        match="No BCD-corrected files found",
    ):
        compare_bcd_corrections(empty_dir)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BCD_ALL_MODES = ("OUT_OUT", "IN_IN", "IN_OUT", "OUT_IN")
BCD_BASE = "fake_calib_IR-LM_LOW"


def _copy_bcd_dir(bcd_dir, tmp_path, name="data"):
    """Copy the reference BCD dataset into a writable temporary directory."""
    dest = Path(tmp_path) / name
    shutil.copytree(bcd_dir, dest)
    return dest


def _make_corrected_dir(bcd_dir, dest, base=BCD_BASE, chop="noChop"):
    """Build a *_bcd_corr.fits directory from the reference BCD dataset."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for mode in BCD_ALL_MODES:
        shutil.copy(
            Path(bcd_dir) / f"{BCD_BASE}_{mode}_noChop.fits",
            dest / f"{base}_{mode}_{chop}_bcd_corr.fits",
        )
    return dest


def _write_poly_csv(directory, coef_x0=1.6, coef_x1=0.0):
    """Write flat polynomial coefficient CSVs for the three corrected modes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "baseline_idx1": idx1,
            "baseline_idx2": idx2,
            "window": window,
            "wl_start_um": start,
            "wl_end_um": end,
            "coef_x1": coef_x1,
            "coef_x0": coef_x0,
        }
        for idx1, idx2 in ((2, 5), (3, 4))
        for window, (start, end) in enumerate([(3.2, 3.8), (4.55, 4.9)])
    ]
    for mode in ("IN_IN", "IN_OUT", "OUT_IN"):
        pd.DataFrame(rows).to_csv(
            directory / f"bcd_{mode}_poly_coeffs.csv", index=False
        )
    return directory


# ---------------------------------------------------------------------------
# correction._validate_file
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("object_value", "expected"),
    [("STD", True), ("LAMP", True), ("SCIENCE", False)],
)
def test_validate_file_object_keyword(bcd_dir, tmp_path, object_value, expected):
    """Only STD/LAMP files are accepted by _validate_file."""
    path = Path(shutil.copy(bcd_dir / f"{BCD_BASE}_IN_IN_noChop.fits", tmp_path))
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["OBJECT"] = object_value

    config = BCDConfig(output_dir=tmp_path, spectral_binlen=118)
    with fits.open(path) as hdul:
        assert correction_module._validate_file(hdul, config) is expected


def test_validate_file_rejects_wrong_spectral_length(bcd_dir, tmp_path):
    """A wavelength table shorter than spectral_binlen is rejected."""
    config = BCDConfig(output_dir=tmp_path, spectral_binlen=42)
    with fits.open(bcd_dir / f"{BCD_BASE}_IN_IN_noChop.fits") as hdul:
        assert len(hdul[3].data["eff_wave"]) == 118
        assert correction_module._validate_file(hdul, config) is False


@pytest.mark.parametrize(
    ("amptyp", "correlated_flux", "expected"),
    [
        (None, False, True),
        (None, True, False),
        ("correlated flux", True, True),
        ("correlated flux", False, False),
    ],
)
def test_validate_file_correlated_flux(
    bcd_dir, tmp_path, amptyp, correlated_flux, expected
):
    """_validate_file keeps only files whose AMPTYP matches the requested mode."""
    path = Path(shutil.copy(bcd_dir / f"{BCD_BASE}_IN_IN_noChop.fits", tmp_path))
    if amptyp is not None:
        with fits.open(path, mode="update") as hdul:
            hdul["OI_VIS2"].header["AMPTYP"] = amptyp

    config = BCDConfig(output_dir=tmp_path, spectral_binlen=118)
    config.correlated_flux = correlated_flux
    with fits.open(path) as hdul:
        assert correction_module._validate_file(hdul, config) is expected


# ---------------------------------------------------------------------------
# correction: small helpers
# ---------------------------------------------------------------------------
def test_combine_baseline_pairs_unknown_mode():
    """_combine_baseline_pairs rejects a BCD mode without defined pairs."""
    data = np.ones((1, 6, 3))
    with pytest.raises(ValueError, match="No baseline pairs defined for OUT_OUT"):
        correction_module._combine_baseline_pairs(data, "OUT_OUT")


def test_combine_baseline_pairs_inverts_partner_baseline():
    """The swapped partner of each pair is stacked as its inverse."""
    data = np.arange(1, 13, dtype=float).reshape(1, 6, 2)

    combined = correction_module._combine_baseline_pairs(data, "IN_IN")

    assert [c.shape for c in combined] == [(2, 2), (2, 2), (2, 2)]
    # IN_IN pairs are [[2, 5], [3, 4]]
    assert np.allclose(combined[1][0], data[0, 2])
    assert np.allclose(combined[1][1], 1.0 / data[0, 5])
    assert np.allclose(combined[2][0], data[0, 3])
    assert np.allclose(combined[2][1], 1.0 / data[0, 4])


@pytest.mark.parametrize("missing_mode", [None, "IN_OUT", "OUT_OUT"])
def test_has_all_bcd_files(bcd_dir, tmp_path, missing_mode):
    """_has_all_bcd_files requires the four BCD modes of a quadruplet."""
    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    if missing_mode is not None:
        (data_dir / f"{BCD_BASE}_{missing_mode}_noChop.fits").unlink()

    result = correction_module._has_all_bcd_files(data_dir, BCD_BASE)

    assert result is (missing_mode is None)


def test_find_calibrator_filename_bases_chopping(bcd_dir, tmp_path):
    """Chopping mode looks for the _OUT_OUT_Chop.fits suffix."""
    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    shutil.copy(
        data_dir / f"{BCD_BASE}_OUT_OUT_noChop.fits",
        data_dir / f"{BCD_BASE}_OUT_OUT_Chop.fits",
    )

    assert correction_module._find_calibrator_filename_bases(
        data_dir, chopping=True
    ) == [BCD_BASE]
    assert correction_module._find_calibrator_filename_bases(
        data_dir, chopping=False
    ) == [BCD_BASE]


def test_find_calibrator_filename_bases_skips_n_band(bcd_dir, tmp_path):
    """N band quadruplets are excluded from the calibrator bases."""
    data_dir = tmp_path / "nband"
    data_dir.mkdir()
    for mode in BCD_ALL_MODES:
        dest = data_dir / f"fake_nband_{mode}_noChop.fits"
        shutil.copy(bcd_dir / f"{BCD_BASE}_{mode}_noChop.fits", dest)
        with fits.open(dest, mode="update") as hdul:
            hdul[0].header["HIERARCH ESO DET CHIP NAME"] = "AQUARIUS"

    assert correction_module._has_all_bcd_files(data_dir, "fake_nband") is True
    assert correction_module._find_calibrator_filename_bases(data_dir) == []


def test_find_calibrator_filename_bases_skips_unreadable(
    bcd_dir, tmp_path, monkeypatch
):
    """A file that cannot be read at all is silently skipped."""
    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)

    class _RaisingReader:
        def __init__(self, path):
            self.path = path

        def read(self):
            raise OSError("corrupted file")

    monkeypatch.setattr(correction_module, "OIFitsReader", _RaisingReader)

    assert correction_module._find_calibrator_filename_bases(data_dir) == []


# ---------------------------------------------------------------------------
# correction: file pairs
# ---------------------------------------------------------------------------
def test_find_bcd_file_pairs_missing_folder(tmp_path, caplog):
    """A non-existent folder is warned about and yields no pair."""
    missing = tmp_path / "does_not_exist"

    pairs = correction_module._find_bcd_file_pairs(
        folders=[str(missing)], bcd_mode="IN_IN", chopping=False
    )

    assert pairs == []
    assert f"Folder does not exist: {missing}" in caplog.text


def test_find_bcd_file_pairs_without_out_out(bcd_dir, tmp_path):
    """A BCD file with no OUT_OUT counterpart is not paired."""
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    shutil.copy(bcd_dir / f"{BCD_BASE}_IN_IN_noChop.fits", lonely)

    pairs = correction_module._find_bcd_file_pairs(
        folders=[str(lonely)], bcd_mode="IN_IN", chopping=False
    )

    assert pairs == []


@pytest.mark.parametrize(("tau0_min", "n_pairs"), [(0.0, 1), (1e6, 0)])
def test_find_bcd_file_pairs_tau0_filter(bcd_dir, tau0_min, n_pairs, caplog):
    """Files below the tau0 threshold are rejected from the pair list."""
    pairs = correction_module._find_bcd_file_pairs(
        folders=[str(bcd_dir)], bcd_mode="IN_IN", chopping=False, tau0_min=tau0_min
    )

    assert len(pairs) == n_pairs
    assert f"Rejected {1 - n_pairs} files" in caplog.text
    if n_pairs:
        out_out, bcd = pairs[0]
        assert out_out.name == f"{BCD_BASE}_OUT_OUT_noChop.fits"
        assert bcd.name == f"{BCD_BASE}_IN_IN_noChop.fits"


def test_process_file_pair_warns_on_near_zero(bcd_dir, tmp_path, caplog):
    """Near-zero visibilities are replaced by NaN and reported."""
    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    bcd_path = data_dir / f"{BCD_BASE}_IN_IN_noChop.fits"
    with fits.open(bcd_path, mode="update") as hdul:
        hdul["OI_VIS2"].data["VIS2DATA"][:, :10] = 0.0

    config = BCDConfig(output_dir=tmp_path, spectral_binlen=118)
    with fits.open(bcd_path) as hdul:
        wavelengths = np.asarray(hdul[3].data["eff_wave"], dtype=float)

    corr_mean, corr_spectral = correction_module._process_file_pair(
        data_dir / f"{BCD_BASE}_OUT_OUT_noChop.fits", bcd_path, config, wavelengths
    )

    assert corr_mean.shape == (6,)
    assert corr_spectral.shape == (6, len(wavelengths))
    assert np.isnan(corr_spectral[:, :10]).all()
    assert "near-zero values in spectral data" in caplog.text


def test_process_file_pair_rejects_invalid_file(bcd_dir, tmp_path):
    """An invalid file yields (None, None) instead of a correction."""
    config = BCDConfig(output_dir=tmp_path, spectral_binlen=7)
    with fits.open(bcd_dir / f"{BCD_BASE}_IN_IN_noChop.fits") as hdul:
        wavelengths = np.asarray(hdul[3].data["eff_wave"], dtype=float)

    result = correction_module._process_file_pair(
        bcd_dir / f"{BCD_BASE}_OUT_OUT_noChop.fits",
        bcd_dir / f"{BCD_BASE}_IN_IN_noChop.fits",
        config,
        wavelengths,
    )

    assert result == (None, None)


# ---------------------------------------------------------------------------
# correction.compute_bcd_corrections
# ---------------------------------------------------------------------------
def test_compute_bcd_corrections_no_valid_file(bcd_dir, tmp_path):
    """All files rejected by validation raises 'No valid corrections computed'."""
    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    config = BCDConfig(output_dir=tmp_path / "out", spectral_binlen=7)

    with pytest.raises(ValueError, match="No valid corrections computed"):
        correction_module.compute_bcd_corrections(
            folders=[str(data_dir)], config=config, chopping=False, show_plots=False
        )


def test_compute_bcd_corrections_warns_on_broken_pair(bcd_dir, tmp_path, caplog):
    """A pair that fails mid-processing is warned about and skipped."""
    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    (data_dir / f"{BCD_BASE}_OUT_OUT_noChop.fits").write_text("not a FITS file")
    config = BCDConfig(output_dir=tmp_path / "out", spectral_binlen=118)

    with pytest.raises(ValueError, match="No valid corrections computed"):
        correction_module.compute_bcd_corrections(
            folders=[str(data_dir)], config=config, chopping=False, show_plots=False
        )

    assert f"Failed to process {BCD_BASE}_IN_IN_noChop.fits" in caplog.text


# ---------------------------------------------------------------------------
# correction: N band and saving
# ---------------------------------------------------------------------------
def test_fit_magic_numbers_n_band(tmp_path):
    """N band uses a single 8.2-12.0 µm window for the polynomial fit."""
    config = BCDConfig(output_dir=tmp_path, band="N", poly_order=1)
    wavs_um = np.linspace(8.0, 13.0, 60)
    line = 1.0 + 0.1 * wavs_um
    combined = [np.tile(line, (2, 1)) for _ in range(3)]

    poly_coef = correction_module.fit_magic_numbers(wavs_um, combined, config)

    assert poly_coef.shape == (2, 1, 2)
    assert np.allclose(poly_coef[0, 0], [0.1, 1.0])
    assert np.allclose(poly_coef[1, 0], [0.1, 1.0])


def test_save_corrections_n_band_writes_csv_and_summary(tmp_path):
    """_save_corrections builds the CSV itself and reports the N band window."""
    config = BCDConfig(output_dir=tmp_path, band="N", bcd_mode="OUT_OUT", poly_order=1)
    wavelengths = np.linspace(8.0e-6, 12.0e-6, 5)
    corrections = [np.full(5, float(i)) for i in range(6)]
    stds = [np.full(5, 0.1 * i) for i in range(6)]
    names = [f"B{i}" for i in range(6)]
    poly_coef = np.ones((2, 1, 2))

    outputs = correction_module._save_corrections(
        wavelengths=wavelengths,
        corrections=corrections,
        stds=stds,
        baseline_names=names,
        df=None,
        poly_coef=poly_coef,
        config=config,
        n_files=3,
    )

    df = pd.read_csv(outputs["csv"])
    assert list(df.columns) == ["wl"] + [c for n in names for c in (n, f"{n}_std")]
    assert np.allclose(df["wl"], wavelengths)
    assert np.allclose(df["B3"], 3.0)
    assert np.allclose(df["B3_std"], 0.3)

    poly_df = pd.read_csv(outputs["poly_csv"])
    assert len(poly_df) == 2
    assert poly_df["baseline_idx1"].isna().all()
    assert poly_df["wl_start_um"].tolist() == [8.2, 8.2]
    assert poly_df["wl_end_um"].tolist() == [12.0, 12.0]

    summary = (tmp_path / "bcd_OUT_OUT_summary.txt").read_text()
    assert "Window 0: 8.20 - 12.00 µm" in summary
    assert "Number of File Pairs:  3" in summary


# ---------------------------------------------------------------------------
# correction.apply_bcd_corrections
# ---------------------------------------------------------------------------
def test_apply_bcd_corrections_no_calibrator(tmp_path):
    """An empty data directory raises before any processing."""
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(ValueError, match="No calibrator files found"):
        correction_module.apply_bcd_corrections(empty, merge=False, plot=False)


def test_apply_bcd_corrections_missing_correction_file(bcd_dir, tmp_path):
    """Missing correction CSVs skip every file and raise a FileNotFoundError."""
    from matisse.core.utils.log_utils import console

    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    shutil.copy(
        data_dir / f"{BCD_BASE}_OUT_OUT_noChop.fits",
        data_dir / f"{BCD_BASE}_OUT_OUT_Chop.fits",
    )
    corrections_dir = tmp_path / "no_corrections"
    corrections_dir.mkdir()
    already_printed = len(console.file.getvalue())

    with pytest.raises(FileNotFoundError, match="Correction files are missing"):
        correction_module.apply_bcd_corrections(
            data_dir,
            corrections_dir=corrections_dir,
            chopping=True,
            merge=False,
            plot=False,
        )

    assert "Missing correction file" in console.file.getvalue()[already_printed:]


def test_apply_bcd_corrections_with_master_calibration(bcd_dir, tmp_path):
    """The bundled calibration corrects the swapped baselines and writes FITS."""
    from matisse.core.utils.log_utils import console

    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    for mode in BCD_ALL_MODES:
        shutil.copy(
            data_dir / f"{BCD_BASE}_{mode}_noChop.fits",
            data_dir / f"fake_second_IR-LM_LOW_{mode}_noChop.fits",
        )
    raw_in_in = fits.getdata(data_dir / f"{BCD_BASE}_IN_IN_noChop.fits", "OI_VIS2")
    raw_vis2 = np.array(raw_in_in["VIS2DATA"], dtype=float)
    already_printed = len(console.file.getvalue())

    correction_module.apply_bcd_corrections(
        data_dir, corrections_dir=None, merge=True, plot=True, verbose=True
    )

    out_dir = tmp_path / "bcd_corrected"
    for mode in BCD_ALL_MODES:
        assert (out_dir / f"{BCD_BASE}_{mode}_noChop_bcd_corr.fits").exists()
    assert (out_dir / f"{BCD_BASE}_noChop_bcd_corr.fits").exists()  # merged
    assert (
        "All corrections are GOOD quality" in console.file.getvalue()[already_printed:]
    )

    corrected = fits.open(out_dir / f"{BCD_BASE}_IN_IN_noChop_bcd_corr.fits")
    try:
        assert corrected[0].header["BCD_CORR"] is True
        assert corrected[0].header["BCD_CORR_MODE"] == "IN_IN"
        new_vis2 = np.array(corrected["OI_VIS2"].data["VIS2DATA"], dtype=float)
    finally:
        corrected.close()

    # Baselines 0 and 1 are untouched, the swapped pairs are rescaled.
    assert np.allclose(new_vis2[0], raw_vis2[0], equal_nan=True)
    assert not np.allclose(new_vis2[5], raw_vis2[5], equal_nan=True)


def test_apply_bcd_corrections_reports_poor_quality(bcd_dir, tmp_path):
    """Deliberately wrong coefficients are summarised as POOR quality."""
    from matisse.core.utils.log_utils import console

    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    corrections_dir = _write_poly_csv(tmp_path / "bad_corr", coef_x0=1.6)
    already_printed = len(console.file.getvalue())

    correction_module.apply_bcd_corrections(
        data_dir,
        corrections_dir=corrections_dir,
        merge=False,
        plot=False,
        verbose=False,
    )

    captured = console.file.getvalue()[already_printed:]
    assert "POOR" in captured
    assert "corrections show POOR quality" in captured


def test_apply_bcd_corrections_single_chopping(bcd_dir, tmp_path):
    """Chopped data are read and written with the Chop suffix."""
    data_dir = tmp_path / "chop"
    data_dir.mkdir()
    for mode in BCD_ALL_MODES:
        shutil.copy(
            bcd_dir / f"{BCD_BASE}_{mode}_noChop.fits",
            data_dir / f"{BCD_BASE}_{mode}_Chop.fits",
        )
    corrections_dir = _write_poly_csv(tmp_path / "corr", coef_x0=1.2)

    dict_data = correction_module._apply_bcd_corrections_single(
        data_dir, corrections_dir, BCD_BASE, chopping=True
    )
    out_dir = correction_module._save_corrected_oifits(
        dict_data, data_dir, BCD_BASE, chopping=True
    )

    assert set(dict_data) == set(BCD_ALL_MODES)
    assert sorted(p.name for p in out_dir.glob("*.fits")) == sorted(
        f"{BCD_BASE}_{mode}_Chop_bcd_corr.fits" for mode in BCD_ALL_MODES
    )
    raw = np.asarray(
        fits.getdata(data_dir / f"{BCD_BASE}_IN_IN_Chop.fits", "OI_VIS2")["VIS2DATA"]
    )
    # IN_IN storage order is [0, 1, 5, 4, 3, 2]: pair (2, 5) scales rows 5 and 2.
    assert np.allclose(dict_data["IN_IN"].vis2["VIS2"][5], raw[5] * 1.2)
    assert np.allclose(dict_data["IN_IN"].vis2["VIS2"][2], raw[2] / 1.2)


# ---------------------------------------------------------------------------
# visualization: helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ratio", "color"),
    [
        (0.0, "#2ca02c"),
        (1.0, "#2ca02c"),
        (1.01, "#ff7f0e"),
        (2.0, "#ff7f0e"),
        (2.01, "#d62728"),
        (10.0, "#d62728"),
    ],
)
def test_ratio_color_thresholds(ratio, color):
    """_ratio_color maps the quality ratio onto green/orange/red."""
    assert visualization_module._ratio_color(ratio) == color


def test_find_merged_file_returns_none(tmp_path):
    """Non-FITS, foreign and per-BCD-mode files are never taken as merged."""
    (tmp_path / "base_notes.txt").write_text("ignored")
    (tmp_path / "other_merged.fits").write_text("ignored")
    (tmp_path / "base_IN_IN_noChop_bcd_corr.fits").write_text("ignored")

    assert visualization_module._find_merged_file(tmp_path, "base") is None


def test_find_merged_file_finds_candidate(tmp_path):
    """A file without any BCD tag is returned as the merged file."""
    (tmp_path / "base_OUT_OUT_noChop_bcd_corr.fits").write_text("ignored")
    merged = tmp_path / "base_merged_bcd_corr.fits"
    merged.write_text("ignored")

    assert visualization_module._find_merged_file(tmp_path, "base") == merged


def test_save_all_plots_uses_open_figures(tmp_path):
    """_save_all_plots falls back to the currently open figures."""
    plt.figure()
    plt.plot([0, 1], [0, 1])

    visualization_module._save_all_plots(tmp_path)

    assert (tmp_path / "diagnostic_plot" / "bcd_diagnostic_1.png").exists()


def test_plot_corrections_n_band(tmp_path):
    """N band plots use the single 8.2-12.0 µm window and a full-range fit."""
    config = BCDConfig(
        output_dir=tmp_path,
        band="N",
        spectral_binlen=40,
        wavelength_low=8.2e-6,
        wavelength_high=12.0e-6,
        poly_order=1,
    )
    wavelengths = np.linspace(8.0e-6, 13.0e-6, 40)
    wavs_um = wavelengths * 1e6
    combined = [np.tile(1.0 + 0.01 * wavs_um, (2, 1)) for _ in range(3)]
    poly_coef = np.array([[[0.01, 1.0]], [[0.01, 1.0]]])

    figures = plot_corrections(
        wavelengths=wavelengths,
        corrections_mean=np.ones((2, 6)),
        corrections_spectral=np.ones((2, 6, 40)),
        combined_spectral=combined,
        poly_coef=poly_coef,
        config=config,
        save_plots=False,
    )

    assert len(figures) == 4
    fit_lines = [
        line
        for ax in figures[3].get_axes()
        for line in ax.get_lines()
        if line.get_label() == "Polynomial fit"
    ]
    assert len(fit_lines) == 2
    for line in fit_lines:
        assert len(line.get_ydata()) == len(wavs_um)
        assert np.allclose(line.get_ydata(), 1.0 + 0.01 * wavs_um)


def test_plot_poly_corrections_results_poly1(tmp_path):
    """Order-1 coefficients drive both the direct and inverse overlays."""
    pd.DataFrame(
        {
            "wavelength": [3.3e-6, 3.4e-6],
            "B0": [1.0, 1.1],
            "B0_std": [0.05, 0.05],
        }
    ).to_csv(tmp_path / "bcd_IN_IN_spectral_corrections.csv", index=False)
    pd.DataFrame(
        [
            {
                "baseline_idx1": 0,
                "baseline_idx2": 0,
                "window": 0,
                "wl_start_um": 3.3,
                "wl_end_um": 3.4,
                "coef_x1": 0.5,
                "coef_x0": 1.0,
            },
            {
                "baseline_idx1": 1,
                "baseline_idx2": 0,
                "window": 1,
                "wl_start_um": 4.55,
                "wl_end_um": 4.9,
                "coef_x1": 0.5,
                "coef_x0": 1.0,
            },
        ]
    ).to_csv(tmp_path / "bcd_IN_IN_poly_coeffs.csv", index=False)

    fig = plot_poly_corrections_results(tmp_path, bcd_mode="IN_IN")

    ax = fig.get_axes()[0]
    _, labels = ax.get_legend_handles_labels()
    assert "Polynomial fit" in labels
    assert "Polynomial fit (inv)" in labels
    direct = next(ln for ln in ax.get_lines() if ln.get_label() == "Polynomial fit")
    inverse = next(
        ln for ln in ax.get_lines() if ln.get_label() == "Polynomial fit (inv)"
    )
    assert np.allclose(direct.get_ydata(), 0.5 * direct.get_xdata() + 1.0)
    assert np.allclose(inverse.get_ydata(), 1.0 / (0.5 * inverse.get_xdata() + 1.0))
    plt.close(fig)


def test_plot_bcd_correction_without_reference_error(bcd_dir, tmp_path):
    """A zero OUT_OUT error leaves the ratio annotation empty."""
    from matisse.core.utils.oifits_reader import OIFitsReader

    dict_data = {}
    for mode in BCD_ALL_MODES:
        data = OIFitsReader(bcd_dir / f"{BCD_BASE}_{mode}_noChop.fits").read()
        if mode == "OUT_OUT":
            data.vis2["VIS2ERR"] = np.zeros_like(np.asarray(data.vis2["VIS2ERR"]))
        dict_data[mode] = data

    fig = visualization_module.plot_bcd_correction(dict_data, title="no error")

    empty_annotations = [
        txt
        for ax in fig.get_axes()
        for txt in ax.texts
        if txt.get_text() == "" and txt.get_color() == "#888888"
    ]
    assert len(empty_annotations) == 6
    plt.close(fig)


# ---------------------------------------------------------------------------
# visualization.compare_bcd_corrections
# ---------------------------------------------------------------------------
def test_compare_bcd_corrections_missing_directory(tmp_path):
    """A non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        visualization_module.compare_bcd_corrections(tmp_path / "nope")


def test_compare_bcd_corrections_without_merged_file(bcd_dir, tmp_path, caplog):
    """A complete group without merged file still produces one figure and a PDF."""
    corrected = _make_corrected_dir(bcd_dir, tmp_path / "bcd_corrected")
    corrections_dir = _write_poly_csv(tmp_path / "corr", coef_x0=1.0)

    figures = visualization_module.compare_bcd_corrections(
        corrected, corrections_dir=corrections_dir
    )

    assert len(figures) == 1
    assert (corrected / f"{BCD_BASE}_bcd_compare.pdf").exists()
    assert "No merged file found" in caplog.text
    for fig in figures:
        plt.close(fig)


def test_compare_bcd_corrections_chopped_group_with_merged(bcd_dir, tmp_path, caplog):
    """Chopped groups are detected, merged data overlaid and the PDF name sanitised."""
    corrected = _make_corrected_dir(bcd_dir, tmp_path / "bcd_corrected", "@@", "Chop")
    shutil.copy(
        bcd_dir / f"{BCD_BASE}_OUT_OUT_noChop.fits", corrected / "@@_merged.fits"
    )
    with fits.open(corrected / "@@_OUT_OUT_Chop_bcd_corr.fits", mode="update") as hdul:
        del hdul[0].header["ESO TPL START"]

    figures = visualization_module.compare_bcd_corrections(corrected)

    assert len(figures) == 1
    assert (corrected / "comparison_bcd_compare.pdf").exists()
    assert "Merged file found: @@_merged.fits" in caplog.text
    assert any("Group: @@" in txt.get_text() for txt in figures[0].texts)
    plt.close(figures[0])


@pytest.mark.parametrize("corrupt", [False, True])
def test_compare_bcd_corrections_skips_incomplete_group(
    bcd_dir, tmp_path, caplog, corrupt
):
    """A group missing (or unreadable in) one BCD mode is skipped with a warning."""
    corrected = _make_corrected_dir(bcd_dir, tmp_path / "bcd_corrected")
    broken = corrected / f"{BCD_BASE}_IN_IN_noChop_bcd_corr.fits"
    if corrupt:
        broken.write_text("not a FITS file")
    else:
        broken.unlink()

    figures = visualization_module.compare_bcd_corrections(corrected)

    assert figures == []
    assert "Missing BCD modes: IN_IN" in caplog.text
    assert not list(corrected.glob("*.pdf"))


def test_compute_bcd_corrections_warns_on_wavelength_mismatch(
    bcd_dir, tmp_path, caplog
):
    """A pair with a different wavelength grid is reported and skipped."""
    data_dir = _copy_bcd_dir(bcd_dir, tmp_path)
    for mode in ("OUT_OUT", "IN_IN"):
        shutil.copy(
            data_dir / f"{BCD_BASE}_{mode}_noChop.fits",
            data_dir / f"zz_other_IR-LM_LOW_{mode}_noChop.fits",
        )
    with fits.open(
        data_dir / "zz_other_IR-LM_LOW_IN_IN_noChop.fits", mode="update"
    ) as hdul:
        hdul[3].data = hdul[3].data[:50]

    config = BCDConfig(output_dir=tmp_path / "out", spectral_binlen=118)
    result = correction_module.compute_bcd_corrections(
        folders=[str(data_dir)], config=config, chopping=False, show_plots=False
    )

    assert result["n_files"] == 1
    assert "Wave shape seems to be different ((50,)/(118,))" in caplog.text
