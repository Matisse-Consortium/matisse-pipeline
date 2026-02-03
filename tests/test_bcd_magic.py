import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from matisse.cli.main import app
from matisse.core.bcd import correction as correction_module
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
            "magic",
            str(dataset_dir),
            "--output-dir",
            str(tmp_path),
            "--plot",
            "--no-chopping",
            "--prefix",
            "TESTMN",
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
    result = runner.invoke(app, ["magic", str(tmp_path)])

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
        ["magic", "--results-dir", str(tmp_path), "--tau0-min", str(tau0_min_value)],
    )

    assert result.exit_code == 0, result.output


def test_magic_cli_plots_existing_results_missing_csv(tmp_path):
    runner = CliRunner()

    result = runner.invoke(app, ["magic", "--results-dir", str(tmp_path)])

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
