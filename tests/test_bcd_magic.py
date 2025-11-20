import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from typer.testing import CliRunner

from matisse_pipeline.cli.main import app
from matisse_pipeline.core.bcd import correction as correction_module
from matisse_pipeline.core.bcd.config import BCDConfig
from matisse_pipeline.core.bcd.visualization import plot_corrections


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
            "magic_num",
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
    wav_file = tmp_path / "TESTMN_wav.npy"
    assert wav_file.exists()
    for i in range(6):
        assert (tmp_path / f"TESTMN_IN_IN{i}.npy").exists()

    png_files = sorted(tmp_path.glob("bcd_diagnostic_*.png"))
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
    result = runner.invoke(app, ["magic_num", str(tmp_path)])

    assert "File not found: No valid file" in caplog.text
    assert result.exit_code == 1
