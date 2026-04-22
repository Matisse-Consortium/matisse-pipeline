from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from matisse.core.flux import airmass, calibrator_spectrum, databases, utils


def _make_hdul(**header_values: str | float) -> fits.HDUList:
    hdu = fits.PrimaryHDU()
    for key, value in header_values.items():
        hdu.header[key] = value
    return fits.HDUList([hdu])


def test_get_dlambda_hawaii_low():
    hdul = _make_hdul(
        **{
            "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
            "HIERARCH ESO INS DIL NAME": "LOW",
        }
    )
    try:
        assert utils.get_dlambda(hdul) == 8.0
    finally:
        hdul.close()


def test_get_dlambda_unknown_detector_returns_nan():
    hdul = _make_hdul(
        **{
            "HIERARCH ESO DET CHIP NAME": "UNKNOWN-DET",
            "HIERARCH ESO INS DIL NAME": "LOW",
        }
    )
    try:
        assert np.isnan(utils.get_dlambda(hdul))
    finally:
        hdul.close()


def test_get_dl_coeffs_aquarius_high():
    hdul = _make_hdul(
        **{
            "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
            "HIERARCH ESO INS DIN NAME": "HIGH",
        }
    )
    try:
        coeffs = utils.get_dl_coeffs(hdul)
    finally:
        hdul.close()

    assert len(coeffs) == 4
    assert coeffs[0] == pytest.approx(-8.02e-05, rel=1e-3)


def test_get_spectral_average_found_and_missing():
    hdul_found = _make_hdul(
        **{
            "HIERARCH ESO PRO REC1 PARAM1 NAME": "otherParam",
            "HIERARCH ESO PRO REC1 PARAM2 NAME": "spectralAverage",
            "HIERARCH ESO PRO REC1 PARAM2 VALUE": 3,
        }
    )
    hdul_missing = _make_hdul()

    try:
        assert utils.get_spectral_average(hdul_found) == 3.0
        assert np.isnan(utils.get_spectral_average(hdul_missing))
    finally:
        hdul_found.close()
        hdul_missing.close()


def test_find_nearest_idx():
    idx = utils.find_nearest_idx([1.0, 2.4, 3.8, 4.9], 3.6)
    assert idx == 2


def test_get_cal_databases_dir_uses_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MATISSE_CAL_DB_PATH", str(tmp_path))
    out = databases.get_cal_databases_dir()
    assert out == tmp_path.resolve()


def test_get_cal_databases_dir_raises_when_zenodo_fails(monkeypatch):
    monkeypatch.delenv("MATISSE_CAL_DB_PATH", raising=False)
    monkeypatch.setattr(
        databases,
        "_ensure_pooch_cache",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    with pytest.raises(RuntimeError, match="Calibrator spectral databases not found"):
        databases.get_cal_databases_dir()


def test_prefetch_databases_calls_pooch(monkeypatch):
    """prefetch_databases delegates to _ensure_pooch_cache."""
    called = {}

    def fake_ensure():
        called["yes"] = True
        return Path("/fake/cache")

    monkeypatch.setattr(databases, "_ensure_pooch_cache", fake_ensure)
    result = databases.prefetch_databases()
    assert result == Path("/fake/cache")
    assert called.get("yes")


def test_database_status_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MATISSE_CAL_DB_PATH", str(tmp_path))
    (tmp_path / "vBoekelDatabase.fits").touch()

    status = databases.database_status()

    assert status["vBoekelDatabase.fits"] == "local_override"
    assert status["calib_spec_db_v10.fits"] == "missing"
    assert status["calib_spec_db_v10_supplement.fits"] == "missing"


def test_database_status_missing_when_no_cache(monkeypatch, tmp_path):
    import sys
    import types

    monkeypatch.delenv("MATISSE_CAL_DB_PATH", raising=False)

    cache_root = tmp_path / "cache_root"
    cache_root.mkdir()
    fake_pooch = types.SimpleNamespace(os_cache=lambda _name: cache_root)
    monkeypatch.setitem(sys.modules, "pooch", fake_pooch)

    status = databases.database_status()

    assert status["vBoekelDatabase.fits"] == "missing"
    assert status["calib_spec_db_v10.fits"] == "missing"
    assert status["calib_spec_db_v10_supplement.fits"] == "missing"


def test_run_skycalc_returns_false_when_cli_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(airmass, "_find_skycalc_cli", lambda: None)

    ok = airmass.run_skycalc(tmp_path / "in.txt", tmp_path / "out.fits")

    assert ok is False


def test_compute_airmass_correction_returns_ones_without_skycalc(monkeypatch, tmp_path):
    monkeypatch.setattr(airmass, "run_skycalc", lambda *_args, **_kwargs: False)

    hdul_sci = _make_hdul(
        **{
            "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
            "HIERARCH ESO INS DIL NAME": "LOW",
        }
    )
    hdul_cal = _make_hdul(
        **{
            "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
            "HIERARCH ESO INS DIL NAME": "LOW",
        }
    )
    wav_sci_m = np.array([3.0e-6, 3.2e-6, 3.4e-6])
    wav_cal_m = np.array([3.0e-6, 3.2e-6, 3.4e-6])

    try:
        corr = airmass.compute_airmass_correction(
            hdul_sci=hdul_sci,
            hdul_cal=hdul_cal,
            wav_sci_m=wav_sci_m,
            wav_cal_m=wav_cal_m,
            airmass_sci=1.1,
            airmass_cal=1.2,
            pwv_sci=2.0,
            pwv_cal=2.0,
            output_dir=tmp_path,
            tag_sci="SCI",
            tag_cal="CAL",
        )
    finally:
        hdul_sci.close()
        hdul_cal.close()

    assert np.array_equal(corr, np.ones_like(wav_sci_m))


# ============================================================================
# Tests for calibrator_spectrum.py
# ============================================================================


@pytest.fixture
def mock_vboekel_database(tmp_path):
    """Create a minimal mock vBoekelDatabase.fits for testing."""
    path = tmp_path / "vBoekelDatabase.fits"

    primary = fits.PrimaryHDU()
    primary.header["HIERARCH ESO PRO CATG"] = "vBoekelDatabase"

    # Minimal SOURCES extension (needed for match_radius logic)
    sources_cols = fits.ColDefs(
        [
            fits.Column(
                name="NAME", format="20A", array=np.array([b"STAR1", b"STAR2"])
            ),
            fits.Column(name="RAEPP", format="D", array=np.array([10.0, 20.0])),
            fits.Column(name="DECEPP", format="D", array=np.array([-30.0, -40.0])),
            fits.Column(
                name="DIAMETER", format="D", array=np.array([0.005, 0.010])
            ),  # mas / 1000
            fits.Column(
                name="DIAMETER_ERR", format="D", array=np.array([0.0005, 0.001])
            ),
        ]
    )
    sources_hdu = fits.BinTableHDU.from_columns(sources_cols, name="SOURCES")

    # DIAMETERS extension (for new format compatibility)
    diameters_cols = fits.ColDefs(
        [
            fits.Column(
                name="DIAMETER", format="D", array=np.array([0.005, 0.010])
            ),  # mas / 1000
            fits.Column(
                name="DIAMETER_ERR", format="D", array=np.array([0.0005, 0.001])
            ),
        ]
    )
    diameters_hdu = fits.BinTableHDU.from_columns(diameters_cols, name="DIAMETERS")

    # Dummy extensions to reach offset 9 for spectrum lookup
    dummy_hdus = [fits.ImageHDU() for _ in range(6)]

    # Spectrum extension for the second source (at index 1 → offset +9)
    spec_cols = fits.ColDefs(
        [
            fits.Column(
                name="WAVELENGTH",
                format="E",
                array=np.array([3.0e-6, 3.5e-6, 4.0e-6], dtype=np.float32),
            ),
            fits.Column(
                name="FLUX",
                format="E",
                array=np.array([100.0, 95.0, 90.0], dtype=np.float32),
            ),
        ]
    )
    spec_hdu = fits.BinTableHDU.from_columns(spec_cols)
    spec_hdu.header["NAME"] = "STAR2_MODEL"

    hdul = fits.HDUList([primary, sources_hdu, diameters_hdu] + dummy_hdus + [spec_hdu])
    hdul.writeto(path, overwrite=True)
    return path


def test_calibrator_spectrum_dataclass_creation():
    spec = calibrator_spectrum.CalibratorSpectrum(
        name="TestStar",
        diameter_mas=5.0,
        diameter_err_mas=0.5,
        wavelength=np.array([3.0e-6, 4.0e-6]),
        flux=np.array([100.0, 90.0]),
        database="test.fits",
        separation_arcsec=1.5,
        ra_deg=10.0,
        dec_deg=45.0,
    )

    assert spec.name == "TestStar"
    assert spec.diameter_mas == 5.0
    assert len(spec.wavelength) == 2


def test_lookup_local_database_file_not_found(tmp_path):
    missing = tmp_path / "missing.fits"
    result = calibrator_spectrum.lookup_local_database(
        missing, ra_deg=10.0, dec_deg=45.0
    )
    assert result is None


def test_lookup_calibrator_spectrum_returns_none_when_all_fail(monkeypatch, tmp_path):
    # Mock both STARSFLUX and local lookup to fail
    monkeypatch.setattr(
        calibrator_spectrum, "lookup_starsflux", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        calibrator_spectrum,
        "lookup_local_database",
        lambda *_args, **_kwargs: None,
    )

    result = calibrator_spectrum.lookup_calibrator_spectrum(
        cal_name="UNKNOWN",
        ra_deg=10.0,
        dec_deg=45.0,
        cal_database_paths=[tmp_path / "nonexistent.fits"],
    )

    assert result is None


def test_lookup_starsflux_no_astroquery(monkeypatch):
    """Test that lookup_starsflux gracefully fails without astroquery."""

    def mock_import(name, *args, **kwargs):
        if "astroquery" in name:
            raise ImportError("no astroquery")
        return __import__(name)

    monkeypatch.setattr("builtins.__import__", mock_import)

    result = calibrator_spectrum.lookup_starsflux("STAR", 10.0, 45.0)
    assert result is None


def test_lookup_starsflux_timeout_returns_none(monkeypatch):
    """Timeouts from optional STARSFLUX access should fall back cleanly."""

    class FakeVizier:
        @staticmethod
        def query_object(*_args, **_kwargs):
            raise TimeoutError("The read operation timed out")

    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "astroquery.vizier":

            class FakeModule:
                Vizier = FakeVizier

            return FakeModule()
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    result = calibrator_spectrum.lookup_starsflux("HD26546", 63.13, 17.0)

    assert result is None


def test_lookup_starsflux_realstar():
    """Test that lookup_starsflux can find a real star (HD26546) in the STARSFLUX database."""
    result = calibrator_spectrum.lookup_starsflux("HD26546", 63.13, 17.0)

    if result is None:
        pytest.skip("STARSFLUX unavailable or timed out")

    wavelengths = result.wavelength
    flux = result.flux

    assert result is not None
    assert result.name == "HD26546"
    assert result.diameter_mas == pytest.approx(0.45, abs=0.01)
    assert len(wavelengths) == len(flux)
