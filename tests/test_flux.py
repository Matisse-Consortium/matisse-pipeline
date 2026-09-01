from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from matisse.core.flux import (
    airmass,
    calibrator_spectrum,
    databases,
    diagnostics,
    utils,
)


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


# ============================================================================
# Tests for diagnostics.py
# ============================================================================


def _make_calibrated_hdul(n_wav: int = 20, n_tel: int = 2) -> fits.HDUList:
    """Build a minimal calibrated OIFITS HDUList for diagnostics tests."""
    wav = np.linspace(3.0e-6, 4.0e-6, n_wav)
    wav_col = fits.Column(name="eff_wave", format="E", array=wav.astype(np.float32))
    oi_wav = fits.BinTableHDU.from_columns([wav_col], name="OI_WAVELENGTH")

    sta_indices = np.arange(1, n_tel + 1, dtype=np.int16)
    sta_names = np.array([f"U{i}" for i in range(1, n_tel + 1)])

    arr_cols = fits.ColDefs(
        [
            fits.Column(name="STA_INDEX", format="I", array=sta_indices),
            fits.Column(name="STA_NAME", format="4A", array=sta_names),
        ]
    )
    oi_array = fits.BinTableHDU.from_columns(arr_cols, name="OI_ARRAY")

    fluxdata = np.ones((n_tel, n_wav), dtype=np.float32) * 10.0
    fluxerr = np.ones((n_tel, n_wav), dtype=np.float32) * 0.5
    flux_cols = fits.ColDefs(
        [
            fits.Column(name="FLUXDATA", format=f"{n_wav}E", array=fluxdata),
            fits.Column(name="FLUXERR", format=f"{n_wav}E", array=fluxerr),
            fits.Column(name="STA_INDEX", format="I", array=sta_indices),
        ]
    )
    oi_flux = fits.BinTableHDU.from_columns(flux_cols, name="OI_FLUX")

    n_bl = n_tel * (n_tel - 1) // 2 or 1
    visamp = np.ones((n_bl, n_wav), dtype=np.float32) * 5.0
    bl_sta = np.array([[1, 2]] * n_bl, dtype=np.int16)
    ucoord = np.ones(n_bl, dtype=np.float32) * 50.0
    vcoord = np.ones(n_bl, dtype=np.float32) * 30.0
    vis_cols = fits.ColDefs(
        [
            fits.Column(name="VISAMP", format=f"{n_wav}E", array=visamp),
            fits.Column(name="STA_INDEX", format="2I", array=bl_sta),
            fits.Column(name="UCOORD", format="E", array=ucoord),
            fits.Column(name="VCOORD", format="E", array=vcoord),
        ]
    )
    oi_vis = fits.BinTableHDU.from_columns(vis_cols, name="OI_VIS")

    return fits.HDUList([fits.PrimaryHDU(), oi_wav, oi_array, oi_flux, oi_vis])


def test_plot_calibrator_spectrum_skips_when_fig_dir_none():
    diagnostics.plot_calibrator_spectrum(
        fig_dir=None,
        cal_name="TEST",
        band="IR-LM",
        wav_model=np.linspace(3e-6, 4e-6, 50),
        flux_model=np.ones(50) * 10.0,
        wav_obs=np.linspace(3e-6, 4e-6, 20),
        spectrum_resampled=np.ones(20) * 10.0,
    )


def test_plot_calibrator_spectrum_creates_file(tmp_path):
    wav = np.linspace(3e-6, 4e-6, 50)
    wav_obs = np.linspace(3e-6, 4e-6, 20)
    diagnostics.plot_calibrator_spectrum(
        fig_dir=tmp_path,
        cal_name="HD 12345",
        band="IR-LM",
        wav_model=wav,
        flux_model=np.ones(50) * 8.0,
        wav_obs=wav_obs,
        spectrum_resampled=np.ones(20) * 8.0,
        is_dense_model=True,
    )
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_plot_calibrator_spectrum_sparse_model(tmp_path):
    wav_obs = np.linspace(3e-6, 4e-6, 20)
    diagnostics.plot_calibrator_spectrum(
        fig_dir=tmp_path,
        cal_name="TestCal",
        band="IR-N",
        wav_model=wav_obs,
        flux_model=np.ones(20) * 5.0,
        wav_obs=wav_obs,
        spectrum_resampled=np.ones(20) * 5.0,
        is_dense_model=False,
    )
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_plot_calibrator_spectrum_with_hdul(tmp_path):
    wav_obs = np.linspace(3e-6, 4e-6, 20)
    hdul = _make_calibrated_hdul(n_wav=20)
    try:
        diagnostics.plot_calibrator_spectrum(
            fig_dir=tmp_path,
            cal_name="TestCal",
            band="IR-LM",
            wav_model=wav_obs,
            flux_model=np.ones(20) * 5.0,
            wav_obs=wav_obs,
            spectrum_resampled=np.ones(20) * 5.0,
            hdul_cal=hdul,
            diameter_mas=2.5,
        )
    finally:
        hdul.close()
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_plot_airmass_correction_skips_when_fig_dir_none():
    diagnostics.plot_airmass_correction(
        fig_dir=None,
        wav_sci_m=np.linspace(3e-6, 4e-6, 20),
        airmass_correction=np.ones(20),
        output_tag="test",
    )


def test_plot_airmass_correction_creates_file(tmp_path):
    wav = np.linspace(3e-6, 5e-6, 40)
    corr = 1.0 + 0.1 * np.sin(np.linspace(0, np.pi, 40))
    diagnostics.plot_airmass_correction(
        fig_dir=tmp_path,
        wav_sci_m=wav,
        airmass_correction=corr,
        output_tag="sci_cal",
    )
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_plot_calibrated_flux_skips_when_fig_dir_none():
    hdul = _make_calibrated_hdul()
    try:
        diagnostics.plot_calibrated_flux(
            fig_dir=None,
            hdul_out=hdul,
            cal_name="CAL",
            sci_name="SCI",
            mode="flux",
            band="IR-LM",
            bcd="OUT_OUT",
        )
    finally:
        hdul.close()


@pytest.mark.parametrize("mode", ["flux", "corrflux", "both"])
def test_plot_calibrated_flux_modes(tmp_path, mode):
    hdul = _make_calibrated_hdul()
    try:
        diagnostics.plot_calibrated_flux(
            fig_dir=tmp_path,
            hdul_out=hdul,
            cal_name="CAL",
            sci_name="SCI",
            mode=mode,
            band="IR-LM",
            bcd="OUT_OUT",
            dark_mode=True,
        )
    finally:
        hdul.close()
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_plot_calibrated_flux_light_mode(tmp_path):
    hdul = _make_calibrated_hdul()
    try:
        diagnostics.plot_calibrated_flux(
            fig_dir=tmp_path,
            hdul_out=hdul,
            cal_name="CAL",
            sci_name="SCI",
            mode="both",
            band="IR-N",
            bcd="IN_IN",
            dark_mode=False,
        )
    finally:
        hdul.close()
    assert len(list(tmp_path.glob("*.png"))) == 1


def test_plot_calibrated_flux_with_spectral_features(tmp_path):
    hdul = _make_calibrated_hdul()
    try:
        diagnostics.plot_calibrated_flux(
            fig_dir=tmp_path,
            hdul_out=hdul,
            cal_name="CAL",
            sci_name="SCI",
            mode="flux",
            band="IR-LM",
            bcd="OUT_OUT",
            spectral_features=True,
        )
    finally:
        hdul.close()
    assert len(list(tmp_path.glob("*.png"))) == 1


# ============================================================================
# Additional coverage: databases.py
# ============================================================================


def _install_fake_pooch(monkeypatch, cache_root: Path, retriever=None):
    """Install a minimal fake ``pooch`` module in ``sys.modules``."""
    import sys
    import types

    fake_pooch = types.SimpleNamespace(
        os_cache=lambda _name: cache_root,
        Untar=lambda: "untar-processor",
    )
    if retriever is not None:
        fake_pooch.create = retriever
    monkeypatch.setitem(sys.modules, "pooch", fake_pooch)
    return fake_pooch


def _disable_pooch_import(monkeypatch):
    """Make ``import pooch`` raise ImportError inside the databases module."""
    import sys

    monkeypatch.setitem(sys.modules, "pooch", None)


class _FakeZenodoResponse:
    """Minimal context-manager stand-in for an ``urlopen`` response."""

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def test_get_cal_databases_dir_ignores_missing_env_override(
    monkeypatch, tmp_path, caplog
):
    """A non-existent MATISSE_CAL_DB_PATH is warned about and ignored."""
    missing = tmp_path / "not_here"
    monkeypatch.setenv("MATISSE_CAL_DB_PATH", str(missing))
    monkeypatch.setattr(databases, "_ensure_pooch_cache", lambda: tmp_path / "cache")

    with caplog.at_level("WARNING", logger=databases.logger.name):
        out = databases.get_cal_databases_dir()

    assert out == tmp_path / "cache"
    assert "does not exist" in caplog.text


def test_pooch_is_available_false_without_pooch(monkeypatch):
    """_pooch_is_available reports False when pooch cannot be imported."""
    _disable_pooch_import(monkeypatch)
    assert databases._pooch_is_available() is False


def test_find_cached_databases_none_without_pooch(monkeypatch):
    """_find_cached_databases returns None when pooch cannot be imported."""
    _disable_pooch_import(monkeypatch)
    assert databases._find_cached_databases() is None


def test_find_cached_databases_none_when_cache_has_no_fits(monkeypatch, tmp_path):
    """An empty extraction directory yields no cached database dir."""
    cache_root = tmp_path / "pooch"
    untar = cache_root / "cal_databases" / "v1.0.0" / f"{databases._ARCHIVE_NAME}.untar"
    untar.mkdir(parents=True)
    (untar / "README.txt").write_text("no fits here")
    _install_fake_pooch(monkeypatch, cache_root)

    assert databases._find_cached_databases() is None


def test_find_cached_databases_returns_extracted_dir(monkeypatch, tmp_path):
    """_find_cached_databases returns the directory holding the FITS files."""
    cache_root = tmp_path / "pooch"
    db_dir = (
        cache_root
        / "cal_databases"
        / "v1.0.0"
        / f"{databases._ARCHIVE_NAME}.untar"
        / "databases"
    )
    db_dir.mkdir(parents=True)
    for fname in databases._DB_FILES:
        (db_dir / fname).touch()
    _install_fake_pooch(monkeypatch, cache_root)

    assert databases._find_cached_databases() == db_dir


def test_resolve_latest_zenodo_record_parses_payload(monkeypatch):
    """The Zenodo API payload is parsed into (record_id, version) without the 'v'."""
    import json

    payload = json.dumps({"id": 123, "metadata": {"version": "v1.2.3"}}).encode()
    monkeypatch.setattr(
        databases.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeZenodoResponse(payload),
    )

    assert databases._resolve_latest_zenodo_record() == ("123", "1.2.3")


def test_resolve_latest_zenodo_record_falls_back_to_record_id(monkeypatch):
    """Without a version in the metadata the record id is used as version."""
    import json

    payload = json.dumps({"id": 456, "metadata": {}}).encode()
    monkeypatch.setattr(
        databases.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeZenodoResponse(payload),
    )

    assert databases._resolve_latest_zenodo_record() == ("456", "456")


def test_resolve_latest_zenodo_record_raises_on_network_error(monkeypatch):
    """A failing urlopen is wrapped into a RuntimeError naming the concept record."""

    def boom(*_args, **_kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(databases.urllib.request, "urlopen", boom)

    with pytest.raises(RuntimeError, match="Failed to query Zenodo API"):
        databases._resolve_latest_zenodo_record()


def test_ensure_pooch_cache_raises_without_pooch(monkeypatch):
    """_ensure_pooch_cache raises a helpful ImportError when pooch is absent."""
    _disable_pooch_import(monkeypatch)

    with pytest.raises(ImportError, match="pooch is required"):
        databases._ensure_pooch_cache()


def _make_fake_retriever(record: dict, extracted: list[str]):
    """Build a fake ``pooch.create`` recording its kwargs and returning a fetcher."""

    class _Retriever:
        @staticmethod
        def fetch(fname, processor=None):
            record["fetched"] = fname
            record["processor"] = processor
            return extracted

    def create(**kwargs):
        record.update(kwargs)
        return _Retriever()

    return create


def test_ensure_pooch_cache_downloads_latest_version(monkeypatch, tmp_path):
    """The resolved Zenodo record drives the pooch base_url and version."""
    cache_root = tmp_path / "pooch"
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    extracted = [
        str(extract_dir / "README.txt"),
        str(extract_dir / "vBoekelDatabase.fits"),
    ]
    record: dict = {}
    _install_fake_pooch(
        monkeypatch, cache_root, retriever=_make_fake_retriever(record, extracted)
    )
    monkeypatch.setattr(
        databases, "_resolve_latest_zenodo_record", lambda: ("999", "2.0.0")
    )

    out = databases._ensure_pooch_cache()

    assert out == extract_dir
    assert record["version"] == "v2.0.0"
    assert record["base_url"] == "https://zenodo.org/records/999/files/"
    assert record["path"] == cache_root / "cal_databases"
    assert record["fetched"] == databases._ARCHIVE_NAME
    assert record["processor"] == "untar-processor"


def test_ensure_pooch_cache_uses_pinned_version_when_api_fails(monkeypatch, tmp_path):
    """An unreachable Zenodo API falls back to the pinned record and version."""
    cache_root = tmp_path / "pooch"
    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir()
    extracted = [str(extract_dir / "calib_spec_db_v10.fits")]
    record: dict = {}
    _install_fake_pooch(
        monkeypatch, cache_root, retriever=_make_fake_retriever(record, extracted)
    )

    def boom():
        raise RuntimeError("zenodo unreachable")

    monkeypatch.setattr(databases, "_resolve_latest_zenodo_record", boom)

    out = databases._ensure_pooch_cache()

    assert out == extract_dir
    assert record["version"] == f"v{databases._ZENODO_PINNED_VERSION}"
    assert (
        record["base_url"]
        == f"https://zenodo.org/records/{databases._ZENODO_CONCEPT_RECORD_ID}/files/"
    )


def test_ensure_pooch_cache_returns_cached_dir(monkeypatch, tmp_path):
    """Already-extracted databases short-circuit the download."""
    cache_root = tmp_path / "pooch"
    db_dir = (
        cache_root
        / "cal_databases"
        / "v1.0.0"
        / f"{databases._ARCHIVE_NAME}.untar"
        / "databases"
    )
    db_dir.mkdir(parents=True)
    (db_dir / "vBoekelDatabase.fits").touch()
    _install_fake_pooch(monkeypatch, cache_root)

    assert databases._ensure_pooch_cache() == db_dir


def test_locate_extracted_fits_returns_parent(tmp_path):
    """_locate_extracted_fits returns the directory holding the FITS files."""
    out = databases._locate_extracted_fits(
        [str(tmp_path / "notes.txt"), str(tmp_path / "vBoekelDatabase.fits")]
    )
    assert out == tmp_path


def test_locate_extracted_fits_raises_without_fits(tmp_path):
    """_locate_extracted_fits raises when the archive contains no FITS file."""
    with pytest.raises(RuntimeError, match="No FITS files found"):
        databases._locate_extracted_fits([str(tmp_path / "notes.txt")])


def test_database_status_reports_cached(monkeypatch, tmp_path):
    """Databases present in the pooch cache are reported as 'cached'."""
    monkeypatch.delenv("MATISSE_CAL_DB_PATH", raising=False)
    cache_root = tmp_path / "pooch"
    db_dir = (
        cache_root
        / "cal_databases"
        / "v1.0.0"
        / f"{databases._ARCHIVE_NAME}.untar"
        / "databases"
    )
    db_dir.mkdir(parents=True)
    for fname in databases._DB_FILES:
        (db_dir / fname).touch()
    _install_fake_pooch(monkeypatch, cache_root)

    status = databases.database_status()

    assert status == dict.fromkeys(databases._DB_FILES, "cached")


def test_database_status_reports_pooch_missing(monkeypatch):
    """Without pooch every database is reported as 'missing (pooch not installed)'."""
    monkeypatch.delenv("MATISSE_CAL_DB_PATH", raising=False)
    _disable_pooch_import(monkeypatch)

    status = databases.database_status()

    assert status == dict.fromkeys(databases._DB_FILES, "missing (pooch not installed)")


# ============================================================================
# Additional coverage: calibrator_spectrum.py
# ============================================================================


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (TimeoutError("read timed out"), True),
        (OSError("socket failure"), True),
        (RuntimeError("The read operation Timed Out"), True),
        (RuntimeError("Connection reset by peer"), True),
        (RuntimeError("Temporary failure in name resolution"), True),
        (RuntimeError("Name or service not known"), True),
        (RuntimeError("Network is unreachable"), True),
        (RuntimeError("Remote end closed connection without response"), True),
        (RuntimeError("query timeout after 30s"), True),
        (ValueError("bad column"), False),
        (KeyError("SOURCES"), False),
    ],
)
def test_is_remote_lookup_error(error, expected):
    """Network-flavoured exceptions are classified as remote lookup failures."""
    assert calibrator_spectrum._is_remote_lookup_error(error) is expected


def _make_diameter_hdul() -> fits.HDUList:
    """Build an HDUList whose DIAMETERS table is also the last-but-one extension."""
    cols = fits.ColDefs(
        [
            fits.Column(name="DIAMETER", format="D", array=np.array([0.005, 0.010])),
            fits.Column(
                name="DIAMETER_ERR", format="D", array=np.array([0.0005, 0.001])
            ),
        ]
    )
    diameters = fits.BinTableHDU.from_columns(cols, name="DIAMETERS")
    return fits.HDUList([fits.PrimaryHDU(), diameters, fits.ImageHDU()])


@pytest.mark.parametrize("is_old_format", [True, False])
def test_read_diameter_vboekel_converts_to_mas(is_old_format):
    """vBoekel diameters are converted from arcsec-like units to mas (x1000)."""
    hdul = _make_diameter_hdul()
    try:
        diam, diam_err = calibrator_spectrum._read_diameter_vboekel(
            hdul, 1, is_old_format
        )
    finally:
        hdul.close()

    assert diam == pytest.approx(10.0)
    assert diam_err == pytest.approx(1.0)


def _make_calib_spec_sources(
    *,
    uddl: float,
    uddn: float,
    e_diam_est: float,
    diam_gaia: float,
) -> fits.BinTableHDU:
    """Build a minimal calib_spec_db SOURCES table with the diameter columns."""
    nan = float("nan")
    cols = fits.ColDefs(
        [
            fits.Column(name="name", format="20A", array=np.array([b"CALSTAR"])),
            fits.Column(name="ra", format="D", array=np.array([10.0])),
            fits.Column(name="dec", format="D", array=np.array([-30.0])),
            fits.Column(name="UDDL_est", format="D", array=np.array([uddl])),
            fits.Column(name="UDDN_est", format="D", array=np.array([uddn])),
            fits.Column(name="e_diam_est", format="D", array=np.array([e_diam_est])),
            fits.Column(name="diam_midi", format="D", array=np.array([nan])),
            fits.Column(name="e_diam_midi", format="D", array=np.array([nan])),
            fits.Column(name="diam_cohen", format="D", array=np.array([nan])),
            fits.Column(name="e_diam_cohen", format="D", array=np.array([nan])),
            fits.Column(name="UDD_meas", format="D", array=np.array([nan])),
            fits.Column(name="e_diam_meas", format="D", array=np.array([nan])),
            fits.Column(name="diam_gaia", format="D", array=np.array([diam_gaia])),
        ]
    )
    return fits.BinTableHDU.from_columns(cols, name="SOURCES")


@pytest.mark.parametrize(
    ("band", "expected"),
    [("L", 3.0), ("N", 7.0)],
)
def test_read_diameter_calib_spec_primary_columns(band, expected):
    """The band-dependent estimated UD diameter is preferred when available."""
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            _make_calib_spec_sources(
                uddl=3.0, uddn=7.0, e_diam_est=0.25, diam_gaia=9.0
            ),
        ]
    )
    try:
        diam, diam_err = calibrator_spectrum._read_diameter_calib_spec(hdul, 0, band)
    finally:
        hdul.close()

    assert diam == pytest.approx(expected)
    assert diam_err == pytest.approx(0.25)


def test_read_diameter_calib_spec_falls_back_to_gaia():
    """With every other column NaN, diam_gaia is used with a 10% uncertainty."""
    nan = float("nan")
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            _make_calib_spec_sources(uddl=nan, uddn=nan, e_diam_est=nan, diam_gaia=2.0),
        ]
    )
    try:
        diam, diam_err = calibrator_spectrum._read_diameter_calib_spec(hdul, 0, "L")
    finally:
        hdul.close()

    assert diam == pytest.approx(2.0)
    assert diam_err == pytest.approx(0.2)


def _make_spectrum_hdu(name: str) -> fits.BinTableHDU:
    """Build a small WAVELENGTH/FLUX spectrum extension."""
    cols = fits.ColDefs(
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
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.header["NAME"] = name
    return hdu


def test_lookup_local_database_vboekel_new_format(mock_vboekel_database):
    """The SOURCES extension is used for the current vBoekel format."""
    result = calibrator_spectrum.lookup_local_database(
        mock_vboekel_database, ra_deg=10.0, dec_deg=-30.0
    )

    assert result is not None
    assert result.name == "STAR2_MODEL"
    assert result.diameter_mas == pytest.approx(5.0)
    assert result.diameter_err_mas == pytest.approx(0.5)
    assert result.ra_deg == pytest.approx(10.0)
    assert result.dec_deg == pytest.approx(-30.0)
    assert result.separation_arcsec < 1.0
    assert result.flux[0] == pytest.approx(100.0)


def test_lookup_local_database_unknown_name_uses_vboekel_layout(
    mock_vboekel_database, tmp_path
):
    """A database with an unrecognised filename falls back to the vBoekel layout."""
    other = tmp_path / "other_database.fits"
    other.write_bytes(Path(mock_vboekel_database).read_bytes())

    result = calibrator_spectrum.lookup_local_database(
        other, ra_deg=10.0, dec_deg=-30.0
    )

    assert result is not None
    assert result.database == "other_database.fits"
    assert result.name == "STAR2_MODEL"
    assert result.diameter_mas == pytest.approx(5.0)


def test_lookup_local_database_returns_none_when_far(mock_vboekel_database):
    """No source within match_radius yields None."""
    result = calibrator_spectrum.lookup_local_database(
        mock_vboekel_database, ra_deg=200.0, dec_deg=60.0
    )
    assert result is None


def test_lookup_local_database_vboekel_old_format(tmp_path):
    """The old vBoekel layout reads sources from extension 8 and diameters from [-2]."""
    path = tmp_path / "vBoekelDatabase.fitsold"

    sources_cols = fits.ColDefs(
        [
            fits.Column(name="NAME", format="20A", array=np.array([b"OLDSTAR"])),
            fits.Column(name="RAEPP", format="D", array=np.array([10.0])),
            fits.Column(name="DECEPP", format="D", array=np.array([-30.0])),
        ]
    )
    sources_hdu = fits.BinTableHDU.from_columns(sources_cols, name="SOURCES")

    diam_cols = fits.ColDefs(
        [
            fits.Column(name="DIAMETER", format="D", array=np.array([0.004])),
            fits.Column(name="DIAMETER_ERR", format="D", array=np.array([0.0002])),
        ]
    )
    diameters_hdu = fits.BinTableHDU.from_columns(diam_cols, name="DIAMETERS")

    hdul = fits.HDUList(
        [fits.PrimaryHDU()]
        + [fits.ImageHDU() for _ in range(7)]  # indices 1..7
        + [sources_hdu]  # index 8
        + [_make_spectrum_hdu("OLDSTAR_MODEL")]  # index 9 = 0 + offset 9
        + [diameters_hdu]  # index 10 = [-2]
        + [fits.ImageHDU()]  # index 11
    )
    hdul.writeto(path, overwrite=True)

    result = calibrator_spectrum.lookup_local_database(path, ra_deg=10.0, dec_deg=-30.0)

    assert result is not None
    assert result.name == "OLDSTAR_MODEL"
    assert result.diameter_mas == pytest.approx(4.0)
    assert result.diameter_err_mas == pytest.approx(0.2)


def test_lookup_local_database_calib_spec(tmp_path):
    """calib_spec_db uses lowercase columns and a spectrum offset of 2."""
    path = tmp_path / "calib_spec_db_v10.fits"
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(),
            _make_calib_spec_sources(
                uddl=3.0, uddn=7.0, e_diam_est=0.25, diam_gaia=9.0
            ),
            _make_spectrum_hdu("CALSTAR_MODEL"),
        ]
    )
    hdul.writeto(path, overwrite=True)

    result = calibrator_spectrum.lookup_local_database(
        path, ra_deg=10.0, dec_deg=-30.0, band="L"
    )

    assert result is not None
    assert result.name == "CALSTAR_MODEL"
    assert result.database == "calib_spec_db_v10.fits"
    assert result.diameter_mas == pytest.approx(3.0)
    assert result.diameter_err_mas == pytest.approx(0.25)
    assert result.wavelength[-1] == pytest.approx(4.0e-6)


def test_lookup_calibrator_spectrum_returns_starsflux_result(monkeypatch, tmp_path):
    """A successful STARSFLUX lookup short-circuits the local database search."""
    expected = calibrator_spectrum.CalibratorSpectrum(
        name="HD1234",
        diameter_mas=1.0,
        diameter_err_mas=0.1,
        wavelength=np.array([3.0e-6]),
        flux=np.array([12.0]),
        database="starsflux",
        separation_arcsec=0.2,
        ra_deg=10.0,
        dec_deg=-30.0,
    )
    monkeypatch.setattr(
        calibrator_spectrum, "lookup_starsflux", lambda *_a, **_k: expected
    )

    def fail(*_a, **_k):
        raise AssertionError("local databases must not be queried")

    monkeypatch.setattr(calibrator_spectrum, "lookup_local_database", fail)

    result = calibrator_spectrum.lookup_calibrator_spectrum(
        cal_name="HD1234",
        ra_deg=10.0,
        dec_deg=-30.0,
        cal_database_paths=[tmp_path / "unused.fits"],
    )

    assert result is expected


def _patch_vizier(monkeypatch, query_object):
    """Inject a fake ``astroquery.vizier`` module exposing a stub Vizier."""

    class FakeVizier:
        pass

    FakeVizier.query_object = staticmethod(query_object)

    class FakeModule:
        Vizier = FakeVizier

    # Inject the stub into sys.modules rather than wrapping builtins.__import__:
    # the latter would route every import in the test through our wrapper.
    monkeypatch.setitem(sys.modules, "astroquery.vizier", FakeModule())


def test_lookup_starsflux_returns_none_when_not_in_mdfc(monkeypatch):
    """An empty Vizier result means the calibrator is absent from MDFC."""
    _patch_vizier(monkeypatch, lambda *_a, **_k: [])

    assert calibrator_spectrum.lookup_starsflux("NOSUCHSTAR", 10.0, -30.0) is None


def test_lookup_starsflux_reraises_non_remote_error(monkeypatch):
    """A non-network error is re-raised instead of being silently swallowed."""

    def boom(*_a, **_k):
        raise ValueError("bad column")

    _patch_vizier(monkeypatch, boom)

    with pytest.raises(ValueError, match="bad column"):
        calibrator_spectrum.lookup_starsflux("HD1234", 10.0, -30.0)


# ============================================================================
# Additional coverage: diagnostics.py and utils.py
# ============================================================================


def test_plot_calibrator_spectrum_without_oi_vis(tmp_path):
    """A calibrator HDUList lacking OI_VIS skips the correlated-flux overlay."""
    wav_obs = np.linspace(3e-6, 4e-6, 20)
    hdul = fits.HDUList([fits.PrimaryHDU()])
    try:
        diagnostics.plot_calibrator_spectrum(
            fig_dir=tmp_path,
            cal_name="NoVisCal",
            band="IR-LM",
            wav_model=wav_obs,
            flux_model=np.ones(20) * 5.0,
            wav_obs=wav_obs,
            spectrum_resampled=np.ones(20) * 5.0,
            hdul_cal=hdul,
            diameter_mas=2.5,
        )
    finally:
        hdul.close()

    assert [p.name for p in tmp_path.glob("*.png")] == [
        "calibrator_LM_NoVisCal_spectrum.png"
    ]


def _drop_extension(hdul: fits.HDUList, name: str) -> fits.HDUList:
    """Return a copy of *hdul* without the named extension."""
    return fits.HDUList([hdu for hdu in hdul if hdu.name != name])


def test_plot_calibrated_flux_without_oi_array(tmp_path):
    """A missing OI_ARRAY leaves the station lookup empty but still plots."""
    hdul = _drop_extension(_make_calibrated_hdul(), "OI_ARRAY")
    try:
        diagnostics.plot_calibrated_flux(
            fig_dir=tmp_path,
            hdul_out=hdul,
            cal_name="CAL",
            sci_name="SCI",
            mode="flux",
            band="IR-LM",
            bcd="OUT_OUT",
        )
    finally:
        hdul.close()

    assert [p.name for p in tmp_path.glob("*.png")] == [
        "calibrated_LM_SCI_cal_CAL_bcd_OUT_OUT.png"
    ]


def test_plot_calibrated_flux_without_oi_vis(tmp_path):
    """The correlated-flux panel degrades to a 'No OI_VIS' message."""
    hdul = _drop_extension(_make_calibrated_hdul(), "OI_VIS")
    try:
        diagnostics.plot_calibrated_flux(
            fig_dir=tmp_path,
            hdul_out=hdul,
            cal_name="CAL",
            sci_name="SCI",
            mode="corrflux",
            band="IR-N",
            bcd="IN_IN",
        )
    finally:
        hdul.close()

    assert [p.name for p in tmp_path.glob("*.png")] == [
        "calibrated_N_SCI_cal_CAL_bcd_IN_IN.png"
    ]


def test_plot_calibrated_flux_unknown_mode_writes_nothing(tmp_path):
    """An unknown mode yields zero panels and no figure on disk."""
    hdul = _make_calibrated_hdul()
    try:
        diagnostics.plot_calibrated_flux(
            fig_dir=tmp_path,
            hdul_out=hdul,
            cal_name="CAL",
            sci_name="SCI",
            mode="none",
            band="IR-LM",
            bcd="OUT_OUT",
        )
    finally:
        hdul.close()

    assert list(tmp_path.glob("*.png")) == []


def test_identify_detector_dispersion_unknown_mode():
    """An unrecognised DISPNAME raises ValueError."""
    hdul = _make_hdul(
        **{
            "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
            "HIERARCH ESO INS DIL NAME": "ULTRA",
        }
    )
    try:
        with pytest.raises(ValueError, match="Unknown dispersion mode: ULTRA"):
            utils._identify_detector_dispersion(hdul[0].header)
    finally:
        hdul.close()


def test_get_dl_coeffs_missing_combination():
    """A detector/mode pair absent from the coefficient table raises ValueError."""
    hdul = _make_hdul(
        **{
            "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
            "HIERARCH ESO INS DIL NAME": "HIGH",
        }
    )
    try:
        with pytest.raises(ValueError, match="No .* coefficients for HAWAII/HIGH"):
            utils.get_dl_coeffs(hdul)
    finally:
        hdul.close()


def test_resolve_latest_zenodo_record_without_certifi(monkeypatch):
    """The default SSL context is used when certifi is not installed."""
    import json
    import sys

    monkeypatch.setitem(sys.modules, "certifi", None)
    payload = json.dumps({"id": 77, "metadata": {"version": "3.0"}}).encode()
    monkeypatch.setattr(
        databases.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeZenodoResponse(payload),
    )

    assert databases._resolve_latest_zenodo_record() == ("77", "3.0")


def test_lookup_calibrator_spectrum_returns_first_local_match(
    monkeypatch, mock_vboekel_database, tmp_path
):
    """When STARSFLUX fails, the first matching local database wins."""
    monkeypatch.setattr(calibrator_spectrum, "lookup_starsflux", lambda *_a, **_k: None)

    result = calibrator_spectrum.lookup_calibrator_spectrum(
        cal_name="STAR1",
        ra_deg=10.0,
        dec_deg=-30.0,
        cal_database_paths=[tmp_path / "missing.fits", mock_vboekel_database],
    )

    assert result is not None
    assert result.name == "STAR2_MODEL"
    assert result.database == "vBoekelDatabase.fits"
