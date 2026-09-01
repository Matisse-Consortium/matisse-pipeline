"""Tests for matisse.core.flux.airmass."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from astropy.io import fits
from numpy.testing import assert_allclose

from matisse.core.flux import airmass

# ---------------------------------------------------------------------------
# run_skycalc
# ---------------------------------------------------------------------------


def test_run_skycalc_returns_false_when_cli_is_absent(monkeypatch, tmp_path):
    """Without skycalc_cli on PATH, run_skycalc reports failure."""
    monkeypatch.setattr(airmass.shutil, "which", lambda _name: None)

    assert airmass.run_skycalc(tmp_path / "in.txt", tmp_path / "out.fits") is False


def test_run_skycalc_invokes_the_cli_and_reports_success(monkeypatch, tmp_path):
    """A successful subprocess call yields True and passes the two file paths."""
    monkeypatch.setattr(airmass.shutil, "which", lambda _name: "/fake/skycalc_cli")
    recorded = []
    monkeypatch.setattr(
        airmass.subprocess, "run", lambda cmd, **_kw: recorded.append(cmd)
    )

    ok = airmass.run_skycalc(tmp_path / "in.txt", tmp_path / "out.fits")

    assert ok is True
    assert recorded == [
        [
            "/fake/skycalc_cli",
            "-i",
            str(tmp_path / "in.txt"),
            "-o",
            str(tmp_path / "out.fits"),
        ]
    ]


def test_run_skycalc_returns_false_when_the_cli_fails(monkeypatch, tmp_path):
    """A non-zero exit status from skycalc_cli is caught and reported as failure."""
    monkeypatch.setattr(airmass.shutil, "which", lambda _name: "/fake/skycalc_cli")

    def boom(_cmd, **_kwargs):
        raise subprocess.CalledProcessError(1, "skycalc_cli", stderr="boom")

    monkeypatch.setattr(airmass.subprocess, "run", boom)

    assert airmass.run_skycalc(tmp_path / "in.txt", tmp_path / "out.fits") is False


# ---------------------------------------------------------------------------
# read_skycalc_output
# ---------------------------------------------------------------------------


def test_read_skycalc_output_converts_nm_to_micrometres(tmp_path):
    """Wavelengths are converted from nm to um and transmission round-trips."""
    lam_nm = np.array([3000.0, 3500.0, 4000.0])
    trans = np.array([0.9, 0.5, 0.95])
    path = tmp_path / "skycalc.fits"
    hdu = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="lam", format="D", array=lam_nm),
            fits.Column(name="trans", format="D", array=trans),
        ]
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    wl_um, trans_out = airmass.read_skycalc_output(path)

    assert_allclose(wl_um, lam_nm * 1e-3, rtol=1e-12)
    assert_allclose(trans_out, trans, rtol=1e-12)


# ---------------------------------------------------------------------------
# _resample_to_matisse_resolution
# ---------------------------------------------------------------------------


def test_resample_to_matisse_resolution_preserves_a_constant():
    """A constant transmission stays that constant after Gaussian resampling."""
    wav_model = np.linspace(2.5, 4.5, 2001)
    flux_model = np.full_like(wav_model, 0.8)
    wav_obs = np.linspace(3.0, 4.0, 51)

    resampled = airmass._resample_to_matisse_resolution(
        wav_model, flux_model, wav_obs, size_spec_channel=4.92, nsigma=5
    )

    assert_allclose(resampled, np.full_like(wav_obs, 0.8), rtol=1e-10)


def test_resample_to_matisse_resolution_is_nan_when_window_is_empty():
    """Observed channels with fewer than 5 model points in their window are NaN."""
    wav_model = np.linspace(3.5, 3.6, 1001)
    flux_model = np.full_like(wav_model, 0.8)
    wav_obs = np.linspace(3.0, 4.0, 101)

    resampled = airmass._resample_to_matisse_resolution(
        wav_model, flux_model, wav_obs, size_spec_channel=4.92, nsigma=5
    )

    # Window half-width is 5 * 4.92 * 0.01 / 2.355 ~= 0.104 um around each channel.
    assert np.all(np.isnan(resampled[wav_obs < 3.3]))
    assert np.all(np.isnan(resampled[wav_obs > 3.8]))
    assert np.all(np.isfinite(resampled[np.abs(wav_obs - 3.55) < 0.05]))


# ---------------------------------------------------------------------------
# calc_corr_offset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shift", [-4, 0, 3, 7])
def test_calc_corr_offset_peaks_at_the_applied_shift(shift):
    """The correlation peaks at the pixel shift that was applied to spectrum2."""
    rng = np.random.default_rng(0)
    spectrum1 = rng.normal(size=200)
    spectrum2 = np.roll(spectrum1, -shift)
    shift_max = 10

    rp = airmass.calc_corr_offset(spectrum1, spectrum2, shift_max)

    assert len(rp) == 2 * shift_max
    assert int(np.argmax(rp)) - shift_max == shift
    assert max(rp) == pytest.approx(1.0, abs=1e-10)


def test_calc_corr_offset_ignores_non_finite_channels():
    """NaNs in spectrum2 are masked out of both spectra before correlating."""
    rng = np.random.default_rng(1)
    spectrum1 = rng.normal(size=200)
    shift, shift_max = 5, 10
    spectrum2 = np.roll(spectrum1, -shift)
    spectrum2[:3] = np.nan
    spectrum2[-3:] = np.nan

    rp = airmass.calc_corr_offset(spectrum1, spectrum2, shift_max)

    assert np.all(np.isfinite(rp))
    assert int(np.argmax(rp)) - shift_max == shift
    assert max(rp) == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# compute_airmass_correction
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_skycalc(monkeypatch):
    """Make SkyCalc succeed and return a flat transmission: 0.5 for SCI, 0.9 for CAL."""
    monkeypatch.setattr(airmass, "run_skycalc", lambda *_a, **_k: True)
    wl_um = np.arange(2.0, 14.0, 1e-3)
    levels = iter([0.5, 0.9])
    monkeypatch.setattr(
        airmass,
        "read_skycalc_output",
        lambda *_a, **_k: (wl_um, np.full_like(wl_um, next(levels))),
    )


@pytest.mark.parametrize(
    ("sci_file", "cal_file", "tag_sci"),
    [
        (
            "2025_fakesci_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop.fits",
            "2025_fakecal_U1U2U3U4_IR-N_LOW_OUT_OUT_noChop.fits",
            "2025_fakesci_U1U2U3U4_IR-N_LOW",
        ),
        (
            "2025_fakesci_U1U2U3U4_IR-LM_LOW_OUT_OUT_noChop.fits",
            "2025_fakecal_U1U2U3U4_IR-LM_LOW_OUT_OUT_noChop.fits",
            "2025_fakesci_U1U2U3U4_IR-LM_LOW",
        ),
    ],
)
def test_compute_airmass_correction_is_the_flat_transmission_ratio(
    flat_skycalc, flux_dir, tmp_path, sci_file, cal_file, tag_sci
):
    """Flat SCI/CAL transmissions give the constant ratio trans_cal / trans_sci."""
    with (
        fits.open(flux_dir / sci_file) as hdul_sci,
        fits.open(flux_dir / cal_file) as hdul_cal,
    ):
        wav_sci_m = np.asarray(hdul_sci["OI_WAVELENGTH"].data["EFF_WAVE"], dtype=float)
        wav_cal_m = np.asarray(hdul_cal["OI_WAVELENGTH"].data["EFF_WAVE"], dtype=float)
        correction = airmass.compute_airmass_correction(
            hdul_sci=hdul_sci,
            hdul_cal=hdul_cal,
            wav_sci_m=wav_sci_m,
            wav_cal_m=wav_cal_m,
            airmass_sci=1.1,
            airmass_cal=1.4,
            pwv_sci=2.0,
            pwv_cal=3.0,
            output_dir=tmp_path,
            tag_sci=tag_sci,
            tag_cal="cal_tag",
        )

    assert correction.shape == wav_sci_m.shape
    # Every channel is genuinely resolved: none fell back to the 1.0 NaN default.
    assert_allclose(correction, np.full_like(wav_sci_m, 0.9 / 0.5), rtol=1e-10)
    # The SkyCalc input files were actually written for both SCI and CAL.
    assert len(list((tmp_path / "skycalc").glob("skycalc_input_*.txt"))) == 2


def test_compute_airmass_correction_unknown_band_without_skycalc(monkeypatch, tmp_path):
    """A tag naming no known band falls back to a unit correction when SkyCalc fails."""
    monkeypatch.setattr(airmass, "run_skycalc", lambda *_a, **_k: False)
    wav_sci_m = np.linspace(3.0e-6, 4.0e-6, 12)

    correction = airmass.compute_airmass_correction(
        hdul_sci=None,
        hdul_cal=None,
        wav_sci_m=wav_sci_m,
        wav_cal_m=wav_sci_m,
        airmass_sci=1.1,
        airmass_cal=1.2,
        pwv_sci=2.0,
        pwv_cal=2.0,
        output_dir=tmp_path,
        tag_sci="mystery_instrument",
        tag_cal="cal_tag",
    )

    assert_allclose(correction, np.ones_like(wav_sci_m), rtol=0, atol=0)


def test_compute_airmass_correction_returns_ones_when_cal_run_fails(
    monkeypatch, tmp_path
):
    """If only the CAL SkyCalc run fails, the correction falls back to all ones."""
    outcomes = [True, False]
    monkeypatch.setattr(airmass, "run_skycalc", lambda *_a, **_k: outcomes.pop(0))
    wav_sci_m = np.linspace(3.0e-6, 4.0e-6, 12)

    correction = airmass.compute_airmass_correction(
        hdul_sci=None,
        hdul_cal=None,
        wav_sci_m=wav_sci_m,
        wav_cal_m=wav_sci_m,
        airmass_sci=1.1,
        airmass_cal=1.2,
        pwv_sci=2.0,
        pwv_cal=2.0,
        output_dir=tmp_path,
        tag_sci="2025_fakesci_IR-LM_LOW",
        tag_cal="cal_tag",
    )

    assert outcomes == []
    assert_allclose(correction, np.ones_like(wav_sci_m), rtol=0, atol=0)
