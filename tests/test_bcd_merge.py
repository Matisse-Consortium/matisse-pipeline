"""Tests for BCD merge helpers, correction file loading and BCDConfig validation."""

import shutil

import pytest
from astropy.io import fits

from matisse.core.bcd.config import BCD_MODES_TO_CORRECT, BCDConfig
from matisse.core.bcd.io_utils import load_bcd_corrections
from matisse.core.bcd.merge import (
    _ensure_hdulists,
    find_sci_filename,
    merge_by_tpl_start,
    merge_oifits,
)

_OUT_OUT = "fake_calib_IR-LM_LOW_OUT_OUT_noChop.fits"
_IN_IN = "fake_calib_IR-LM_LOW_IN_IN_noChop.fits"


@pytest.fixture
def bcd_copies(tmp_path, bcd_dir):
    """Copy the OUT_OUT and IN_IN test files into tmp_path and return their paths."""
    return [shutil.copy(bcd_dir / name, tmp_path / name) for name in (_OUT_OUT, _IN_IN)]


def test_merge_oifits_warns_when_flux_missing_in_one_file(bcd_copies, caplog):
    """A file without OI_FLUX is tolerated: a warning is logged and the merge completes."""
    first, second = (fits.open(path) for path in bcd_copies)
    del second["OI_FLUX"]

    merged = merge_oifits([first, second])

    assert "OI_FLUX extension missing in some files" in caplog.text
    assert merged["OI_VIS2"].data["VIS2DATA"].shape == (6, 118)
    assert merged["OI_T3"].data["T3PHI"].shape[0] == 4


def test_merge_by_tpl_start_rejects_non_directory_source(bcd_copies):
    """A regular file passed as source yields two empty lists."""
    assert merge_by_tpl_start(str(bcd_copies[0])) == ([], [])


def test_merge_by_tpl_start_accepts_a_list_of_paths(bcd_copies):
    """A list of file paths is merged into a single group (both share one TPL START)."""
    merged, raw = merge_by_tpl_start([str(path) for path in bcd_copies])

    assert len(raw) == 2
    assert len(merged) == 1
    assert merged[0]["OI_VIS2"].data["VIS2DATA"].shape == (6, 118)


def test_ensure_hdulists_with_empty_input():
    """An empty input list produces an empty output list."""
    assert _ensure_hdulists([]) == []


def test_find_sci_filename_chopping_suffix(tmp_path, bcd_dir):
    """chopping=True selects _Chop files and ignores the _noChop ones."""
    chop_file = tmp_path / "fake_calib_IR-LM_LOW_OUT_OUT_Chop.fits"
    shutil.copy(bcd_dir / _OUT_OUT, chop_file)
    shutil.copy(bcd_dir / _IN_IN, tmp_path / _IN_IN)

    found = find_sci_filename(tmp_path, chopping=True, band="LM", include_cal=True)

    assert found == [chop_file]


def test_find_sci_filename_skips_unreadable_and_calibrator_files(tmp_path, bcd_dir):
    """Only science files are returned: calibrators and unreadable files are dropped."""
    science = tmp_path / "fake_sci_IR-LM_LOW_OUT_OUT_Chop.fits"
    shutil.copy(bcd_dir / _OUT_OUT, science)
    with fits.open(science, mode="update") as hdul:
        hdul[0].header["ESO PRO CATG"] = "TARGET_RAW_INT"
    shutil.copy(bcd_dir / _OUT_OUT, tmp_path / "fake_calib_IR-LM_LOW_OUT_OUT_Chop.fits")
    (tmp_path / "broken_IR-LM_LOW_OUT_OUT_Chop.fits").write_text("not a FITS file\n")

    found = find_sci_filename(tmp_path, chopping=True, band="LM")

    assert found == [science]


@pytest.mark.parametrize("bcd_mode", BCD_MODES_TO_CORRECT)
def test_load_bcd_corrections_uses_bundled_calibration(bcd_mode):
    """With corrections_dir=None the CSV bundled with the package is read."""
    df = load_bcd_corrections(bcd_mode, None)

    assert not df.empty
    assert {
        "baseline_idx1",
        "baseline_idx2",
        "window",
        "wl_start_um",
        "wl_end_um",
        "coef_x0",
    } <= set(df.columns)


def test_load_bcd_corrections_missing_external_file(tmp_path):
    """An external directory without the expected CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="bcd_NOPE_poly_coeffs.csv"):
        load_bcd_corrections("NOPE", tmp_path)


@pytest.mark.parametrize(
    ("extension", "vis_column"),
    [("OI_VIS", "visamp"), ("OI_VIS2", "vis2data")],
)
def test_bcd_config_vis_column(tmp_path, extension, vis_column):
    """The visibility column name is derived from the extension."""
    config = BCDConfig(extension=extension, output_dir=tmp_path)

    assert config.vis_column == vis_column


def test_bcd_config_rejects_unknown_extension(tmp_path):
    """An unsupported extension is rejected at construction time."""
    with pytest.raises(ValueError, match="Unknown extension: NOPE"):
        BCDConfig(extension="NOPE", output_dir=tmp_path)
