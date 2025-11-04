from pathlib import Path

from astropy.io import fits

from matisse_pipeline.core.tidyup import change_oifits_filename, tidyup_path


def _write_oifits(path: Path, rm_list: list = None, **overrides) -> None:
    """Helper to create a minimal FITS file with MATISSE headers."""
    hdu = fits.PrimaryHDU()
    header = hdu.header
    header["HIERARCH ESO PRO CATG"] = overrides.get("catg", "CALIB_RAW_INT")
    header["HIERARCH ESO OBS TARG NAME"] = overrides.get("target", "Test Target")
    header["HIERARCH ESO ISS CONF STATION1"] = overrides.get("station1", "A1")
    header["HIERARCH ESO ISS CONF STATION2"] = overrides.get("station2", "B2")
    header["HIERARCH ESO ISS CONF STATION3"] = overrides.get("station3", "C3")
    header["HIERARCH ESO ISS CONF STATION4"] = overrides.get("station4", "D4")
    header["HIERARCH ESO DET CHIP TYPE"] = overrides.get("chip", "IR-LM")
    header["HIERARCH ESO INS DIL NAME"] = overrides.get("dil", "HIGH")
    header["HIERARCH ESO INS DIN NAME"] = overrides.get("din", "MED")
    header["HIERARCH ESO ISS CHOP ST"] = overrides.get("chop", "T")
    header["HIERARCH ESO TPL START"] = overrides.get("tpl_start", "2024-01-01T10:20:30")
    header["HIERARCH ESO INS BCD1 NAME"] = overrides.get("bcd1", "IN")
    header["HIERARCH ESO INS BCD2 NAME"] = overrides.get("bcd2", "IN")
    if rm_list:
        for rm_key in rm_list:
            header.remove(rm_key)
    hdu.writeto(path, overwrite=True)


def test_tidyup_path_renames_single_oifits(tmp_path):
    """Ensure a single file is renamed in place using FITS metadata."""
    original = tmp_path / "original.fits"
    _write_oifits(original, target="Alpha Centauri", chop="T")

    tidyup_path(original)

    expected_name = (
        "2024-01-01T102030_AlphaCentauri_A1B2C3D4_IR-LM_HIGH_IN_IN_Chop.fits"
    )
    renamed = tmp_path / expected_name

    assert renamed.exists(), f"Expected renamed file {renamed.name} not found."
    assert not original.exists(), "Original FITS file should have been renamed."


def test_tidyup_missing_conf(tmp_path):
    """Ensure a single file is renamed in place using FITS metadata."""
    original = tmp_path / "original.fits"
    _write_oifits(
        original,
        target="4 Sgr",
        chop="T",
        rm_list=["HIERARCH ESO ISS CONF STATION4"],
    )

    tidyup_path(original)

    expected_stations_config = "noConf"
    expected_name = (
        f"2024-01-01T102030_4Sgr_{expected_stations_config}_IR-LM_HIGH_IN_IN_Chop.fits"
    )
    renamed = tmp_path / expected_name

    assert renamed.exists(), f"Expected renamed file {renamed.name} not found."
    assert not original.exists(), "Original FITS file should have been renamed."


def test_tidyup_no_resol(tmp_path):
    """Ensure a single file is renamed in place using FITS metadata."""
    original = tmp_path / "original.fits"
    _write_oifits(
        original,
        target="4 Sgr",
        chop="T",
        rm_list=["HIERARCH ESO DET CHIP TYPE"],
    )

    tidyup_path(original)

    expected_resol = "noRes"
    expected_name = (
        f"2024-01-01T102030_4Sgr_A1B2C3D4_unknown_{expected_resol}_IN_IN_Chop.fits"
    )
    renamed = tmp_path / expected_name

    assert renamed.exists(), f"Expected renamed file {renamed.name} not found."
    assert not original.exists(), "Original FITS file should have been renamed."


def test_tidyup_missing_file(caplog, tmp_path):
    """Ensure a missing file is well handled."""
    original = tmp_path / "missing.fits"

    with caplog.at_level("WARNING"):
        result = change_oifits_filename(original)
    assert result is None
    assert any("Could not process missing.fits" in m for m in caplog.messages)


def test_tidyup_path_creates_backup_directory(monkeypatch, tmp_path):
    """Ensure files are copied into <cwd>/<basename>_OIFITS and renamed there."""
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "Iter1"
    source_dir.mkdir()

    valid_file = source_dir / "valid.fits"
    _write_oifits(valid_file, target="Beta Pic", chip="IR-N", din="LOW")

    skipped_file = source_dir / "TARGET_CAL_0123.fits"
    _write_oifits(skipped_file)

    nofits_file = source_dir / "TARGET_CAL_0123.json"
    _write_oifits(nofits_file)

    tidyup_path(source_dir)

    backup_dir = tmp_path / "Iter1_OIFITS"
    assert backup_dir.exists(), "Backup directory should be created in CWD."

    renamed_files = list(backup_dir.glob("*.fits"))
    assert len(renamed_files) == 1, "Only the valid file should be processed."

    expected_name = "2024-01-01T102030_BetaPic_A1B2C3D4_IR-N_LOW_IN_IN_Chop.fits"
    assert renamed_files[0].name == expected_name
    assert valid_file.exists(), "Original file must remain in place after copy."
    assert skipped_file.exists(), "Skipped files should remain untouched."


def test_tidyup_path_exists(monkeypatch, tmp_path, caplog):
    """Ensure files are copied into <cwd>/<basename>_OIFITS and renamed there."""
    monkeypatch.chdir(tmp_path)
    source_dir = tmp_path / "Iter1"
    source_dir.mkdir()

    backup_dir = tmp_path / "Iter1_OIFITS"

    with caplog.at_level("INFO"):
        result1 = tidyup_path(source_dir)

    with caplog.at_level("INFO"):
        result2 = tidyup_path(source_dir)

    assert result1 is None
    assert result2 is None
    assert any("No OIFITS/FITS files found" in m for m in caplog.messages)
    assert any(f"{backup_dir} already exists." in m for m in caplog.messages)


def test_tidyup_no_path(caplog):
    """Ensure files are copied into <cwd>/<basename>_OIFITS and renamed there."""
    missing_dir = Path("some/path/no/exist/")
    with caplog.at_level("WARNING"):
        result = tidyup_path(missing_dir)
    assert result is None
    assert any("Path not found:" in m for m in caplog.messages)
