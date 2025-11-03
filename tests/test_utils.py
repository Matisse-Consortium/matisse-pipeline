from __future__ import annotations

import importlib.util
import io
import json
import logging
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from matisse_pipeline.core.utils import common, io_utils, log_utils, oifits_reader


def _reset_console() -> io.StringIO:
    stream = log_utils.console.file
    stream.seek(0)
    stream.truncate(0)
    return stream


def test_is_json_list_detection():
    assert io_utils._is_json_list("[1, 2, 3]")
    assert not io_utils._is_json_list("{not a list}")


def test_read_list_file_skips_comments(tmp_path):
    listing = tmp_path / "files.lst"
    listing.write_text("""\n# comment\nfirst.fits\n\nsecond.fits\n""")

    assert io_utils._read_list_file(listing) == ["first.fits", "second.fits"]


def test_resolve_raw_input_directory_glob(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    (data_dir / "MATIS_test1.fits").touch()
    (data_dir / "MATIS_test2.fits").touch()
    (data_dir / "ignore.txt").touch()

    files, source = io_utils.resolve_raw_input(str(data_dir))

    assert source == "directory glob (MATIS*.fits)"
    assert {path.name for path in files} == {"MATIS_test1.fits", "MATIS_test2.fits"}


def test_resolve_raw_input_json_and_dedup(tmp_path):
    file_a = tmp_path / "a.fits"
    file_b = tmp_path / "b.fits"
    file_a.touch()
    file_b.touch()

    spec = json.dumps([str(file_a), str(file_a), str(file_b)])
    files, source = io_utils.resolve_raw_input(spec)

    assert source == "JSON list"
    assert files == [file_a, file_b]


def test_resolve_raw_input_errors(tmp_path):
    with pytest.raises(ValueError):
        io_utils.resolve_raw_input("[not valid json]")

    with pytest.raises(FileNotFoundError):
        io_utils.resolve_raw_input([str(tmp_path / "missing.fits")])


def test_resolve_raw_input_single_file(tmp_path):
    file_path = tmp_path / "single.fits"
    file_path.touch()
    files, source = io_utils.resolve_raw_input(str(file_path))
    assert files == [file_path]
    assert source == "single FITS file"


def test_resolve_raw_input_text_list(tmp_path):
    data_file = tmp_path / "data.fits"
    data_file.touch()
    listing = tmp_path / "paths.lst"
    listing.write_text(str(data_file))
    files, source = io_utils.resolve_raw_input(str(listing))
    assert files == [data_file]
    assert source == f"text file list ({listing.name})"


def test_resolve_raw_input_custom_file_list(tmp_path):
    data_file = tmp_path / "data2.fits"
    data_file.touch()
    listing = tmp_path / "paths.cfg"
    listing.write_text(str(data_file))
    files, source = io_utils.resolve_raw_input(str(listing))
    assert files == [data_file]
    assert source == f"custom file list ({listing.name})"


def test_resolve_raw_input_glob_pattern(tmp_path):
    data_file = tmp_path / "pattern.fits"
    data_file.touch()
    files, source = io_utils.resolve_raw_input(str(tmp_path / "*.fits"))
    assert files == [data_file]
    assert source == "glob pattern"


def test_check_for_calib_file_uses_matisse_type(monkeypatch, capsys):
    calls: list[str] = []

    def fake_matisse_type(header):
        calls.append(header["ID"])
        return header["ID"]

    monkeypatch.setattr(io_utils, "matisse_type", fake_matisse_type)

    io_utils.check_for_calib_file([{"ID": "A"}, {"ID": "B"}])

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured == ["A", "B"]
    assert calls == ["A", "B"]


def test_remove_double_parameter_eliminates_duplicates():
    result = common.remove_double_parameter("--gain=1 --temp=2 --gain=3 --mode=fast")
    assert "--gain=1" in result
    assert "--gain=3" not in result
    assert "--temp=2" in result and "--mode=fast" in result


def test_open_oifits_and_v2_equivalence(real_oifits):
    legacy = oifits_reader.open_oifits(real_oifits)
    reader = oifits_reader.OIFitsReader(real_oifits)
    modern_data = reader.read()
    assert modern_data is not None
    modern = modern_data.to_dict()

    assert legacy and modern
    assert set(legacy) == set(modern)

    for key in legacy:
        lhs = legacy[key]
        rhs = modern[key]
        if isinstance(lhs, np.ndarray):
            assert np.array_equal(lhs, rhs)
        elif isinstance(lhs, dict):
            assert set(lhs) == set(rhs)
            for sub_key in lhs:
                sub_lhs = lhs[sub_key]
                sub_rhs = rhs[sub_key]
                if isinstance(sub_lhs, np.ndarray):
                    assert np.array_equal(sub_lhs, sub_rhs)
                else:
                    assert sub_lhs == sub_rhs
        else:
            assert lhs == rhs


def test_open_oifits_missing_file_returns_empty_dict(tmp_path):
    missing = tmp_path / "none.fits"
    assert oifits_reader.open_oifits(missing) == {}


def test_oifits_reader_handles_tf2_and_mjd_fix(real_oifits, tmp_path):
    with fits.open(real_oifits) as hdul:
        hdus = fits.HDUList([hdu.copy() for hdu in hdul])

    hdus[0].header["MJD-OBS"] = 59000.0
    vis = hdus["OI_VIS"].data
    vis["MJD"] = np.full_like(vis["MJD"], 40000.0)

    rows = len(vis)
    sta_index = vis["STA_INDEX"].astype(np.int16)
    tf2_cols = fits.ColDefs(
        [
            fits.Column(
                name="TF2", format="E", array=np.full(rows, 0.8, dtype=np.float32)
            ),
            fits.Column(
                name="TF2ERR", format="E", array=np.full(rows, 0.1, dtype=np.float32)
            ),
            fits.Column(
                name="MJD", format="E", array=np.full(rows, 40000.0, dtype=np.float32)
            ),
            fits.Column(
                name="STA_INDEX", format=f"{sta_index.shape[1]}I", array=sta_index
            ),
        ]
    )
    tf2_hdu = fits.BinTableHDU.from_columns(tf2_cols, name="TF2")
    if "TF2" in hdus:
        del hdus["TF2"]
    hdus.append(tf2_hdu)
    target = tmp_path / "with_tf2.fits"
    hdus.writeto(target, overwrite=True)
    hdus.close()

    data = oifits_reader.open_oifits(target)
    assert "TF2" in data
    assert np.allclose(data["TF2"]["TF2"], 0.8)
    assert np.allclose(data["TF2"]["TF2ERR"], 0.1)
    assert np.all(data["VIS"]["TIME"] == 59000.0)

    reader = oifits_reader.OIFitsReader(target)
    reader._hdu = fits.open(target)
    try:
        missing = reader._read_array_data("UNKNOWN_COLUMN")
        assert missing.size == 0
    finally:
        reader._hdu.close()


def test_oifits_reader_fix_mjd_without_header(tmp_path):
    reader = oifits_reader.OIFitsReader(tmp_path / "dummy.fits")
    reader._hdu = fits.HDUList([fits.PrimaryHDU()])
    mjd = np.array([40000.0])
    fixed = reader._fix_mjd_if_needed(mjd)
    assert fixed is mjd
    reader._hdu.close()


def test_oifits_reader_parse_all_data(tmp_path):
    primary = fits.PrimaryHDU()
    hdr = primary.header
    hdr["HIERARCH ESO ISS AMBI FWHM START"] = 0.5
    hdr["HIERARCH ESO ISS AMBI FWHM END"] = 0.7
    hdr["HIERARCH ESO ISS AMBI TAU0 START"] = 3.0
    hdr["HIERARCH ESO ISS AMBI TAU0 END"] = 5.0
    hdr["ESO PRO CATG"] = "TARGET_RAW_INT"
    hdr["DATE-OBS"] = "2025-05-20"
    hdr["HIERARCH ESO DET CHIP NAME"] = "AQUARIUS"
    hdr["HIERARCH ESO INS DIN NAME"] = "LOW"
    hdr["HIERARCH ESO INS DIL NAME"] = "LOW"
    hdr["HIERARCH ESO DET SEQ1 DIT"] = 0.5
    hdr["HIERARCH ESO INS BCD1 NAME"] = "BCD1"
    hdr["HIERARCH ESO INS BCD2 NAME"] = "BCD2"
    hdr["MJD-OBS"] = 60000.0

    target_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(name="TARGET", format="16A", array=np.array([b"STAR"])),
            ]
        ),
        name="OI_TARGET",
    )

    array_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="TEL_NAME", format="16A", array=np.array([b"A1", b"B1"])
                ),
                fits.Column(
                    name="STA_NAME", format="16A", array=np.array([b"STA1", b"STA2"])
                ),
                fits.Column(
                    name="STA_INDEX",
                    format="1I",
                    array=np.array([1, 2], dtype=np.int16),
                ),
            ]
        ),
        name="OI_ARRAY",
    )

    wavelength_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="EFF_WAVE",
                    format="E",
                    array=np.array([3.4, 3.6], dtype=np.float32),
                ),
            ]
        ),
        name="OI_WAVELENGTH",
    )

    sta_index_vis = np.tile(np.array([1, 2], dtype=np.int16), (2, 1))
    vis_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="VISAMP", format="E", array=np.full(2, 1.0, dtype=np.float32)
                ),
                fits.Column(
                    name="VISAMPERR",
                    format="E",
                    array=np.full(2, 0.1, dtype=np.float32),
                ),
                fits.Column(
                    name="VISPHI", format="E", array=np.zeros(2, dtype=np.float32)
                ),
                fits.Column(
                    name="VISPHIERR",
                    format="E",
                    array=np.full(2, 0.01, dtype=np.float32),
                ),
                fits.Column(name="FLAG", format="L", array=np.zeros(2, dtype=np.bool_)),
                fits.Column(
                    name="UCOORD",
                    format="E",
                    array=np.array([0.0, 10.0], dtype=np.float32),
                ),
                fits.Column(
                    name="VCOORD",
                    format="E",
                    array=np.array([0.0, -10.0], dtype=np.float32),
                ),
                fits.Column(name="MJD", format="D", array=np.full(2, 40000.0)),
                fits.Column(name="STA_INDEX", format="2I", array=sta_index_vis),
                fits.Column(
                    name="CFXAMP", format="E", array=np.full(2, 5.0, dtype=np.float32)
                ),
                fits.Column(
                    name="CFXAMPERR",
                    format="E",
                    array=np.full(2, 0.5, dtype=np.float32),
                ),
            ]
        ),
        name="OI_VIS",
    )

    vis2_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="VIS2DATA", format="E", array=np.full(2, 0.8, dtype=np.float32)
                ),
                fits.Column(
                    name="VIS2ERR", format="E", array=np.full(2, 0.05, dtype=np.float32)
                ),
                fits.Column(
                    name="UCOORD",
                    format="E",
                    array=np.array([1.0, 2.0], dtype=np.float32),
                ),
                fits.Column(
                    name="VCOORD",
                    format="E",
                    array=np.array([3.0, 4.0], dtype=np.float32),
                ),
                fits.Column(name="MJD", format="D", array=np.full(2, 40000.0)),
                fits.Column(name="FLAG", format="L", array=np.zeros(2, dtype=np.bool_)),
                fits.Column(name="STA_INDEX", format="2I", array=sta_index_vis),
            ]
        ),
        name="OI_VIS2",
    )

    sta_index_t3 = np.tile(np.array([1, 2, 3], dtype=np.int16), (2, 1))
    t3_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="T3AMP", format="E", array=np.full(2, 1.2, dtype=np.float32)
                ),
                fits.Column(
                    name="T3AMPERR", format="E", array=np.full(2, 0.1, dtype=np.float32)
                ),
                fits.Column(
                    name="T3PHI", format="E", array=np.zeros(2, dtype=np.float32)
                ),
                fits.Column(
                    name="T3PHIERR",
                    format="E",
                    array=np.full(2, 0.02, dtype=np.float32),
                ),
                fits.Column(
                    name="U1COORD",
                    format="E",
                    array=np.array([1.0, 2.0], dtype=np.float32),
                ),
                fits.Column(
                    name="V1COORD",
                    format="E",
                    array=np.array([1.5, 2.5], dtype=np.float32),
                ),
                fits.Column(
                    name="U2COORD",
                    format="E",
                    array=np.array([3.0, 4.0], dtype=np.float32),
                ),
                fits.Column(
                    name="V2COORD",
                    format="E",
                    array=np.array([3.5, 4.5], dtype=np.float32),
                ),
                fits.Column(name="MJD", format="D", array=np.full(2, 40000.0)),
                fits.Column(name="FLAG", format="L", array=np.zeros(2, dtype=np.bool_)),
                fits.Column(name="STA_INDEX", format="3I", array=sta_index_t3),
            ]
        ),
        name="OI_T3",
    )

    flux_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="FLUXDATA", format="E", array=np.full(2, 1.5, dtype=np.float32)
                ),
                fits.Column(
                    name="FLUXERR", format="E", array=np.full(2, 0.2, dtype=np.float32)
                ),
                fits.Column(name="MJD", format="D", array=np.full(2, 40000.0)),
                fits.Column(name="FLAG", format="L", array=np.zeros(2, dtype=np.bool_)),
                fits.Column(
                    name="STA_INDEX",
                    format="1I",
                    array=np.array([1, 2], dtype=np.int16),
                ),
            ]
        ),
        name="OI_FLUX",
    )

    tf2_hdu = fits.BinTableHDU.from_columns(
        fits.ColDefs(
            [
                fits.Column(
                    name="TF2", format="E", array=np.full(2, 0.9, dtype=np.float32)
                ),
                fits.Column(
                    name="TF2ERR", format="E", array=np.full(2, 0.05, dtype=np.float32)
                ),
                fits.Column(name="MJD", format="D", array=np.full(2, 40000.0)),
                fits.Column(name="STA_INDEX", format="2I", array=sta_index_vis),
            ]
        ),
        name="TF2",
    )

    hdus = fits.HDUList(
        [
            primary,
            target_hdu,
            array_hdu,
            wavelength_hdu,
            vis_hdu,
            vis2_hdu,
            t3_hdu,
            flux_hdu,
            tf2_hdu,
        ]
    )

    reader = oifits_reader.OIFitsReader(tmp_path / "synthetic.fits")
    reader._hdu = hdus
    data = reader._parse_all_data()
    hdus.close()

    assert data.seeing == pytest.approx(0.6)
    assert data.tau0 == pytest.approx(4.0)
    assert data.category == "SCI"
    assert data.band == "N"
    assert data.dispersion_name == "LOW"
    assert data.dit == pytest.approx(0.5)
    assert data.bcd1_name == "BCD1"
    assert data.bcd2_name == "BCD2"
    assert data.vis is not None and "CFLUX" in data.vis
    assert data.vis2 is not None and data.vis2["VIS2"].shape == (2,)
    assert data.t3 is not None and data.t3["STA_INDEX"].shape == (2, 3)
    assert data.flux is not None
    assert data.tf2 is not None
    assert np.all(data.vis["TIME"] == 60000.0)


def test_oifits_reader_additional_branches(tmp_path):
    reader = oifits_reader.OIFitsReader(tmp_path / "dummy.fits")
    with pytest.raises(RuntimeError):
        reader._ensure_hdu()

    hdr = fits.Header()
    hdr["FILTER1"] = "H_band"
    reader._hdu = fits.HDUList([fits.PrimaryHDU(header=hdr)])
    try:
        assert reader._read_band(reader._hdu[0].header) == "H"
        assert reader._read_dispersion_name(reader._hdu[0].header) == ""
        assert np.isnan(reader._read_dit(reader._hdu[0].header))
        assert reader._read_bcd1_name(reader._hdu[0].header) == ""
        assert reader._read_bcd2_name(reader._hdu[0].header) == ""
    finally:
        reader._hdu.close()

    reader._hdu = fits.HDUList([fits.PrimaryHDU()])
    try:
        header = reader._hdu[0].header
        assert reader._read_wavelength().size == 0
        assert reader._read_target_name() == ""
        assert reader._read_category(header) == "CAL"
        assert reader._read_date_obs(header) == ""
        assert reader._read_detector_name(header) == ""
        assert reader._read_band(header) == ""
    finally:
        reader._hdu.close()


def test_oifits_reader_read_handles_parse_error(tmp_path, monkeypatch):
    sample = tmp_path / "empty.fits"
    fits.PrimaryHDU().writeto(sample)

    reader = oifits_reader.OIFitsReader(sample)
    monkeypatch.setattr(
        reader, "_parse_all_data", lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert reader.read() is None


def test_log_utils_reconfigures_logging_when_no_handlers(monkeypatch):
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    for handler in original_handlers:
        root.removeHandler(handler)

    spec = importlib.util.spec_from_file_location(
        "log_utils_shadow", Path(log_utils.__file__)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for handler in original_handlers:
        root.addHandler(handler)


def test_oifits_reader_missing_tables_return_none():
    reader = oifits_reader.OIFitsReader(Path("/tmp/unused.fits"))
    reader._hdu = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(name="OI_ARRAY")])
    try:
        assert reader._read_array_data("TEL_NAME").size == 0
    finally:
        reader._hdu.close()

    for name in ["OI_VIS", "OI_VIS2", "OI_T3", "OI_FLUX", "TF2"]:
        reader._hdu = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(name=name)])
        try:
            method = {
                "OI_VIS": reader._read_vis_table,
                "OI_VIS2": reader._read_vis2_table,
                "OI_T3": reader._read_t3_table,
                "OI_FLUX": reader._read_flux_table,
                "TF2": reader._read_tf2_table,
            }[name]
            assert method() is None
            del reader._hdu[name]
            assert method() is None
        finally:
            reader._hdu.close()


def test_set_verbosity_adjusts_levels():
    logger = logging.getLogger("utils-test")
    log_utils.set_verbosity(logger, True)
    assert logger.level == logging.DEBUG
    log_utils.set_verbosity(logger, False)
    assert logger.level == logging.INFO


def test_section_and_iteration_banner_output():
    stream = _reset_console()
    log_utils.section("Example")
    log_utils.iteration_banner(3)
    output = stream.getvalue()
    assert "Example" in output
    assert "ITERATION 3" in output


def test_get_detector_and_target_name_helpers():
    block = {
        "input": [
            [
                "file.fits",
                "TAG",
                {"HIERARCH ESO DET CHIP NAME": "AQUARIUS", "ESO OBS TARG NAME": "HD"},
            ]
        ],
    }
    assert log_utils.get_detector_name(block) == "AQUARIUS"
    assert log_utils.get_target_name(block) == "HD"

    empty_block = {"input": []}
    assert log_utils.get_detector_name(empty_block) == "N/A"
    assert log_utils.get_target_name(empty_block) == "N/A"


def test_show_calibration_status_outputs_table():
    stream = _reset_console()
    blocks = [
        {
            "input": [
                [
                    "frame.fits",
                    "BADPIX",
                    {
                        "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
                        "ESO OBS TARG NAME": "CAL",
                    },
                ]
            ],
            "calib": [
                ("badpix.fits", "BADPIX"),
                ("flat.fits", "OBS_FLATFIELD"),
            ],
            "action": "ACTION_MAT_EST_KAPPA",
        }
    ]

    log_utils.show_calibration_status(blocks, log_utils.console)
    output = stream.getvalue()
    assert "Calibration Summary" in output
    assert "✅" in output


def test_show_blocs_status_breaks_loop():
    stream = _reset_console()
    blocks = [
        {
            "tplstart": "2025-01-01T00:00:00.HAWAII-2RG",
            "input": [
                [
                    "file1.fits",
                    "TAG1",
                    {
                        "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
                        "ESO OBS TARG NAME": "T1",
                    },
                ]
            ],
            "action": "ACTION1",
            "status": 1,
            "iter": 1,
        },
        {
            "tplstart": "tpl",
            "input": [
                [
                    "file2.fits",
                    "TAG2",
                    {
                        "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
                        "ESO OBS TARG NAME": "T2",
                    },
                ]
            ],
            "action": "ACTION2",
            "status": -2,
            "iter": 1,
        },
        {
            "tplstart": "tpl",
            "input": [
                [
                    "file3.fits",
                    "TAG3",
                    {
                        "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
                        "ESO OBS TARG NAME": "T3",
                    },
                ]
            ],
            "action": "NO-ACTION",
            "status": 0,
            "iter": 1,
        },
        {
            "tplstart": "tpl",
            "input": [
                [
                    "file4.fits",
                    "TAG4",
                    {
                        "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
                        "ESO OBS TARG NAME": "T4",
                    },
                ]
            ],
            "action": "ACTION3",
            "status": 0,
            "iter": 1,
        },
    ]

    should_break = log_utils.show_blocs_status([], 1, 1, blocks, check_blocks=False)
    assert should_break is True
    output = stream.getvalue()
    assert "Global Pipeline Statistics" in output
    assert "⚠" in output


def test_show_blocs_status_continue_branch():
    blocks = [
        {
            "tplstart": "tpl",
            "input": [
                [
                    "file.fits",
                    "TAG",
                    {
                        "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
                        "ESO OBS TARG NAME": "T",
                    },
                ]
            ],
            "action": "ACTION",
            "status": 0,
            "iter": 1,
        }
    ]

    should_break = log_utils.show_blocs_status(["cmd"], 1, 2, blocks, check_blocks=True)
    assert should_break is False
