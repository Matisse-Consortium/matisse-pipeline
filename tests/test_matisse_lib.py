from __future__ import annotations

import pytest
from astropy.io import fits

import matisse_pipeline.core.lib_auto_pipeline as lib_auto_pipeline
from matisse_pipeline.core.lib_auto_pipeline import (
    CalibEntry,
    matisse_action,
    matisse_calib,
    matisse_recipes,
    matisse_type,
)


def _header(path):
    return fits.getheader(path)


def _make_base_header(**overrides):
    header = {
        "HIERARCH ESO DET NAME": "MATISSE-LM",
        "HIERARCH ESO DET READ CURNAME": "SCI-SLOW-SPEED",
        "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
        "HIERARCH ESO DET SEQ1 DIT": 0.1,
        "HIERARCH ESO DET SEQ1 PERIOD": 0.2,
        "HIERARCH ESO INS PIL ID": "PHOTO",
        "HIERARCH ESO INS PIN ID": "PHOTO",
        "HIERARCH ESO INS DIL ID": "LOW",
        "HIERARCH ESO INS DIN ID": "LOW",
        "HIERARCH ESO INS POL ID": "POL",
        "HIERARCH ESO INS FIL ID": "FILTER",
        "HIERARCH ESO INS PON ID": "PON",
        "HIERARCH ESO INS FIN ID": "FIN",
        "HIERARCH ESO DET WIN MTRH2": 1.0,
        "HIERARCH ESO DET WIN MTRS2": 1.0,
    }
    header.update(overrides)
    return header


def test_matisse_type_identifies_sky_raw_from_sample_fits(real_obs_sky):
    header = _header(real_obs_sky)
    assert matisse_type(header) == "SKY_RAW"


def test_matisse_type_identifies_target_raw_from_sample_fits(real_obs_target):
    header = _header(real_obs_target)
    assert matisse_type(header) == "TARGET_RAW"


def test_matisse_type_identifies_calib_raw(real_obs_calib):
    header = _header(real_obs_calib)
    assert matisse_type(header) == "CALIB_RAW"


def test_matisse_type_defaults_to_category_when_no_mapping_matches():
    header = {"HIERARCH ESO PRO CATG": "UNKNOWN"}
    assert matisse_type(header) == "UNKNOWN"


def test_matisse_action_selects_fast_speed_detector_calibration():
    header = _make_base_header(**{"HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED"})
    assert matisse_action(header, "DARK") == "ACTION_MAT_CAL_DET_FAST_SPEED"


def test_matisse_action_selects_high_gain_branch():
    header = _make_base_header(
        **{
            "HIERARCH ESO DET NAME": "MATISSE-N",
            "HIERARCH ESO DET READ CURNAME": "SCI-HIGH-GAIN",
        }
    )
    assert matisse_action(header, "FLAT") == "ACTION_MAT_CAL_DET_HIGH_GAIN"


def test_matisse_action_returns_im_recipe_for_periodic_tag():
    header = _make_base_header()
    assert matisse_action(header, "IM_PERIODIC") == "ACTION_MAT_IM_REM"


def test_matisse_action_uses_raw_estimates_for_science_frame(real_obs_target):
    header = _header(real_obs_target)
    tag = matisse_type(header)
    assert matisse_action(header, tag) == "ACTION_MAT_RAW_ESTIMATES"


def test_matisse_recipes_return_expected_options_for_hawaii(real_obs_target):
    header = _header(real_obs_target)
    tag = matisse_type(header)
    action = matisse_action(header, tag)
    recipe, params = matisse_recipes(
        action,
        header["HIERARCH ESO DET CHIP NAME"],
        header.get("TELESCOP", ""),
        header.get("HIERARCH ESO INS DIL NAME", ""),
    )

    assert recipe == "mat_raw_estimates"
    assert (
        params
        == "--useOpdMod=FALSE --tartyp=57 --compensate=[pb,nl,if,rb,bp,od] --hampelFilterKernel=10"
    )


def test_matisse_recipes_return_calibration_parameters():
    recipe, params = matisse_recipes(
        "ACTION_MAT_CAL_DET_SLOW_SPEED",
        "HAWAII-2RG",
        "",
        "LOW",
    )

    assert recipe == "mat_cal_det"
    assert "--gain=2.73" in params


def test_matisse_recipes_apply_aquarius_telescope_override():
    recipe, params = matisse_recipes(
        "ACTION_MAT_RAW_ESTIMATES",
        "AQUARIUS",
        "ESO-VLTI-A1234",
        "",
    )

    assert recipe == "mat_raw_estimates"
    assert params == "--useOpdMod=TRUE --replaceTel=3"


def test_matisse_calib_short_circuits_for_detector_calibration():
    header = _make_base_header()
    existing: list[CalibEntry] = [("existing.fits", "BADPIX")]
    returned, status = matisse_calib(
        header,
        "ACTION_MAT_CAL_DET_SLOW_SPEED",
        [],
        existing,
        "2025-01-01T00:00:00",
    )

    assert status == 1
    assert returned is existing


def _write_calibration_fits(tmp_path, filename, **header_values):
    hdu = fits.PrimaryHDU()
    for key, value in header_values.items():
        hdu.header[key] = value
    path = tmp_path / filename
    hdu.writeto(path, overwrite=True)
    return path


@pytest.mark.parametrize(
    ("catg", "typ", "tech", "expected"),
    [
        ("CALIB", "DARK,DETCAL", "IMAGE", "DARK"),
        ("CALIB", "FLAT,DETCAL", "IMAGE", "FLAT"),
        ("CALIB", "DARK", "SPECTRUM", "OBSDARK"),
        ("CALIB", "FLAT", "SPECTRUM", "OBSFLAT"),
        ("CALIB", "DARK,WAVE", "IMAGE", "DISTOR_HOTDARK"),
        ("CALIB", "SOURCE,WAVE", "IMAGE", "DISTOR_IMAGES"),
        ("CALIB", "SOURCE,LAMP", "SPECTRUM", "SPECTRA_HOTDARK"),
        ("CALIB", "SOURCE,WAVE", "SPECTRUM", "SPECTRA_IMAGES"),
        ("CALIB", "DARK,FLUX", "IMAGE", "KAPPA_HOTDARK"),
        ("CALIB", "SOURCE,FLUX", "IMAGE", "KAPPA_SRC"),
        ("CALIB", "DARK,IMB", "IMAGE", "IM_COLD"),
        ("CALIB", "FLAT,IME", "IMAGE", "IM_FLAT"),
        ("CALIB", "DARK,IME", "IMAGE", "IM_DARK"),
        ("CALIB", "DARK,FLAT", "IMAGE", "IM_PERIODIC"),
        ("CALIB", "DARK", "INTERFEROMETRY", "HOT_DARK"),
        ("CALIB", "LAMP", "INTERFEROMETRY", "CALIB_SRC_RAW"),
        ("SCIENCE", "OBJECT", "IMAGE", "TARGET_RAW"),
        ("CALIB", "OBJECT", "IMAGE", "CALIB_RAW"),
        ("SCIENCE", "OBJECT", "INTERFEROMETRY", "TARGET_RAW"),
        ("CALIB", "OBJECT", "INTERFEROMETRY", "CALIB_RAW"),
        ("SCIENCE", "SKY", "INTERFEROMETRY", "SKY_RAW"),
    ],
)
def test_matisse_type_maps_known_combinations(catg, typ, tech, expected):
    header = {
        "HIERARCH ESO DPR CATG": catg,
        "HIERARCH ESO DPR TYPE": typ,
        "HIERARCH ESO DPR TECH": tech,
    }
    assert matisse_type(header) == expected


def test_matisse_calib_est_flat_collects_required_calibrations(tmp_path):
    lib_auto_pipeline.cacheHdr.cache.clear()
    header = _make_base_header()
    tplstart = "2025-01-01T02:00:00"

    badpix_path = _write_calibration_fits(
        tmp_path,
        "badpix.fits",
        **{
            "HIERARCH ESO PRO CATG": "BADPIX",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
            "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
        },
    )
    flatfield_path = _write_calibration_fits(
        tmp_path,
        "flatfield.fits",
        **{
            "HIERARCH ESO PRO CATG": "FLATFIELD",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
            "HIERARCH ESO TPL START": "2025-01-01T01:10:00",
        },
    )
    nonlinearity_path = _write_calibration_fits(
        tmp_path,
        "nonlinearity.fits",
        **{
            "HIERARCH ESO PRO CATG": "NONLINEARITY",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
            "HIERARCH ESO TPL START": "2025-01-01T01:20:00",
        },
    )

    paths = [
        str(badpix_path),
        str(flatfield_path),
        str(nonlinearity_path),
    ]

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_EST_FLAT",
        paths,
        [],
        tplstart,
    )

    assert status == 1
    assert {tag for _, tag in returned} == {
        "BADPIX",
        "FLATFIELD",
        "NONLINEARITY",
    }


def test_matisse_calib_im_basic_prefers_closest_badpix(tmp_path):
    lib_auto_pipeline.cacheHdr.cache.clear()
    header = _make_base_header()
    tplstart = "2025-01-01T01:30:00"

    older_path = _write_calibration_fits(
        tmp_path,
        "older_badpix.fits",
        **{
            "HIERARCH ESO PRO CATG": "BADPIX",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO TPL START": "2025-01-01T00:00:00",
        },
    )
    closer_path = _write_calibration_fits(
        tmp_path,
        "closer_badpix.fits",
        **{
            "HIERARCH ESO PRO CATG": "BADPIX",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
        },
    )

    existing = [(str(older_path), "BADPIX")]

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_IM_BASIC",
        [str(closer_path)],
        existing,
        tplstart,
    )

    assert status == 1
    assert returned[0][0] == str(closer_path)


def test_matisse_calib_raw_estimates_collects_full_set(tmp_path):
    lib_auto_pipeline.cacheHdr.cache.clear()
    header = _make_base_header(**{"HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED"})
    tplstart = "2025-01-01T02:45:00"
    common = {
        "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
        "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
        "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
        "HIERARCH ESO DET SEQ1 PERIOD": header["HIERARCH ESO DET SEQ1 PERIOD"],
        "HIERARCH ESO INS PIL ID": header["HIERARCH ESO INS PIL ID"],
        "HIERARCH ESO INS PIN ID": header["HIERARCH ESO INS PIN ID"],
        "HIERARCH ESO INS DIL ID": header["HIERARCH ESO INS DIL ID"],
        "HIERARCH ESO INS DIN ID": header["HIERARCH ESO INS DIN ID"],
        "HIERARCH ESO INS POL ID": header["HIERARCH ESO INS POL ID"],
        "HIERARCH ESO INS FIL ID": header["HIERARCH ESO INS FIL ID"],
        "HIERARCH ESO INS PON ID": header["HIERARCH ESO INS PON ID"],
        "HIERARCH ESO INS FIN ID": header["HIERARCH ESO INS FIN ID"],
        "HIERARCH ESO DET WIN MTRH2": header["HIERARCH ESO DET WIN MTRH2"],
        "HIERARCH ESO DET WIN MTRS2": header["HIERARCH ESO DET WIN MTRS2"],
    }

    badpix_path = _write_calibration_fits(
        tmp_path,
        "raw_badpix.fits",
        **{
            **common,
            "HIERARCH ESO PRO CATG": "BADPIX",
            "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
        },
    )
    obs_flatfield_path = _write_calibration_fits(
        tmp_path,
        "obs_flatfield.fits",
        **{
            **common,
            "HIERARCH ESO PRO CATG": "OBS_FLATFIELD",
            "HIERARCH ESO TPL START": "2025-01-01T01:05:00",
        },
    )
    nonlinearity_path = _write_calibration_fits(
        tmp_path,
        "raw_nonlinearity.fits",
        **{
            **common,
            "HIERARCH ESO PRO CATG": "NONLINEARITY",
            "HIERARCH ESO TPL START": "2025-01-01T01:10:00",
        },
    )
    shift_map_path = _write_calibration_fits(
        tmp_path,
        "shift_map.fits",
        **{
            **common,
            "HIERARCH ESO PRO CATG": "SHIFT_MAP",
            "HIERARCH ESO TPL START": "2025-01-01T01:15:00",
        },
    )
    kappa_matrix_path = _write_calibration_fits(
        tmp_path,
        "kappa_matrix.fits",
        **{
            **common,
            "HIERARCH ESO PRO CATG": "KAPPA_MATRIX",
            "HIERARCH ESO TPL START": "2025-01-01T01:20:00",
        },
    )

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_RAW_ESTIMATES",
        [
            str(badpix_path),
            str(obs_flatfield_path),
            str(nonlinearity_path),
            str(shift_map_path),
            str(kappa_matrix_path),
        ],
        [],
        tplstart,
    )

    assert status == 1
    assert {tag for _, tag in returned} == {
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
        "SHIFT_MAP",
        "KAPPA_MATRIX",
    }
