from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from astropy.io import fits

import matisse.core.lib_auto_pipeline as lib_auto_pipeline
from matisse.core.lib_auto_pipeline import (
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


def test_matisse_type_identifies_target_raw_from_sample_fits(real_obs_target):
    header = _header(real_obs_target)
    assert matisse_type(header) == "TARGET_RAW"


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


def test_matisse_calib_est_flat_prefers_closest_flatfield(tmp_path):
    header = _make_base_header()
    tplstart = "2025-01-01T02:30:00"

    badpix_path = _write_calibration_fits(
        tmp_path,
        "existing_badpix.fits",
        **{
            "HIERARCH ESO PRO CATG": "BADPIX",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
            "HIERARCH ESO DET SEQ1 PERIOD": header["HIERARCH ESO DET SEQ1 PERIOD"],
            "HIERARCH ESO TPL START": "2025-01-01T02:00:00",
        },
    )
    older_flatfield_path = _write_calibration_fits(
        tmp_path,
        "older_flatfield.fits",
        **{
            "HIERARCH ESO PRO CATG": "FLATFIELD",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
            "HIERARCH ESO DET SEQ1 PERIOD": header["HIERARCH ESO DET SEQ1 PERIOD"],
            "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
        },
    )
    nonlinearity_path = _write_calibration_fits(
        tmp_path,
        "existing_nonlinearity.fits",
        **{
            "HIERARCH ESO PRO CATG": "NONLINEARITY",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
            "HIERARCH ESO DET SEQ1 PERIOD": header["HIERARCH ESO DET SEQ1 PERIOD"],
            "HIERARCH ESO TPL START": "2025-01-01T02:05:00",
        },
    )
    closer_flatfield_path = _write_calibration_fits(
        tmp_path,
        "closer_flatfield.fits",
        **{
            "HIERARCH ESO PRO CATG": "FLATFIELD",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
            "HIERARCH ESO DET SEQ1 PERIOD": header["HIERARCH ESO DET SEQ1 PERIOD"],
            "HIERARCH ESO TPL START": "2025-01-01T02:20:00",
        },
    )

    existing = [
        (str(badpix_path), "BADPIX"),
        (str(older_flatfield_path), "FLATFIELD"),
        (str(nonlinearity_path), "NONLINEARITY"),
    ]

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_EST_FLAT",
        [str(closer_flatfield_path)],
        existing,
        tplstart,
    )

    assert status == 1
    assert any(
        path == str(closer_flatfield_path) and tag == "FLATFIELD"
        for path, tag in returned
    )
    assert all(
        path != str(older_flatfield_path)
        for path, tag in returned
        if tag == "FLATFIELD"
    )


@pytest.mark.parametrize(
    "tag",
    [
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
        "SHIFT_MAP",
        "KAPPA_MATRIX",
    ],
)
def test_matisse_calib_raw_estimates_prefers_closest_calibration(tmp_path, tag):
    header = _make_base_header(
        **{
            "HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED",
            "HIERARCH ESO INS DIL ID": "LOW",
            "HIERARCH ESO INS DIN ID": "LOW",
        }
    )
    tplstart = "2025-01-01T02:45:00"

    common = {
        key: header[key]
        for key in [
            "HIERARCH ESO DET READ CURNAME",
            "HIERARCH ESO DET CHIP NAME",
            "HIERARCH ESO DET SEQ1 DIT",
            "HIERARCH ESO DET SEQ1 PERIOD",
            "HIERARCH ESO INS PIL ID",
            "HIERARCH ESO INS PIN ID",
            "HIERARCH ESO INS DIL ID",
            "HIERARCH ESO INS DIN ID",
            "HIERARCH ESO INS POL ID",
            "HIERARCH ESO INS FIL ID",
            "HIERARCH ESO INS PON ID",
            "HIERARCH ESO INS FIN ID",
            "HIERARCH ESO DET WIN MTRH2",
            "HIERARCH ESO DET WIN MTRS2",
        ]
    }

    tags = [
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
        "SHIFT_MAP",
        "KAPPA_MATRIX",
    ]

    existing: list[CalibEntry] = []
    old_target_path: str | None = None
    new_target_path: str | None = None

    for current in tags:
        old_path = _write_calibration_fits(
            tmp_path,
            f"old_{current.lower()}.fits",
            **{
                **common,
                "HIERARCH ESO PRO CATG": current,
                "HIERARCH ESO TPL START": "2025-01-01T00:00:00",
            },
        )
        existing.append((str(old_path), current))
        if current == tag:
            new_path = _write_calibration_fits(
                tmp_path,
                f"new_{current.lower()}.fits",
                **{
                    **common,
                    "HIERARCH ESO PRO CATG": current,
                    "HIERARCH ESO TPL START": "2025-01-01T02:30:00",
                },
            )
            new_target_path = str(new_path)
            old_target_path = str(old_path)

    assert new_target_path is not None
    assert old_target_path is not None

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_RAW_ESTIMATES",
        [new_target_path],
        existing,
        tplstart,
    )

    assert status == 1
    assert any(
        path == new_target_path and calib_tag == tag for path, calib_tag in returned
    )
    assert all(
        path != old_target_path for path, calib_tag in returned if calib_tag == tag
    )


@pytest.mark.parametrize(
    "tag",
    [
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
        "SHIFT_MAP",
    ],
)
def test_matisse_calib_est_kappa_prefers_closest_calibration(tmp_path, tag):
    header = _make_base_header(
        **{
            "HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED",
            "HIERARCH ESO INS DIL ID": "LOW",
            "HIERARCH ESO INS DIN ID": "LOW",
        }
    )
    tplstart = "2025-01-01T03:15:00"

    common = {
        key: header[key]
        for key in [
            "HIERARCH ESO DET READ CURNAME",
            "HIERARCH ESO DET CHIP NAME",
            "HIERARCH ESO DET SEQ1 DIT",
            "HIERARCH ESO DET SEQ1 PERIOD",
            "HIERARCH ESO INS PIL ID",
            "HIERARCH ESO INS PIN ID",
            "HIERARCH ESO INS DIL ID",
            "HIERARCH ESO INS DIN ID",
            "HIERARCH ESO INS POL ID",
            "HIERARCH ESO INS FIL ID",
            "HIERARCH ESO INS PON ID",
            "HIERARCH ESO INS FIN ID",
            "HIERARCH ESO DET WIN MTRH2",
            "HIERARCH ESO DET WIN MTRS2",
        ]
    }

    tags = [
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
        "SHIFT_MAP",
    ]

    existing: list[CalibEntry] = []
    old_target_path: str | None = None
    new_target_path: str | None = None

    for current in tags:
        old_path = _write_calibration_fits(
            tmp_path,
            f"kappa_old_{current.lower()}.fits",
            **{
                **common,
                "HIERARCH ESO PRO CATG": current,
                "HIERARCH ESO TPL START": "2025-01-01T00:00:00",
            },
        )
        existing.append((str(old_path), current))
        if current == tag:
            new_path = _write_calibration_fits(
                tmp_path,
                f"kappa_new_{current.lower()}.fits",
                **{
                    **common,
                    "HIERARCH ESO PRO CATG": current,
                    "HIERARCH ESO TPL START": "2025-01-01T03:05:00",
                },
            )
            new_target_path = str(new_path)
            old_target_path = str(old_path)

    assert new_target_path is not None
    assert old_target_path is not None

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_EST_KAPPA",
        [new_target_path],
        existing,
        tplstart,
    )

    assert status == 1
    assert any(
        path == new_target_path and calib_tag == tag for path, calib_tag in returned
    )
    assert all(
        path != old_target_path for path, calib_tag in returned if calib_tag == tag
    )


@pytest.mark.parametrize(
    "tag",
    [
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
    ],
)
def test_matisse_calib_est_shift_prefers_closest_calibration(tmp_path, tag):
    header = _make_base_header(
        **{
            "HIERARCH ESO DET READ CURNAME": "SCI-SLOW-SPEED",
            "HIERARCH ESO INS DIL ID": "LOW",
            "HIERARCH ESO INS DIN ID": "LOW",
        }
    )
    tplstart = "2025-01-01T03:45:00"

    common = {
        key: header[key]
        for key in [
            "HIERARCH ESO DET READ CURNAME",
            "HIERARCH ESO DET CHIP NAME",
            "HIERARCH ESO DET SEQ1 DIT",
            "HIERARCH ESO DET SEQ1 PERIOD",
            "HIERARCH ESO INS PIL ID",
            "HIERARCH ESO INS PIN ID",
            "HIERARCH ESO INS DIL ID",
            "HIERARCH ESO INS DIN ID",
            "HIERARCH ESO INS POL ID",
            "HIERARCH ESO INS FIL ID",
            "HIERARCH ESO INS PON ID",
            "HIERARCH ESO INS FIN ID",
            "HIERARCH ESO DET WIN MTRH2",
            "HIERARCH ESO DET WIN MTRS2",
        ]
    }

    tags = [
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
    ]

    existing: list[CalibEntry] = []
    old_target_path: str | None = None
    new_target_path: str | None = None

    for current in tags:
        old_path = _write_calibration_fits(
            tmp_path,
            f"shift_old_{current.lower()}.fits",
            **{
                **common,
                "HIERARCH ESO PRO CATG": current,
                "HIERARCH ESO TPL START": "2025-01-01T00:00:00",
            },
        )
        existing.append((str(old_path), current))
        if current == tag:
            new_path = _write_calibration_fits(
                tmp_path,
                f"shift_new_{current.lower()}.fits",
                **{
                    **common,
                    "HIERARCH ESO PRO CATG": current,
                    "HIERARCH ESO TPL START": "2025-01-01T03:35:00",
                },
            )
            new_target_path = str(new_path)
            old_target_path = str(old_path)

    assert new_target_path is not None
    assert old_target_path is not None

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_EST_SHIFT",
        [new_target_path],
        existing,
        tplstart,
    )

    assert status == 1
    assert any(
        path == new_target_path and calib_tag == tag for path, calib_tag in returned
    )
    assert all(
        path != old_target_path for path, calib_tag in returned if calib_tag == tag
    )


def test_matisse_calib_im_basic_prefers_closest_badpix(tmp_path):
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


# ---------------------------------------------------------------------------
# Shared fixtures / helpers for the tests below
# ---------------------------------------------------------------------------


_CALIB_MATCH_KEYS = (
    "HIERARCH ESO DET READ CURNAME",
    "HIERARCH ESO DET CHIP NAME",
    "HIERARCH ESO DET SEQ1 DIT",
    "HIERARCH ESO DET SEQ1 PERIOD",
    "HIERARCH ESO INS PIL ID",
    "HIERARCH ESO INS PIN ID",
    "HIERARCH ESO INS DIL ID",
    "HIERARCH ESO INS DIN ID",
    "HIERARCH ESO INS POL ID",
    "HIERARCH ESO INS FIL ID",
    "HIERARCH ESO INS PON ID",
    "HIERARCH ESO INS FIN ID",
    "HIERARCH ESO DET WIN MTRH2",
    "HIERARCH ESO DET WIN MTRS2",
)


def _matching_calib_keys(header, **overrides):
    """Return the calibration header keys that make a file match ``header``."""
    values = {key: header[key] for key in _CALIB_MATCH_KEYS}
    values.update(overrides)
    return values


# ---------------------------------------------------------------------------
# headerCache
# ---------------------------------------------------------------------------


def test_header_cache_reports_its_size_and_membership():
    """headerCache.size mirrors the number of cached entries."""
    cache = lib_auto_pipeline.headerCache()

    assert cache.size == 0
    assert "a.fits" not in cache

    cache.update("a.fits", {"KEY": 1})

    assert cache.size == 1
    assert "a.fits" in cache
    assert cache.cache["a.fits"]["value"] == {"KEY": 1}


# ---------------------------------------------------------------------------
# probe_spectral_param_name
# ---------------------------------------------------------------------------


def _fake_esorex_run(recorded, stdout="", stderr="", exc=None):
    """Build a subprocess.run replacement recording argv and returning canned output."""

    def _run(cmd, **_kwargs):
        recorded.append(list(cmd))
        if exc is not None:
            raise exc
        return SimpleNamespace(stdout=stdout, stderr=stderr)

    return _run


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("--spectralAverage=<int> average spectral channels", "", "spectralAverage"),
        ("", "--spectralBinning=<int> bin spectral channels", "spectralBinning"),
        ("nothing relevant in this man page", "", "spectralAverage"),
        ("spectralAverage and spectralBinning both listed", "", "spectralAverage"),
    ],
)
def test_probe_spectral_param_name_detects_recipe_option(
    monkeypatch, stdout, stderr, expected
):
    """The parameter name is taken from the esorex man page, defaulting to spectralAverage."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        lib_auto_pipeline.subprocess,
        "run",
        _fake_esorex_run(recorded, stdout=stdout, stderr=stderr),
    )

    assert lib_auto_pipeline.probe_spectral_param_name() == expected
    assert recorded == [["esorex", "--man", "mat_raw_estimates"]]


def test_probe_spectral_param_name_forwards_recipe_dir(monkeypatch, tmp_path):
    """A recipe_dir is forwarded to esorex as --recipe-dir=<path>."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        lib_auto_pipeline.subprocess,
        "run",
        _fake_esorex_run(recorded, stdout="--spectralBinning=<int>"),
    )

    assert lib_auto_pipeline.probe_spectral_param_name(recipe_dir=tmp_path) == (
        "spectralBinning"
    )
    assert recorded == [
        ["esorex", f"--recipe-dir={tmp_path}", "--man", "mat_raw_estimates"]
    ]


def test_probe_spectral_param_name_falls_back_when_esorex_missing(monkeypatch):
    """A missing esorex binary yields the documented spectralAverage fallback."""
    recorded: list[list[str]] = []
    monkeypatch.setattr(
        lib_auto_pipeline.subprocess,
        "run",
        _fake_esorex_run(recorded, exc=FileNotFoundError("esorex")),
    )

    assert lib_auto_pipeline.probe_spectral_param_name() == "spectralAverage"
    assert recorded == [["esorex", "--man", "mat_raw_estimates"]]


# ---------------------------------------------------------------------------
# matisse_recipes
# ---------------------------------------------------------------------------


_CAL_DET_SLOW = (
    "--gain=2.73 --darklimit=100.0 --flatlimit=0.3 --max_nonlinear_range=36000.0 "
    "--max_abs_deviation=2000.0 --max_rel_deviation=0.01 --nltype=2"
)
_CAL_DET_FAST = (
    "--gain=2.60 --darklimit=100.0 --flatlimit=0.3 --max_nonlinear_range=36000.0 "
    "--max_abs_deviation=2000.0 --max_rel_deviation=0.01 --nltype=2"
)
_CAL_DET_LOW_GAIN = (
    "--gain=190.0 --darklimit=100.0 --flatlimit=0.2 --max_nonlinear_range=36000.0 "
    "--max_abs_deviation=2000.0 --max_rel_deviation=0.02 --nt=true --nltype=2"
)
_CAL_DET_HIGH_GAIN = (
    "--gain=20.0 --darklimit=200.0 --flatlimit=0.2 --max_nonlinear_range=36000.0 "
    "--max_abs_deviation=2000.0 --max_rel_deviation=0.01 --nt=true --nltype=2"
)
_RAW_HAWAII = (
    "--useOpdMod=FALSE --tartyp=57 --compensate=[pb,nl,if,rb,bp,od] "
    "--hampelFilterKernel=10"
)


@pytest.mark.parametrize(
    ("action", "det", "tel", "expected"),
    [
        (
            "ACTION_MAT_CAL_DET_SLOW_SPEED",
            "HAWAII-2RG",
            "",
            ["mat_cal_det", _CAL_DET_SLOW],
        ),
        (
            "ACTION_MAT_CAL_DET_FAST_SPEED",
            "HAWAII-2RG",
            "",
            ["mat_cal_det", _CAL_DET_FAST],
        ),
        (
            "ACTION_MAT_CAL_DET_LOW_GAIN",
            "AQUARIUS",
            "",
            ["mat_cal_det", _CAL_DET_LOW_GAIN],
        ),
        (
            "ACTION_MAT_CAL_DET_HIGH_GAIN",
            "AQUARIUS",
            "",
            ["mat_cal_det", _CAL_DET_HIGH_GAIN],
        ),
        (
            "ACTION_MAT_EST_FLAT",
            "HAWAII-2RG",
            "",
            ["mat_est_flat", "--obsflat_type=det"],
        ),
        (
            "ACTION_MAT_EST_SHIFT",
            "HAWAII-2RG",
            "",
            ["mat_est_shift", "--obsCorrection=TRUE"],
        ),
        ("ACTION_MAT_EST_KAPPA", "HAWAII-2RG", "", ["mat_est_kappa", ""]),
        (
            "ACTION_MAT_RAW_ESTIMATES",
            "HAWAII-2RG",
            "ESO-VLTI-U1234",
            ["mat_raw_estimates", _RAW_HAWAII],
        ),
        (
            "ACTION_MAT_RAW_ESTIMATES",
            "AQUARIUS",
            "ESO-VLTI-U1234",
            ["mat_raw_estimates", "--useOpdMod=TRUE"],
        ),
        (
            "ACTION_MAT_RAW_ESTIMATES",
            "AQUARIUS",
            "ESO-VLTI-A1234",
            ["mat_raw_estimates", "--useOpdMod=TRUE --replaceTel=3"],
        ),
        ("ACTION_MAT_RAW_ESTIMATES", "UNKNOWN-DET", "", ["mat_raw_estimates", ""]),
        ("ACTION_MAT_IM_BASIC", "HAWAII-2RG", "", ["mat_im_basic", ""]),
        ("ACTION_MAT_IM_EXTENDED", "HAWAII-2RG", "", ["mat_im_extended", ""]),
        ("ACTION_MAT_IM_REM", "HAWAII-2RG", "", ["mat_im_rem", ""]),
        ("NO-ACTION", "HAWAII-2RG", "", ["", ""]),
        ("ACTION_MAT_DOES_NOT_EXIST", "HAWAII-2RG", "", ["", ""]),
    ],
)
def test_matisse_recipes_returns_expected_recipe_and_options(
    action, det, tel, expected
):
    """Every known action maps to its recipe name and CLI options."""
    assert matisse_recipes(action, det, tel, "LOW") == expected


# ---------------------------------------------------------------------------
# matisse_action
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("det_name", "read_curname", "tag", "expected"),
    [
        ("MATISSE-LM", "SCI-SLOW-SPEED", "DARK", "ACTION_MAT_CAL_DET_SLOW_SPEED"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "FLAT", "ACTION_MAT_CAL_DET_SLOW_SPEED"),
        ("MATISSE-LM", "SCI-FAST-SPEED", "DARK", "ACTION_MAT_CAL_DET_FAST_SPEED"),
        ("MATISSE-N", "SCI-LOW-GAIN", "DARK", "ACTION_MAT_CAL_DET_LOW_GAIN"),
        ("MATISSE-N", "SCI-LOW-GAIN", "FLAT", "ACTION_MAT_CAL_DET_LOW_GAIN"),
        ("MATISSE-N", "SCI-HIGH-GAIN", "DARK", "ACTION_MAT_CAL_DET_HIGH_GAIN"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "OBSDARK", "ACTION_MAT_EST_FLAT"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "OBSFLAT", "ACTION_MAT_EST_FLAT"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "DISTOR_HOTDARK", "ACTION_MAT_EST_SHIFT"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "DISTOR_IMAGES", "ACTION_MAT_EST_SHIFT"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "SPECTRA_HOTDARK", "ACTION_MAT_EST_SHIFT"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "SPECTRA_IMAGES", "ACTION_MAT_EST_SHIFT"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "KAPPA_HOTDARK", "ACTION_MAT_EST_KAPPA"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "KAPPA_SRC", "ACTION_MAT_EST_KAPPA"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "KAPPA_SKY", "ACTION_MAT_EST_KAPPA"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "KAPPA_OBJ", "ACTION_MAT_EST_KAPPA"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "TARGET_RAW", "ACTION_MAT_RAW_ESTIMATES"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "CALIB_RAW", "ACTION_MAT_RAW_ESTIMATES"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "HOT_DARK", "ACTION_MAT_RAW_ESTIMATES"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "CALIB_SRC_RAW", "ACTION_MAT_RAW_ESTIMATES"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "SKY_RAW", "ACTION_MAT_RAW_ESTIMATES"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "IM_COLD", "ACTION_MAT_IM_BASIC"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "IM_FLAT", "ACTION_MAT_IM_EXTENDED"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "IM_DARK", "ACTION_MAT_IM_EXTENDED"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "IM_PERIODIC", "ACTION_MAT_IM_REM"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "DARK", "ACTION_MAT_CAL_DET_SLOW_SPEED"),
        ("MATISSE-N", "SCI-SLOW-SPEED", "DARK", "NO-ACTION"),
        ("MATISSE-LM", "SCI-SLOW-SPEED", "UNKNOWN_TAG", "NO-ACTION"),
    ],
)
def test_matisse_action_maps_tag_and_detector_to_action(
    det_name, read_curname, tag, expected
):
    """Each tag/detector combination selects the documented pipeline action."""
    header = _make_base_header(
        **{
            "HIERARCH ESO DET NAME": det_name,
            "HIERARCH ESO DET READ CURNAME": read_curname,
        }
    )
    assert matisse_action(header, tag) == expected


def test_matisse_type_returns_empty_string_without_any_category_keyword():
    """A header carrying neither PRO CATG nor DPR keys degrades to an empty tag."""
    assert matisse_type({}) == ""


# ---------------------------------------------------------------------------
# matisse_calib - image actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    ["ACTION_MAT_IM_BASIC", "ACTION_MAT_IM_EXTENDED", "ACTION_MAT_IM_REM"],
)
def test_matisse_calib_image_actions_append_missing_badpix(tmp_path, action):
    """Image actions append the BADPIX calibration when none is registered yet."""
    header = _make_base_header()
    badpix = _write_calibration_fits(
        tmp_path,
        f"{action.lower()}_badpix.fits",
        **{
            "HIERARCH ESO PRO CATG": "BADPIX",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO TPL START": "2025-01-01T00:30:00",
        },
    )

    returned, status = matisse_calib(
        header, action, [str(badpix)], [], "2025-01-01T01:00:00"
    )

    assert status == 1
    assert returned == [(str(badpix), "BADPIX")]


def test_matisse_calib_image_action_reports_incomplete_without_badpix(tmp_path):
    """An image action without any BADPIX candidate reports an incomplete set."""
    header = _make_base_header()
    flatfield = _write_calibration_fits(
        tmp_path,
        "im_basic_flatfield.fits",
        **{
            "HIERARCH ESO PRO CATG": "FLATFIELD",
            "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
            "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
            "HIERARCH ESO TPL START": "2025-01-01T00:30:00",
        },
    )

    returned, status = matisse_calib(
        header, "ACTION_MAT_IM_BASIC", [str(flatfield)], [], "2025-01-01T01:00:00"
    )

    assert status == 0
    assert returned == []


# ---------------------------------------------------------------------------
# matisse_calib - ACTION_MAT_EST_FLAT duplicate handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("catg", ["BADPIX", "NONLINEARITY"])
@pytest.mark.parametrize(
    ("candidate_tpl", "replaced"),
    [("2025-01-01T01:50:00", True), ("2025-01-01T00:10:00", False)],
)
def test_matisse_calib_est_flat_keeps_closest_duplicate(
    tmp_path, catg, candidate_tpl, replaced
):
    """EST_FLAT replaces an already registered calibration only when the new one is closer."""
    header = _make_base_header()
    tplstart = "2025-01-01T02:00:00"
    common = {
        "HIERARCH ESO PRO CATG": catg,
        "HIERARCH ESO DET READ CURNAME": header["HIERARCH ESO DET READ CURNAME"],
        "HIERARCH ESO DET CHIP NAME": header["HIERARCH ESO DET CHIP NAME"],
        "HIERARCH ESO DET SEQ1 DIT": header["HIERARCH ESO DET SEQ1 DIT"],
    }

    previous = _write_calibration_fits(
        tmp_path,
        f"prev_{catg.lower()}.fits",
        **{**common, "HIERARCH ESO TPL START": "2025-01-01T01:00:00"},
    )
    candidate = _write_calibration_fits(
        tmp_path,
        f"cand_{catg.lower()}.fits",
        **{**common, "HIERARCH ESO TPL START": candidate_tpl},
    )

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_EST_FLAT",
        [str(candidate)],
        [(str(previous), catg)],
        tplstart,
    )

    expected = str(candidate) if replaced else str(previous)
    assert [entry for entry in returned if entry[1] == catg] == [(expected, catg)]
    # Only one of the three required calibrations is known: the set is incomplete.
    assert status == 0


# ---------------------------------------------------------------------------
# matisse_calib - ACTION_MAT_RAW_ESTIMATES
# ---------------------------------------------------------------------------


def test_matisse_calib_raw_estimates_warns_once_on_dit_mismatch(
    tmp_path, monkeypatch, caplog
):
    """A flatfield taken with another DIT is reported and not used as calibration."""
    monkeypatch.setattr(lib_auto_pipeline, "_warning_shown", False)
    header = _make_base_header(
        **{"HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED"},
    )
    flatfield = _write_calibration_fits(
        tmp_path,
        "dit_mismatch_flatfield.fits",
        **_matching_calib_keys(
            header,
            **{
                "HIERARCH ESO PRO CATG": "OBS_FLATFIELD",
                "HIERARCH ESO DET SEQ1 DIT": 5.0,
                "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
            },
        ),
    )

    with caplog.at_level(logging.DEBUG, logger="matisse.core.lib_auto_pipeline"):
        returned, status = matisse_calib(
            header,
            "ACTION_MAT_RAW_ESTIMATES",
            [str(flatfield)],
            [],
            "2025-01-01T02:00:00",
        )

    assert "Different DIT detected for flatfield" in caplog.text
    assert lib_auto_pipeline._warning_shown is True
    assert returned == []
    assert status == 0


@pytest.mark.parametrize("already_present", [False, True])
def test_matisse_calib_raw_estimates_registers_jsdc_catalog_once(
    tmp_path, already_present
):
    """The JSDC catalogue is appended only when it is not registered yet."""
    header = _make_base_header()
    candidate = _write_calibration_fits(
        tmp_path,
        "jsdc_candidate.fits",
        **{"HIERARCH ESO PRO CATG": "JSDC_CAT"},
    )
    previous_jsdc = _write_calibration_fits(
        tmp_path,
        "jsdc_previous.fits",
        **{"HIERARCH ESO PRO CATG": "JSDC_CAT"},
    )
    previous: list[CalibEntry] = (
        [(str(previous_jsdc), "JSDC_CAT")] if already_present else []
    )

    returned, status = matisse_calib(
        header,
        "ACTION_MAT_RAW_ESTIMATES",
        [str(candidate)],
        previous,
        "2025-01-01T02:00:00",
    )

    jsdc_entries = [entry for entry in returned if entry[1] == "JSDC_CAT"]
    if already_present:
        assert jsdc_entries == [(str(previous_jsdc), "JSDC_CAT")]
    else:
        assert jsdc_entries == [(str(candidate), "JSDC_CAT")]
    # No science calibration was provided, the set stays incomplete.
    assert status == 0


@pytest.mark.parametrize(
    ("catgs", "expected_status"),
    [
        (["BADPIX", "OBS_FLATFIELD", "NONLINEARITY", "SHIFT_MAP"], 1),
        (["BADPIX", "OBS_FLATFIELD", "NONLINEARITY"], 0),
    ],
)
def test_matisse_calib_raw_estimates_status_without_photometry(
    tmp_path, catgs, expected_status
):
    """Without photometric beams four calibrations are enough to complete the set."""
    header = _make_base_header(
        **{
            "HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED",
            "HIERARCH ESO INS PIL ID": "INTER",
        }
    )
    paths = [
        str(
            _write_calibration_fits(
                tmp_path,
                f"nophoto_{catg.lower()}.fits",
                **_matching_calib_keys(
                    header,
                    **{
                        "HIERARCH ESO PRO CATG": catg,
                        "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
                    },
                ),
            )
        )
        for catg in catgs
    ]

    returned, status = matisse_calib(
        header, "ACTION_MAT_RAW_ESTIMATES", paths, [], "2025-01-01T02:00:00"
    )

    assert status == expected_status
    assert {tag for _, tag in returned} == set(catgs)


# ---------------------------------------------------------------------------
# matisse_calib - ACTION_MAT_EST_KAPPA / ACTION_MAT_EST_SHIFT
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("catgs", "expected_status"),
    [
        (["BADPIX", "OBS_FLATFIELD", "NONLINEARITY", "SHIFT_MAP"], 1),
        (["BADPIX", "OBS_FLATFIELD", "NONLINEARITY"], 0),
    ],
)
def test_matisse_calib_est_kappa_collects_calibrations(
    tmp_path, catgs, expected_status
):
    """EST_KAPPA is complete only once its four calibrations have been appended."""
    header = _make_base_header(**{"HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED"})
    paths = [
        str(
            _write_calibration_fits(
                tmp_path,
                f"kappa_new_{catg.lower()}.fits",
                **_matching_calib_keys(
                    header,
                    **{
                        "HIERARCH ESO PRO CATG": catg,
                        "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
                    },
                ),
            )
        )
        for catg in catgs
    ]

    returned, status = matisse_calib(
        header, "ACTION_MAT_EST_KAPPA", paths, [], "2025-01-01T02:00:00"
    )

    assert status == expected_status
    assert [tag for _, tag in returned] == catgs
    assert [path for path, _ in returned] == paths


@pytest.mark.parametrize(
    ("catgs", "expected_status"),
    [
        (["BADPIX", "OBS_FLATFIELD", "NONLINEARITY"], 1),
        (["BADPIX", "NONLINEARITY"], 0),
    ],
)
def test_matisse_calib_est_shift_collects_calibrations(
    tmp_path, catgs, expected_status
):
    """EST_SHIFT is complete only once its three calibrations have been appended."""
    header = _make_base_header(**{"HIERARCH ESO DET READ CURNAME": "SCI-SLOW-SPEED"})
    paths = [
        str(
            _write_calibration_fits(
                tmp_path,
                f"shift_new2_{catg.lower()}.fits",
                **_matching_calib_keys(
                    header,
                    **{
                        "HIERARCH ESO PRO CATG": catg,
                        "HIERARCH ESO TPL START": "2025-01-01T01:00:00",
                    },
                ),
            )
        )
        for catg in catgs
    ]

    returned, status = matisse_calib(
        header, "ACTION_MAT_EST_SHIFT", paths, [], "2025-01-01T02:00:00"
    )

    assert status == expected_status
    assert [tag for _, tag in returned] == catgs
    assert [path for path, _ in returned] == paths


def test_matisse_calib_returns_incomplete_for_unknown_action():
    """An unhandled action returns the untouched calibration list and status 0."""
    header = _make_base_header()
    previous: list[CalibEntry] = [("previous.fits", "BADPIX")]

    returned, status = matisse_calib(
        header, "ACTION_MAT_NOT_A_REAL_ACTION", [], previous, "2025-01-01T02:00:00"
    )

    assert status == 0
    assert returned is previous
    assert returned == [("previous.fits", "BADPIX")]


# ---------------------------------------------------------------------------
# add_mdfc_fluxes
# ---------------------------------------------------------------------------


_MDFC_DEFAULT_RESULT = object()


class _DummyVizier:
    """Minimal Vizier stub returning a single MDFC row and recording its queries."""

    def __init__(self, result=_MDFC_DEFAULT_RESULT):
        if result is _MDFC_DEFAULT_RESULT:
            result = [[[1.5, 2.5, 3.5]]]
        self.result = result
        self.queries: list[str] = []

    def query_region(self, target, **_kwargs):
        self.queries.append(target)
        return self.result


def test_add_mdfc_fluxes_writes_catalog_values(tmp_path, caplog):
    """Fluxes returned by Vizier are written into the OIFITS primary header."""
    path = _write_calibration_fits(
        tmp_path, "oifits_ok.fits", **{"ESO OBS TARG NAME": "HD1234"}
    )
    vizier = _DummyVizier()

    with caplog.at_level(logging.INFO, logger="matisse.core.lib_auto_pipeline"):
        lib_auto_pipeline.add_mdfc_fluxes([path], vizier)

    header = fits.getheader(path, 0)
    assert vizier.queries == ["HD1234"]
    assert header["HIERARCH PRO MDFC FLUX L"] == 1.5
    assert header["HIERARCH PRO MDFC FLUX M"] == 2.5
    assert header["HIERARCH PRO MDFC FLUX N"] == 3.5
    assert "Added MDFC fluxes for HD1234" in caplog.text


def test_add_mdfc_fluxes_skips_unreadable_and_targetless_files(tmp_path, caplog):
    """Files with an unreadable header or no target name never reach Vizier."""
    broken = tmp_path / "broken.fits"
    broken.write_text("this is definitely not a FITS file\n")
    no_target = _write_calibration_fits(tmp_path, "no_target.fits", **{"FOO": "BAR"})
    vizier = _DummyVizier()

    with caplog.at_level(logging.WARNING, logger="matisse.core.lib_auto_pipeline"):
        lib_auto_pipeline.add_mdfc_fluxes([broken, no_target], vizier)

    assert vizier.queries == []
    assert "No target name (ESO OBS TARG NAME) in 2 file(s)" in caplog.text


def test_add_mdfc_fluxes_warns_when_write_back_fails(tmp_path, monkeypatch, caplog):
    """A failing write-back is reported and leaves the file untouched."""
    path = _write_calibration_fits(
        tmp_path, "oifits_readonly.fits", **{"ESO OBS TARG NAME": "HD4321"}
    )
    vizier = _DummyVizier()

    def _boom(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(lib_auto_pipeline.fits, "open", _boom)

    with caplog.at_level(logging.WARNING, logger="matisse.core.lib_auto_pipeline"):
        lib_auto_pipeline.add_mdfc_fluxes([path], vizier)

    assert vizier.queries == ["HD4321"]
    assert "Failed to write MDFC fluxes in oifits_readonly.fits" in caplog.text
    monkeypatch.undo()
    assert "HIERARCH PRO MDFC FLUX L" not in fits.getheader(path, 0)


@pytest.mark.parametrize("result", [None, [], [[]]])
def test_add_mdfc_fluxes_ignores_targets_missing_from_catalog(tmp_path, result):
    """Empty Vizier answers leave the OIFITS header unchanged."""
    path = _write_calibration_fits(
        tmp_path, f"oifits_empty_{id(result)}.fits", **{"ESO OBS TARG NAME": "HD9999"}
    )

    lib_auto_pipeline.add_mdfc_fluxes([path], _DummyVizier(result=result))

    assert "HIERARCH PRO MDFC FLUX L" not in fits.getheader(path, 0)


def test_add_mdfc_fluxes_warns_when_query_fails(tmp_path, caplog):
    """A Vizier query error is logged and the remaining files are left alone."""
    path = _write_calibration_fits(
        tmp_path, "oifits_query_fail.fits", **{"ESO OBS TARG NAME": "HD5555"}
    )

    class _FailingVizier:
        def query_region(self, *_args, **_kwargs):
            raise RuntimeError("network down")

    with caplog.at_level(logging.WARNING, logger="matisse.core.lib_auto_pipeline"):
        lib_auto_pipeline.add_mdfc_fluxes([path], _FailingVizier())

    assert "MDFC query failed for HD5555" in caplog.text
    assert "HIERARCH PRO MDFC FLUX L" not in fits.getheader(path, 0)
