from __future__ import annotations

import io

# import sys
from pathlib import Path

import pytest
from astropy.io import fits
from rich.console import Console

from matisse_pipeline.core import auto_pipeline
from matisse_pipeline.core.utils import log_utils


class _DummyProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *_args, **_kwargs):
        return 0

    def advance(self, *_args, **_kwargs):
        return None


class _DummyVizier:
    def __init__(self, *_args, **_kwargs):
        pass

    def query_region(self, *_args, **_kwargs):
        return [[[1.0, 2.0, 3.0]]]


def _write_fits(path: Path, **header_values):
    path.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU()
    for key, value in header_values.items():
        hdu.header[key] = value
    hdu.writeto(path, overwrite=True)


@pytest.mark.parametrize(
    "detector_overrides",
    [
        pytest.param(
            {
                "HIERARCH ESO DET NAME": "MATISSE-LM",
                "HIERARCH ESO DET CHIP NAME": "HAWAII-2RG",
                "HIERARCH ESO DET READ CURNAME": "SCI-SLOW-SPEED",
                "HIERARCH ESO INS PIL ID": "PHOTO",
                "HIERARCH ESO INS PIN ID": "PHOTO",
                "HIERARCH ESO INS DIL NAME": "LOW",
                "HIERARCH ESO INS DIN NAME": "LOW",
            },
            id="hawaii-2rg",
        ),
        pytest.param(
            {
                "HIERARCH ESO DET NAME": "MATISSE-N",
                "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
                "HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED",
                "HIERARCH ESO INS PIL ID": "PHOTO",
                "HIERARCH ESO INS PIN ID": "PHOTO",
                "HIERARCH ESO INS DIL NAME": "LOW",
                "HIERARCH ESO INS DIN NAME": "LOW",
            },
            id="aquarius",
        ),
    ],
)
def test_run_pipeline_check_calibration_summary(
    tmp_path, monkeypatch, detector_overrides
):
    raw_dir = tmp_path / "raw"
    calib_dir = tmp_path / "calib"
    result_dir = tmp_path / "results"

    capture_stream = io.StringIO()
    test_console = Console(file=capture_stream, force_terminal=False)

    monkeypatch.setattr(auto_pipeline, "console", test_console)
    monkeypatch.setattr(log_utils, "console", test_console)

    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    original_show_calibration_status = auto_pipeline.show_calibration_status
    captured_blocks: list[auto_pipeline.RedBlock] = []

    def capture_show_calibration_status(blocks, console):
        captured_blocks.extend(blocks)
        return original_show_calibration_status(blocks, console)

    monkeypatch.setattr(
        auto_pipeline, "show_calibration_status", capture_show_calibration_status
    )
    monkeypatch.setattr(
        log_utils, "show_calibration_status", capture_show_calibration_status
    )

    raw_header = {
        "HIERARCH ESO DPR CATG": "SCIENCE",
        "HIERARCH ESO DPR TYPE": "OBJECT",
        "HIERARCH ESO DPR TECH": "INTERFEROMETRY",
        "HIERARCH ESO DPR SEQ": "SEQ",
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
        "HIERARCH ESO TPL START": "2025-01-01T00:00:00",
        "HIERARCH ESO TPL ID": "TPL1",
        "HIERARCH ESO INS DIL NAME": "LOW",
        "HIERARCH ESO INS DIN NAME": "LOW",
        "ESO OBS TARG NAME": "TARGET-STAR",
    }
    raw_header.update(detector_overrides)

    raw_path = raw_dir / "MATIS_RAW001.fits"
    _write_fits(raw_path, **raw_header)

    common = {
        key: raw_header[key]
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
            "HIERARCH ESO INS DIL NAME",
            "HIERARCH ESO INS DIN NAME",
        ]
    }

    calib_specs = [
        ("badpix.fits", "BADPIX", "2025-01-01T00:10:00"),
        ("obs_flatfield.fits", "OBS_FLATFIELD", "2025-01-01T00:12:00"),
        ("nonlinearity.fits", "NONLINEARITY", "2025-01-01T00:14:00"),
        ("shift_map.fits", "SHIFT_MAP", "2025-01-01T00:16:00"),
        ("kappa_matrix.fits", "KAPPA_MATRIX", "2025-01-01T00:18:00"),
    ]

    for filename, catg, tpl_start in calib_specs:
        _write_fits(
            calib_dir / filename,
            **{
                **common,
                "HIERARCH ESO PRO CATG": catg,
                "HIERARCH ESO TPL START": tpl_start,
            },
        )

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        check_calib=True,
        maxIter=1,
    )

    output = capture_stream.getvalue()
    assert "Calibration Summary" in output

    assert captured_blocks, "expected calibration blocks"
    block = captured_blocks[0]
    tags = {tag for _, tag in block["calib"]}
    assert {
        "BADPIX",
        "OBS_FLATFIELD",
        "NONLINEARITY",
        "SHIFT_MAP",
        "KAPPA_MATRIX",
    }.issubset(tags)
    assert block["status"] == 1


def test_run_pipeline_writes_sof_and_invokes_esorex(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    calib_dir = tmp_path / "calib"
    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    capture_stream = io.StringIO()
    test_console = Console(file=capture_stream, force_terminal=False)

    monkeypatch.setattr(auto_pipeline, "console", test_console)
    monkeypatch.setattr(log_utils, "console", test_console)

    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    captured_commands: list[str] = []
    captured_tasks: list[tuple] = []

    def fake_run_esorex(args):
        cmd, block_index, _lock = args
        captured_commands.append(cmd)

        output_dir = None
        for part in cmd.split():
            if part.startswith("--output-dir="):
                output_dir = Path(part.split("=", 1)[1])
                break

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            oifits_path = output_dir / "TARGET_RAW_INT_001.fits"
            hdu = fits.PrimaryHDU()
            hdu.header["ESO OBS TARG NAME"] = "CAL FILE"
            hdu.writeto(oifits_path, overwrite=True)

        return block_index, True

    class _LocalManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def Lock(self):
            class _LocalLock:
                pass

            return _LocalLock()

    class _LocalPool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            captured_tasks.extend(iterable)
            return [func(item) for item in iterable]

    monkeypatch.setattr(auto_pipeline, "run_esorex", fake_run_esorex)
    monkeypatch.setattr(auto_pipeline, "Manager", lambda: _LocalManager())
    monkeypatch.setattr(auto_pipeline, "Pool", _LocalPool)

    raw_header = {
        "HIERARCH ESO DPR CATG": "CALIB",
        "HIERARCH ESO DPR TYPE": "DARK,IMB",
        "HIERARCH ESO DPR TECH": "IMAGE",
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
        "HIERARCH ESO TPL START": "2025-01-02T00:00:00",
        "HIERARCH ESO TPL ID": "TPL-CAL",
        "HIERARCH ESO INS DIL NAME": "LOW",
        "ESO OBS TARG NAME": "Altair",
    }

    raw_path = raw_dir / "MATIS_RAW_INT_CAL001.fits"
    _write_fits(raw_path, **raw_header)

    calib_header = {
        key: raw_header[key]
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

    _write_fits(
        calib_dir / "badpix.fits",
        **{
            **calib_header,
            "HIERARCH ESO PRO CATG": "BADPIX",
            "HIERARCH ESO TPL START": "2025-01-02T00:05:00",
        },
    )

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        nbCore=1,
        maxIter=1,
        check_calib=False,
    )

    iter_dir = result_dir / "Iter1"
    sof_path = iter_dir / "mat_im_basic.2025-01-02T00_00_00.HAWAII-2RG.sof"
    output_dir = iter_dir / "mat_im_basic.2025-01-02T00_00_00.HAWAII-2RG.rb"

    assert sof_path.exists(), "expected sof file to be created"
    assert output_dir.is_dir(), "expected output directory to be created"
    assert captured_commands, "expected esorex command to be scheduled"
    assert captured_tasks, "pool should receive tasks"

    oifits_path = output_dir / "TARGET_RAW_INT_001.fits"
    assert oifits_path.exists(), "expected synthesized OIFITS file"
    with fits.open(oifits_path) as hdul:
        header = hdul[0].header
        assert header["HIERARCH PRO MDFC FLUX L"] == 1.0
        assert header["HIERARCH PRO MDFC FLUX M"] == 2.0
        assert header["HIERARCH PRO MDFC FLUX N"] == 3.0


def test_run_pipeline_exits_when_no_raw(monkeypatch):
    def fake_resolve_raw_input(_path: str):
        raise FileNotFoundError("missing raw data")

    monkeypatch.setattr(auto_pipeline, "resolve_raw_input", fake_resolve_raw_input)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    with pytest.raises(SystemExit) as excinfo:
        auto_pipeline.run_pipeline(dirRaw="/does/not/exist")

    assert excinfo.value.code == 1


def test_run_pipeline_uses_previous_iteration_outputs(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    calib_dir = tmp_path / "calib"
    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    capture_stream = io.StringIO()
    test_console = Console(file=capture_stream, force_terminal=False)

    monkeypatch.setattr(auto_pipeline, "console", test_console)
    monkeypatch.setattr(log_utils, "console", test_console)

    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    class _DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyManager:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def Lock(self):
            return _DummyLock()

    class _DummyPool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, tasks):
            return [func(task) for task in tasks]

    monkeypatch.setattr(auto_pipeline, "Manager", lambda: _DummyManager())
    monkeypatch.setattr(auto_pipeline, "Pool", _DummyPool)

    def fake_run_esorex(args):
        cmd, block_index, _lock = args
        output_dir = None
        for part in cmd.split():
            if part.startswith("--output-dir="):
                output_dir = Path(part.split("=", 1)[1])
                break

        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        oifits_path = output_dir / f"TARGET_RAW_INT_{block_index:03d}.fits"
        hdu = fits.PrimaryHDU()
        hdu.header["ESO OBS TARG NAME"] = "TARGET-STAR"
        hdu.writeto(oifits_path, overwrite=True)

        return block_index, True

    monkeypatch.setattr(auto_pipeline, "run_esorex", fake_run_esorex)

    raw_header = {
        "HIERARCH ESO DPR CATG": "SCIENCE",
        "HIERARCH ESO DPR TYPE": "OBJECT",
        "HIERARCH ESO DPR TECH": "INTERFEROMETRY",
        "HIERARCH ESO DPR SEQ": "SEQ",
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
        "HIERARCH ESO TPL START": "2025-01-01T00:00:00",
        "HIERARCH ESO TPL ID": "TPL1",
        "HIERARCH ESO INS DIL NAME": "LOW",
        "HIERARCH ESO INS DIN NAME": "LOW",
        "ESO OBS TARG NAME": "TARGET-STAR",
    }

    raw_path = raw_dir / "MATIS_RAW001.fits"
    _write_fits(raw_path, **raw_header)

    common = {
        key: raw_header[key]
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
            "HIERARCH ESO INS DIL NAME",
            "HIERARCH ESO INS DIN NAME",
        ]
    }

    calib_specs = [
        ("badpix.fits", "BADPIX", "2025-01-01T00:10:00"),
        ("obs_flatfield.fits", "OBS_FLATFIELD", "2025-01-01T00:12:00"),
        ("nonlinearity.fits", "NONLINEARITY", "2025-01-01T00:14:00"),
        ("shift_map.fits", "SHIFT_MAP", "2025-01-01T00:16:00"),
        ("kappa_matrix.fits", "KAPPA_MATRIX", "2025-01-01T00:18:00"),
    ]

    for filename, catg, tpl_start in calib_specs:
        _write_fits(
            calib_dir / filename,
            **{
                **common,
                "HIERARCH ESO PRO CATG": catg,
                "HIERARCH ESO TPL START": tpl_start,
            },
        )

    captured_sources: list[list[str]] = []
    original_matisse_calib = auto_pipeline.matisse_calib

    def spy_matisse_calib(header, action, list_calib_file, calib_previous, tplstart):
        captured_sources.append(list(list_calib_file))
        return original_matisse_calib(
            header, action, list_calib_file, calib_previous, tplstart
        )

    monkeypatch.setattr(auto_pipeline, "matisse_calib", spy_matisse_calib)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        maxIter=2,
        overwrite=1,
    )

    assert any(
        any("Iter1" in path for path in sources) for sources in captured_sources
    ), "expected previous iteration files to be reused as calibrations"
