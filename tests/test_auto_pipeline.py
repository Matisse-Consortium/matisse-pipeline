from __future__ import annotations

import io

# import sys
from pathlib import Path
from threading import Lock

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


def test_run_esorex_invokes_esorex_command(monkeypatch, tmp_path):
    messages: list[str] = []

    class _ConsoleStub:
        def print(self, *args, **_kwargs):
            messages.append(" ".join(str(arg) for arg in args))

    dummy_console = _ConsoleStub()
    monkeypatch.setattr(auto_pipeline, "console", dummy_console)

    captured_command: dict[str, str] = {}

    def fake_system(command: str) -> int:
        captured_command["value"] = command
        return 0

    monkeypatch.setattr(auto_pipeline.os, "system", fake_system)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    job_path = tmp_path / "mat_job"

    base_cmd = f"esorex --working-dir={workdir} {job_path}"
    original_cmd = f"{base_cmd} % simulated progress output"
    block_index = 4

    result = auto_pipeline.run_esorex((original_cmd, block_index, Lock()))

    assert result == (block_index, True)

    expected_system_call = (
        f"cd {workdir}; {base_cmd} > {job_path}.log 2> {job_path}.err"
    )

    assert captured_command["value"] == expected_system_call
    assert any(f"Block {block_index}" in message for message in messages)


class _StopPipeline(Exception):
    pass


@pytest.mark.parametrize(
    ("tplidsel", "tplstartsel", "expected"),
    [
        ("TPL-MATCH", "2025-01-01T00:00:00", ["match"]),
        ("TPL-MATCH", "", ["match"]),
        ("", "2025-01-01T00:00:00", ["match"]),
        ("", "", ["match", "other"]),
    ],
)
@pytest.mark.parametrize(
    ("detector", "skipL", "skipN"),
    [("HAWAII-2RG", False, True), ("AQUARIUS", True, False)],
)
def test_run_pipeline_filters_tpl_selection(
    monkeypatch, tmp_path, detector, skipL, skipN, tplidsel, tplstartsel, expected
):
    match_path = tmp_path / "MATCH.fits"
    other_path = tmp_path / "OTHER.fits"
    match_path.write_text("")
    other_path.write_text("")

    def _header(label: str, tplid: str, tplstart: str):
        hdr = fits.Header()
        hdr["HIERARCH ESO DPR TYPE"] = "OBJECT"
        hdr["HIERARCH ESO DET CHIP NAME"] = detector
        hdr["HIERARCH ESO INS DIL NAME"] = "LOW"
        hdr["HIERARCH ESO INS DIN NAME"] = "LOW"
        hdr["HIERARCH ESO TPL ID"] = tplid
        hdr["HIERARCH ESO TPL START"] = tplstart
        hdr["HIERARCH ESO DPR TECH"] = "INTERFEROMETRY"
        hdr["HIERARCH ESO DPR CATG"] = "SCIENCE"
        hdr["TEST_NAME"] = label
        return hdr

    headers = {
        str(match_path): _header("match", "TPL-MATCH", "2025-01-01T00:00:00"),
        str(other_path): _header("other", "TPL-OTHER", "2024-12-31T23:59:59"),
    }

    selected: list[str] = []

    class _ConsoleStub:
        def print(self, *_args, **_kwargs):
            return None

    dummy_console = _ConsoleStub()
    monkeypatch.setattr(auto_pipeline, "console", dummy_console)
    monkeypatch.setattr(log_utils, "console", dummy_console)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)
    monkeypatch.setattr(auto_pipeline, "section", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(auto_pipeline, "iteration_banner", lambda *_args: None)

    def fake_resolve_raw_input(_path: str):
        return [str(match_path), str(other_path)], "manual"

    def fake_getheader(path: str, _index: int):
        return headers[path]

    monkeypatch.setattr(auto_pipeline, "resolve_raw_input", fake_resolve_raw_input)
    monkeypatch.setattr(auto_pipeline, "getheader", fake_getheader)

    def fake_type(hdr):
        selected.append(hdr["TEST_NAME"])
        return "TAG"

    def stop_action(*_args, **_kwargs):
        raise _StopPipeline()

    monkeypatch.setattr(auto_pipeline, "matisse_type", fake_type)
    monkeypatch.setattr(auto_pipeline, "matisse_action", stop_action)

    def fake_recipes(*_args, **_kwargs):
        return "recipe", "params"

    monkeypatch.setattr(auto_pipeline, "matisse_recipes", fake_recipes)

    result_dir = tmp_path / "results"
    result_dir.mkdir()

    with pytest.raises(_StopPipeline):
        auto_pipeline.run_pipeline(
            dirRaw=str(tmp_path),
            dirResult=str(result_dir),
            skipL=skipL,
            skipN=skipN,
            tplidsel=tplidsel,
            tplstartsel=tplstartsel,
        )

    assert selected == expected


def test_run_pipeline_existing_output_dir(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    calib_dir = tmp_path / "calib"
    result_dir = tmp_path / "results"
    raw_dir.mkdir()
    calib_dir.mkdir()
    result_dir.mkdir()

    raw_path = raw_dir / "raw.fits"
    raw_path.write_text("")

    tplstart = "2025-01-01T00:00:00"
    chip = "HAWAII-2RG"
    rbname_safe = f"recipe.{tplstart}.HAWAII-2RG".replace(":", "_")
    iter_dir = result_dir / "Iter1"
    iter_dir.mkdir()
    output_dir = iter_dir / f"{rbname_safe}.rb"
    output_dir.mkdir()
    (output_dir / "dummy.txt").write_text("content")
    logfile = output_dir / ".logfile"
    logfile.write_text("old log")

    sof_path = iter_dir / f"{rbname_safe}.sof"
    sof_path.write_text("existing sof")

    class _ConsoleStub:
        def print(self, *_args, **_kwargs):
            return None

    dummy_console = _ConsoleStub()
    monkeypatch.setattr(auto_pipeline, "console", dummy_console)
    monkeypatch.setattr(log_utils, "console", dummy_console)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)
    monkeypatch.setattr(auto_pipeline, "section", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(auto_pipeline, "iteration_banner", lambda *_args: None)
    monkeypatch.setattr(auto_pipeline, "remove_double_parameter", lambda value: value)
    monkeypatch.setattr(
        auto_pipeline, "show_calibration_status", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        log_utils, "show_calibration_status", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        auto_pipeline, "show_blocs_status", lambda *_args, **_kwargs: True
    )

    def fake_resolve_raw_input(_path: str):
        return [str(raw_path)], "manual"

    def fake_getheader(_path: str, _index: int):
        hdr = fits.Header()
        hdr["HIERARCH ESO DPR TYPE"] = "OBJECT"
        hdr["HIERARCH ESO DPR TECH"] = "INTERFEROMETRY"
        hdr["HIERARCH ESO DPR CATG"] = "SCIENCE"
        hdr["HIERARCH ESO DET CHIP NAME"] = chip
        hdr["HIERARCH ESO INS DIL NAME"] = "LOW"
        hdr["HIERARCH ESO TPL ID"] = "TPL-ID"
        hdr["HIERARCH ESO TPL START"] = tplstart
        hdr["ESO OBS TARG NAME"] = "TARGET-STAR"
        hdr["HIERARCH ESO INS DIN NAME"] = "LOW"
        return hdr

    monkeypatch.setattr(auto_pipeline, "resolve_raw_input", fake_resolve_raw_input)
    monkeypatch.setattr(auto_pipeline, "getheader", fake_getheader)

    def fake_type(_hdr):
        return "TAG"

    def fake_action(*_args, **_kwargs):
        return "ACTION"

    def fake_recipes(*_args, **_kwargs):
        return "recipe", "params"

    def fake_calib(*_args, **_kwargs):
        return [], 1

    monkeypatch.setattr(auto_pipeline, "matisse_type", fake_type)
    monkeypatch.setattr(auto_pipeline, "matisse_action", fake_action)
    monkeypatch.setattr(auto_pipeline, "matisse_recipes", fake_recipes)
    monkeypatch.setattr(auto_pipeline, "matisse_calib", fake_calib)

    run_called = False

    def fail_run(*_args, **_kwargs):
        nonlocal run_called
        run_called = True
        return 0, True

    monkeypatch.setattr(auto_pipeline, "run_esorex", fail_run)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        skipN=True,
        check_blocks=True,
    )

    assert not logfile.exists()
    assert run_called is False


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
