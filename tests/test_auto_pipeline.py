from __future__ import annotations

import io

# import sys
from pathlib import Path

import pytest
from astropy.io import fits
from rich.console import Console

from matisse.core import auto_pipeline
from matisse.core.lib_auto_pipeline import add_mdfc_fluxes
from matisse.core.utils import log_utils


class _DummyProgress:
    def __init__(self, *args, **kwargs):
        self.console = type("_C", (), {"print": staticmethod(lambda *a, **k: None)})()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *_args, **_kwargs):
        return 0

    def advance(self, *_args, **_kwargs):
        return None

    @staticmethod
    def get_default_columns():
        return ()


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

    captured: dict[str, object] = {}

    def fake_subprocess_run(
        cmd_args, *, cwd=None, stdout=None, stderr=None, check=False
    ):
        captured["args"] = cmd_args
        captured["cwd"] = cwd

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_subprocess_run)

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    job_path = tmp_path / "esorex"

    base_cmd = f"esorex --working-dir={workdir} {job_path}"
    original_cmd = f"{base_cmd} % simulated progress output"
    block_index = 4

    result = auto_pipeline.run_esorex((original_cmd, block_index, "mat_test_recipe"))

    assert result == (
        block_index,
        True,
        "mat_test_recipe",
        str(workdir / "esorex.log"),
    )

    assert captured["args"] == ["esorex", f"--working-dir={workdir}", str(job_path)]
    assert captured["cwd"] == str(workdir)


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
    iter_dir = result_dir / "reduced"
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

    assert logfile.exists()  # check mode is purely informational, no filesystem changes
    assert run_called is False


def test_run_pipeline_reports_expected_oifits_with_rerun(monkeypatch, tmp_path):
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
    rbname_safe = f"mat_raw_estimates.{tplstart}.HAWAII-2RG".replace(":", "_")
    iter_dir = result_dir / "reduced"
    iter_dir.mkdir()
    output_dir = iter_dir / f"{rbname_safe}.rb"
    output_dir.mkdir()
    # Simulate an already processed block.
    (output_dir / "TARGET_RAW_INT_001.fits").write_text("existing")

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
        auto_pipeline, "add_mdfc_fluxes", lambda *_args, **_kwargs: None
    )
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
    monkeypatch.setattr(auto_pipeline, "matisse_type", lambda _hdr: "TAG")
    monkeypatch.setattr(
        auto_pipeline,
        "matisse_action",
        lambda *_args, **_kwargs: "ACTION_MAT_RAW_ESTIMATES",
    )
    monkeypatch.setattr(
        auto_pipeline,
        "matisse_recipes",
        lambda *_args, **_kwargs: ("mat_raw_estimates", "params"),
    )
    monkeypatch.setattr(
        auto_pipeline, "matisse_calib", lambda *_args, **_kwargs: ([], 1)
    )

    summary = auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        skipN=True,
        check_blocks=False,
    )

    assert summary["n_action_raw_estimates_final"] == 1
    assert summary["expected_oifits"] == 6


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

    def capture_show_calibration_status(blocks, console, **kwargs):
        captured_blocks.extend(blocks)
        return original_show_calibration_status(blocks, console, **kwargs)

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
        cmd, block_index, _recipe = args
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

        return block_index, True, _recipe, "dummy.err"

    class _LocalPool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, iterable):
            items = list(iterable)
            captured_tasks.extend(items)
            return [func(item) for item in items]

    monkeypatch.setattr(auto_pipeline, "run_esorex", fake_run_esorex)
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
        check_calib=False,
    )

    iter_dir = result_dir / "reduced"
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

    class _DummyPool:
        def __init__(self, processes):
            self.processes = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def imap_unordered(self, func, tasks):
            return [func(task) for task in tasks]

    monkeypatch.setattr(auto_pipeline, "Pool", _DummyPool)

    def fake_run_esorex(args):
        cmd, block_index, _recipe = args
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

        return block_index, True, _recipe, "dummy.err"

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
        overwrite=1,
    )

    assert any(
        any("reduced" in path for path in sources) for sources in captured_sources
    ), "expected previous iteration files to be reused as calibrations"


# =====================================================================
# Tests for add_mdfc_fluxes (extracted from auto_pipeline into lib)
# =====================================================================


def test_add_mdfc_fluxes_writes_header_keywords(tmp_path):
    """add_mdfc_fluxes should write L/M/N flux keywords into FITS headers."""
    f1 = tmp_path / "target1.fits"
    _write_fits(f1, **{"ESO OBS TARG NAME": "Vega"})

    vizier = _DummyVizier()
    add_mdfc_fluxes([f1], vizier)

    with fits.open(f1) as hdul:
        hdr = hdul[0].header
        assert hdr["HIERARCH PRO MDFC FLUX L"] == 1.0
        assert hdr["HIERARCH PRO MDFC FLUX M"] == 2.0
        assert hdr["HIERARCH PRO MDFC FLUX N"] == 3.0


def test_add_mdfc_fluxes_groups_by_target(tmp_path):
    """Files with the same target should trigger only one Vizier query."""
    f1 = tmp_path / "a.fits"
    f2 = tmp_path / "b.fits"
    _write_fits(f1, **{"ESO OBS TARG NAME": "Sirius"})
    _write_fits(f2, **{"ESO OBS TARG NAME": "Sirius"})

    query_count = 0

    class _CountingVizier:
        def query_region(self, *_args, **_kwargs):
            nonlocal query_count
            query_count += 1
            return [[[10.0, 20.0, 30.0]]]

    add_mdfc_fluxes([f1, f2], _CountingVizier())

    assert query_count == 1, "expected a single Vizier query for the same target"
    for path in [f1, f2]:
        with fits.open(path) as hdul:
            assert hdul[0].header["HIERARCH PRO MDFC FLUX L"] == 10.0


def test_add_mdfc_fluxes_skips_files_without_target(tmp_path):
    """Files without ESO OBS TARG NAME should be skipped gracefully."""
    f1 = tmp_path / "no_target.fits"
    _write_fits(f1)  # no target keyword

    vizier = _DummyVizier()
    add_mdfc_fluxes([f1], vizier)

    with fits.open(f1) as hdul:
        assert "HIERARCH PRO MDFC FLUX L" not in hdul[0].header


def test_add_mdfc_fluxes_handles_vizier_failure(tmp_path):
    """When Vizier raises an exception, the function should not crash."""
    f1 = tmp_path / "target.fits"
    _write_fits(f1, **{"ESO OBS TARG NAME": "Unknown"})

    class _FailingVizier:
        def query_region(self, *_args, **_kwargs):
            raise ConnectionError("network error")

    add_mdfc_fluxes([f1], _FailingVizier())

    with fits.open(f1) as hdul:
        assert "HIERARCH PRO MDFC FLUX L" not in hdul[0].header


def test_add_mdfc_fluxes_handles_empty_result(tmp_path):
    """When Vizier returns no results, files should be left unchanged."""
    f1 = tmp_path / "target.fits"
    _write_fits(f1, **{"ESO OBS TARG NAME": "NoMatch"})

    class _EmptyVizier:
        def query_region(self, *_args, **_kwargs):
            return None

    add_mdfc_fluxes([f1], _EmptyVizier())

    with fits.open(f1) as hdul:
        assert "HIERARCH PRO MDFC FLUX L" not in hdul[0].header


# =====================================================================
# Test: run_pipeline handles both bands in a single call
# =====================================================================


def test_run_pipeline_processes_both_bands_in_single_call(tmp_path, monkeypatch):
    """run_pipeline should handle both LM and N band files when skipL=False, skipN=False."""
    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

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

    lm_path = raw_dir / "LM.fits"
    n_path = raw_dir / "N.fits"
    lm_path.parent.mkdir(parents=True, exist_ok=True)
    lm_path.write_text("")
    n_path.write_text("")

    def _header(detector: str):
        hdr = fits.Header()
        hdr["HIERARCH ESO DPR TYPE"] = "OBJECT"
        hdr["HIERARCH ESO DPR TECH"] = "INTERFEROMETRY"
        hdr["HIERARCH ESO DPR CATG"] = "SCIENCE"
        hdr["HIERARCH ESO DET CHIP NAME"] = detector
        hdr["HIERARCH ESO INS DIL NAME"] = "LOW"
        hdr["HIERARCH ESO INS DIN NAME"] = "LOW"
        hdr["HIERARCH ESO TPL ID"] = "TPL1"
        hdr["HIERARCH ESO TPL START"] = "2025-01-01T00:00:00"
        hdr["TEST_DETECTOR"] = detector
        return hdr

    headers = {
        str(lm_path): _header("HAWAII-2RG"),
        str(n_path): _header("AQUARIUS"),
    }

    selected_detectors: list[str] = []

    def fake_resolve_raw_input(_path):
        return [str(lm_path), str(n_path)], "manual"

    def fake_getheader(path, _index):
        return headers[path]

    def fake_type(hdr):
        selected_detectors.append(hdr["TEST_DETECTOR"])
        return "TAG"

    def stop_action(*_args, **_kwargs):
        raise _StopPipeline()

    monkeypatch.setattr(auto_pipeline, "resolve_raw_input", fake_resolve_raw_input)
    monkeypatch.setattr(auto_pipeline, "getheader", fake_getheader)
    monkeypatch.setattr(auto_pipeline, "matisse_type", fake_type)
    monkeypatch.setattr(auto_pipeline, "matisse_action", stop_action)
    monkeypatch.setattr(auto_pipeline, "matisse_recipes", lambda *a, **k: ("r", "p"))

    with pytest.raises(_StopPipeline):
        auto_pipeline.run_pipeline(
            dirRaw=str(raw_dir),
            dirResult=str(result_dir),
            skipL=False,
            skipN=False,
        )

    # Both detectors should have been processed in the same call
    assert "HAWAII-2RG" in selected_detectors
    assert "AQUARIUS" in selected_detectors


# =====================================================================
# Additional coverage for run_esorex / run_pipeline internals
# =====================================================================


class _SerialPool:
    """multiprocessing.Pool stand-in running every task in the current process."""

    def __init__(self, processes):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap_unordered(self, func, tasks):
        return [func(task) for task in tasks]


class _EchoProgress(_DummyProgress):
    """Progress stub whose console is the pipeline console, so prints are captured."""

    def __init__(self, *args, **kwargs):
        self.console = kwargs.get("console") or auto_pipeline.console


_RAW_HEADER_BASE = {
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

_AQUARIUS_OVERRIDES = {
    "HIERARCH ESO DET NAME": "MATISSE-N",
    "HIERARCH ESO DET CHIP NAME": "AQUARIUS",
    "HIERARCH ESO DET READ CURNAME": "SCI-FAST-SPEED",
}

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
    "HIERARCH ESO INS DIL NAME",
    "HIERARCH ESO INS DIN NAME",
)

_CALIB_SPECS = (
    ("badpix.fits", "BADPIX", "2025-01-01T00:10:00"),
    ("obs_flatfield.fits", "OBS_FLATFIELD", "2025-01-01T00:12:00"),
    ("nonlinearity.fits", "NONLINEARITY", "2025-01-01T00:14:00"),
    ("shift_map.fits", "SHIFT_MAP", "2025-01-01T00:16:00"),
    ("kappa_matrix.fits", "KAPPA_MATRIX", "2025-01-01T00:18:00"),
)


def _no_esorex(monkeypatch, spectral_param="spectralAverage", filter_params=""):
    """Replace the two esorex probes called at the top of run_pipeline."""
    monkeypatch.setattr(
        auto_pipeline, "probe_spectral_param_name", lambda **_kwargs: spectral_param
    )
    monkeypatch.setattr(
        auto_pipeline, "build_raw_estimates_params", lambda **_kwargs: filter_params
    )


def _capture_console(monkeypatch):
    """Route every pipeline print into a StringIO-backed rich console."""
    stream = io.StringIO()
    test_console = Console(file=stream, force_terminal=False, width=220)
    monkeypatch.setattr(auto_pipeline, "console", test_console)
    monkeypatch.setattr(log_utils, "console", test_console)
    return stream


def _make_dataset(tmp_path, calib=True, **raw_overrides):
    """Write one raw MATISSE block plus its matching calibration archive."""
    raw_dir = tmp_path / "raw"
    calib_dir = tmp_path / "calib"
    result_dir = tmp_path / "results"
    header = {**_RAW_HEADER_BASE, **raw_overrides}
    _write_fits(raw_dir / "MATIS_RAW001.fits", **header)
    calib_dir.mkdir(parents=True, exist_ok=True)
    if calib:
        common = {key: header[key] for key in _CALIB_MATCH_KEYS}
        for filename, catg, tpl_start in _CALIB_SPECS:
            _write_fits(
                calib_dir / filename,
                **{
                    **common,
                    "HIERARCH ESO PRO CATG": catg,
                    "HIERARCH ESO TPL START": tpl_start,
                },
            )
    return raw_dir, calib_dir, result_dir


def _recording_esorex(commands):
    """Build a run_esorex stub recording commands and creating the output dir."""

    def fake_run_esorex(args):
        cmd, block_index, recipe = args
        commands.append(cmd)
        for part in cmd.split():
            if part.startswith("--output-dir="):
                Path(part.split("=", 1)[1]).mkdir(parents=True, exist_ok=True)
        return block_index, True, recipe, "dummy.err"

    return fake_run_esorex


def test_run_esorex_defaults_workdir_to_current_directory(monkeypatch, tmp_path):
    """A command without a --working-dir option runs in the current directory."""
    captured: dict[str, object] = {}

    def fake_subprocess_run(
        cmd_args, *, cwd=None, stdout=None, stderr=None, check=False
    ):
        captured["args"] = cmd_args
        captured["cwd"] = cwd

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(auto_pipeline.subprocess, "run", fake_subprocess_run)

    job_path = tmp_path / "block.sof"
    cmd = f"esorex mat_im_basic {job_path}"

    result = auto_pipeline.run_esorex((cmd, 7, "mat_im_basic"))

    assert result == (7, True, "mat_im_basic", "esorex.log")
    assert captured["cwd"] == "."
    assert captured["args"] == ["esorex", "mat_im_basic", str(job_path)]
    assert (tmp_path / "block.sof.log").exists()
    assert (tmp_path / "block.sof.err").exists()


def test_run_pipeline_inventory_returns_early(monkeypatch, tmp_path):
    """inventory=True prints the file inventory and returns an empty summary."""
    _no_esorex(monkeypatch)
    stream = _capture_console(monkeypatch)

    raw_dir = tmp_path / "raw"
    _write_fits(raw_dir / "MATIS_RAW001.fits", **_RAW_HEADER_BASE)
    _write_fits(
        raw_dir / "MATIS_RAW002.fits",
        **{**_RAW_HEADER_BASE, **_AQUARIUS_OVERRIDES},
    )

    summary = auto_pipeline.run_pipeline(dirRaw=str(raw_dir), inventory=True)

    assert summary == {"n_action_raw_estimates_final": None, "expected_oifits": None}
    output = stream.getvalue()
    assert "MATISSE files inventory" in output
    assert "MATIS_RAW001.fits" in output
    assert "MATIS_RAW002.fits" in output


@pytest.mark.parametrize(
    ("spectral_param", "expected_message"),
    [
        ("spectralBinning", "Detected legacy recipe parameter"),
        ("spectralAverage", "Using current recipe parameter"),
    ],
)
def test_run_pipeline_reports_spectral_parameter_flavour(
    monkeypatch, caplog, spectral_param, expected_message
):
    """The detected recipe parameter name drives the legacy/current log message."""
    import logging

    _no_esorex(monkeypatch, spectral_param=spectral_param)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    def stop(_path):
        raise _StopPipeline()

    monkeypatch.setattr(auto_pipeline, "resolve_raw_input", stop)

    with caplog.at_level(logging.DEBUG, logger="matisse"), pytest.raises(_StopPipeline):
        auto_pipeline.run_pipeline(dirRaw="/does/not/matter")

    assert any(expected_message in record.message for record in caplog.records)


def test_run_pipeline_missing_calib_directory_leaves_blocks_unready(
    monkeypatch, caplog, tmp_path
):
    """A non-existent dirCalib yields an empty archive and no ready block."""
    import logging

    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, _calib_dir, result_dir = _make_dataset(tmp_path, calib=False)
    result_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.INFO, logger="matisse"):
        summary = auto_pipeline.run_pipeline(
            dirRaw=str(raw_dir),
            dirCalib=str(tmp_path / "no_such_calib_dir"),
            dirResult=str(result_dir),
        )

    assert summary == {"n_action_raw_estimates_final": 0, "expected_oifits": 0}
    assert any(
        "Block not ready to be processed" in record.message for record in caplog.records
    )
    assert list((result_dir / "reduced").glob("*.sof")) == []


def test_run_pipeline_reports_unreadable_raw_file(monkeypatch, caplog, tmp_path):
    """A raw file that is not valid FITS is reported and excluded from the run."""
    import logging

    _no_esorex(monkeypatch)
    stream = _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir = tmp_path / "raw"
    calib_dir = tmp_path / "calib"
    result_dir = tmp_path / "results"
    calib_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_fits(raw_dir / "MATIS_ok.fits", **_RAW_HEADER_BASE)
    (raw_dir / "MATIS_bad.fits").write_text("this is definitely not a FITS file")

    with caplog.at_level(logging.INFO, logger="matisse"):
        auto_pipeline.run_pipeline(
            dirRaw=str(raw_dir),
            dirCalib=str(calib_dir),
            dirResult=str(result_dir),
        )

    assert "1 files failed to read headers" in stream.getvalue()
    messages = [record.message for record in caplog.records]
    assert any("Header read failures: ['MATIS_bad.fits']" in msg for msg in messages)
    assert any("Successfully read 1 FITS headers." in msg for msg in messages)


def test_run_pipeline_skips_files_with_other_resolution(monkeypatch, caplog, tmp_path):
    """resol='MED' discards both LM and N files recorded in LOW resolution."""
    import logging

    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"  # deliberately not created yet
    _write_fits(raw_dir / "MATIS_lm.fits", **_RAW_HEADER_BASE)
    _write_fits(raw_dir / "MATIS_n.fits", **{**_RAW_HEADER_BASE, **_AQUARIUS_OVERRIDES})

    with caplog.at_level(logging.INFO, logger="matisse"):
        summary = auto_pipeline.run_pipeline(
            dirRaw=str(raw_dir), dirResult=str(result_dir), resol="MED"
        )

    assert summary == {"n_action_raw_estimates_final": 0, "expected_oifits": 0}
    assert any(
        "Discovered 0 unique (TPL_START, DETECTOR) combinations." in record.message
        for record in caplog.records
    )
    assert (result_dir / "reduced").is_dir()


def test_run_pipeline_warns_about_both_skipped_bands(monkeypatch, caplog, tmp_path):
    """--skipL and --skipN each report how many files of their band were dropped."""
    import logging

    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir = tmp_path / "raw"
    result_dir = tmp_path / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_fits(raw_dir / "MATIS_lm.fits", **_RAW_HEADER_BASE)
    _write_fits(raw_dir / "MATIS_n.fits", **{**_RAW_HEADER_BASE, **_AQUARIUS_OVERRIDES})

    with caplog.at_level(logging.INFO, logger="matisse"):
        auto_pipeline.run_pipeline(
            dirRaw=str(raw_dir),
            dirResult=str(result_dir),
            skipL=True,
            skipN=True,
        )

    messages = [record.message for record in caplog.records]
    assert "Skipped 1 LM-band files (--skipL)." in messages
    assert "Skipped 1 N-band files (--skipN)." in messages


def test_run_pipeline_keeps_high_resolution_ordering_key(monkeypatch, tmp_path):
    """A HIGH-resolution LM block still reaches esorex with its own resolution tag."""
    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, calib_dir, result_dir = _make_dataset(
        tmp_path, **{"HIERARCH ESO INS DIL NAME": "HIGH"}
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    monkeypatch.setattr(auto_pipeline, "run_esorex", _recording_esorex(commands))
    monkeypatch.setattr(auto_pipeline, "Pool", _SerialPool)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
    )

    assert commands, "expected one esorex command"
    assert commands[0].endswith("%HIGH")


def test_run_pipeline_attaches_gra4mat_rmnrec_file(monkeypatch, caplog, tmp_path):
    """A GRAVITY RMNREC file is skipped as raw input but attached to its block."""
    import logging

    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, calib_dir, result_dir = _make_dataset(tmp_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_fits(
        raw_dir / "MATIS_rmnrec.fits",
        **{
            **_RAW_HEADER_BASE,
            "HIERARCH ESO DPR TYPE": "RMNREC",
            "HIERARCH ESO DEL FT SENSOR": "GRAVITY",
        },
    )

    commands: list[str] = []
    monkeypatch.setattr(auto_pipeline, "run_esorex", _recording_esorex(commands))
    monkeypatch.setattr(auto_pipeline, "Pool", _SerialPool)

    with caplog.at_level(logging.INFO, logger="matisse"):
        auto_pipeline.run_pipeline(
            dirRaw=str(raw_dir),
            dirCalib=str(calib_dir),
            dirResult=str(result_dir),
        )

    messages = [record.message for record in caplog.records]
    assert any("MATIS_rmnrec.fits is a RMNREC file!" in msg for msg in messages)
    assert "1 files identified as GRA4MAT." in messages

    sof_files = list((result_dir / "reduced").glob("*.sof"))
    assert len(sof_files) == 1
    sof_content = sof_files[0].read_text()
    assert f"{raw_dir / 'MATIS_rmnrec.fits'} RMNREC" in sof_content


def test_run_pipeline_uses_telescop_keyword_for_recipe_options(monkeypatch, tmp_path):
    """The TELESCOP keyword selects the telescope-specific recipe options."""
    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, calib_dir, result_dir = _make_dataset(
        tmp_path, **{**_AQUARIUS_OVERRIDES, "TELESCOP": "ESO-VLTI-A1234"}
    )
    result_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    monkeypatch.setattr(auto_pipeline, "run_esorex", _recording_esorex(commands))
    monkeypatch.setattr(auto_pipeline, "Pool", _SerialPool)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
    )

    assert commands, "expected one esorex command"
    assert "--replaceTel=3" in commands[0]


@pytest.mark.parametrize(
    ("overrides", "spectral_average", "filter_params", "expected", "unexpected"),
    [
        pytest.param(
            {}, -1, "", ["--spectralAverage=5"], ["--vfactor"], id="lm-default"
        ),
        pytest.param(
            {}, 3, "", ["--spectralAverage=3"], ["--vfactor"], id="lm-explicit"
        ),
        pytest.param(
            _AQUARIUS_OVERRIDES,
            -1,
            "",
            ["--spectralAverage=7"],
            ["--vfactor"],
            id="n-default",
        ),
        pytest.param(
            _AQUARIUS_OVERRIDES,
            3,
            "",
            ["--spectralAverage=3"],
            ["--vfactor"],
            id="n-explicit",
        ),
        pytest.param(
            {},
            3,
            "--vfactor=TRUE",
            ["--spectralAverage=3", "--vfactor=TRUE"],
            [],
            id="lm-filter-params",
        ),
        pytest.param(
            _AQUARIUS_OVERRIDES,
            3,
            "--vfactor=TRUE",
            ["--spectralAverage=3"],
            ["--vfactor"],
            id="n-filter-params-ignored",
        ),
    ],
)
def test_run_pipeline_builds_spectral_parameters(
    monkeypatch,
    tmp_path,
    overrides,
    spectral_average,
    filter_params,
    expected,
    unexpected,
):
    """spectralAverage defaults and L/M-only filtering params land in the esorex command."""
    _no_esorex(monkeypatch, filter_params=filter_params)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, calib_dir, result_dir = _make_dataset(tmp_path, **overrides)
    result_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    monkeypatch.setattr(auto_pipeline, "run_esorex", _recording_esorex(commands))
    monkeypatch.setattr(auto_pipeline, "Pool", _SerialPool)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        spectralAverage=spectral_average,
    )

    assert commands, "expected one esorex command"
    for fragment in expected:
        assert fragment in commands[0]
    for fragment in unexpected:
        assert fragment not in commands[0]


def test_run_pipeline_detailed_block_forces_calibration_check(monkeypatch, tmp_path):
    """detailed_block turns on the calibration report even when check_calib is off."""
    _no_esorex(monkeypatch)
    stream = _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, calib_dir, result_dir = _make_dataset(tmp_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    def fail_run(*_args, **_kwargs):
        raise AssertionError("esorex must not run in calibration-check mode")

    monkeypatch.setattr(auto_pipeline, "run_esorex", fail_run)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        check_calib=False,
        detailed_block=0,
    )

    output = stream.getvalue()
    assert "Calibration Summary" in output
    assert "Block #0 is out of range (available: 1-1)." in output
    assert not (result_dir / "reduced").exists()


def test_run_pipeline_passes_custom_recipe_dir(monkeypatch, tmp_path):
    """custom_recipes_dir adds --recipe-dir to the generated esorex command."""
    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, calib_dir, result_dir = _make_dataset(tmp_path)
    result_dir.mkdir(parents=True, exist_ok=True)
    recipes_dir = tmp_path / "custom_recipes"
    recipes_dir.mkdir()

    commands: list[str] = []
    monkeypatch.setattr(auto_pipeline, "run_esorex", _recording_esorex(commands))
    monkeypatch.setattr(auto_pipeline, "Pool", _SerialPool)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        custom_recipes_dir=recipes_dir,
    )

    assert commands, "expected one esorex command"
    assert f"--recipe-dir={recipes_dir}" in commands[0]


def test_run_pipeline_saves_svg_report(monkeypatch, tmp_path):
    """save_report_svg=True writes matisse_report.svg into the result directory."""
    _no_esorex(monkeypatch)
    _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _DummyProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)

    raw_dir, calib_dir, result_dir = _make_dataset(tmp_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    commands: list[str] = []
    monkeypatch.setattr(auto_pipeline, "run_esorex", _recording_esorex(commands))
    monkeypatch.setattr(auto_pipeline, "Pool", _SerialPool)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
        save_report_svg=True,
    )

    report_path = result_dir / "matisse_report.svg"
    assert report_path.exists()
    assert report_path.read_text(encoding="utf-8").lstrip().startswith("<svg")


def test_run_pipeline_reports_esorex_failures(monkeypatch, tmp_path):
    """A failing esorex block is summarised with its missing inputs and status -1."""
    _no_esorex(monkeypatch)
    stream = _capture_console(monkeypatch)
    monkeypatch.setattr(auto_pipeline, "Progress", _EchoProgress)
    monkeypatch.setattr(auto_pipeline, "Vizier", _DummyVizier)
    monkeypatch.setattr(auto_pipeline, "Pool", _SerialPool)

    raw_dir, calib_dir, result_dir = _make_dataset(tmp_path)
    result_dir.mkdir(parents=True, exist_ok=True)

    def failing_run_esorex(args):
        cmd, block_index, recipe = args
        output_dir = None
        for part in cmd.split():
            if part.startswith("--output-dir="):
                output_dir = Path(part.split("=", 1)[1])
                break
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "esorex.log"
        log_path.write_text(
            "esorex: starting recipe\nERROR: Missing input frame: BADPIX\n",
            encoding="utf-8",
        )
        return block_index, False, recipe, str(log_path)

    monkeypatch.setattr(auto_pipeline, "run_esorex", failing_run_esorex)

    final_blocks: list[list[auto_pipeline.RedBlock]] = []
    original_show_blocs_status = auto_pipeline.show_blocs_status

    def spy_show_blocs_status(list_cmd_esorex, list_red_blocks, check_blocks):
        final_blocks.append(list(list_red_blocks))
        return original_show_blocs_status(
            list_cmd_esorex, list_red_blocks, check_blocks
        )

    monkeypatch.setattr(auto_pipeline, "show_blocs_status", spy_show_blocs_status)

    auto_pipeline.run_pipeline(
        dirRaw=str(raw_dir),
        dirCalib=str(calib_dir),
        dirResult=str(result_dir),
    )

    output = stream.getvalue()
    assert "1 block(s) failed" in output
    assert "esorex.log" in output
    assert "Missing inputs detected in esorex log" in output
    assert "ERROR: Missing input frame: BADPIX" in output

    assert len(final_blocks) == 2, "expected the loop to run a second iteration"
    for blocks in final_blocks:
        assert blocks[0]["status"] == -1
        assert blocks[0]["error_msg"] == "Missing: input frame"
