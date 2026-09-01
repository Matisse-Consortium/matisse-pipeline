"""Tests for the esorex recipe parameter compatibility layer."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from matisse.core import recipe_compat


def _fake_run(stdout="", stderr="", exc=None, argv=None):
    """Build a stand-in for subprocess.run recording argv and returning stdout/stderr."""

    def _run(cmd, **kwargs):
        if argv is not None:
            argv.extend(cmd)
        if exc is not None:
            raise exc
        return SimpleNamespace(stdout=stdout, stderr=stderr)

    return _run


@pytest.mark.parametrize(
    ("stdout", "stderr", "expected"),
    [
        ("  --vfactor <bool>\n  --pfactor <bool>\n", "", True),
        ("  --vfactor <bool>\n", "  --pfactor <bool>\n", True),
        ("  --nbimages <int>\n", "", False),
        ("  --vfactor <bool>\n", "", False),
        ("  --pfactor <bool>\n", "", False),
    ],
)
def test_detect_has_filtering_params_parses_man_page(
    monkeypatch, stdout, stderr, expected
):
    """Detection is True only when both --vfactor and --pfactor appear in the man page."""
    monkeypatch.setattr(
        recipe_compat.subprocess, "run", _fake_run(stdout=stdout, stderr=stderr)
    )

    assert recipe_compat.detect_has_filtering_params() is expected


def test_detect_has_filtering_params_passes_recipe_dir(monkeypatch, tmp_path):
    """A custom recipe_dir is forwarded to esorex as --recipe-dir=<path>."""
    argv = []
    monkeypatch.setattr(
        recipe_compat.subprocess, "run", _fake_run(stdout="", argv=argv)
    )

    recipe_compat.detect_has_filtering_params(recipe_dir=tmp_path)

    assert argv == [
        "esorex",
        f"--recipe-dir={tmp_path}",
        "--man-page",
        "mat_raw_estimates",
    ]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("esorex not installed"),
        subprocess.TimeoutExpired(cmd=["esorex"], timeout=10),
    ],
)
def test_detect_has_filtering_params_returns_false_on_failure(monkeypatch, error):
    """A missing esorex or a timeout is reported as 'no filtering support'."""
    monkeypatch.setattr(recipe_compat.subprocess, "run", _fake_run(exc=error))

    assert (
        recipe_compat.detect_has_filtering_params(recipe_dir=Path("/nowhere")) is False
    )


@pytest.mark.parametrize(
    ("vfactor_mode", "pfactor_mode", "expected"),
    [
        (True, True, "--vfactor=TRUE --pfactor=TRUE --filter=vf,pf,jp"),
        (True, False, "--vfactor=TRUE --pfactor=FALSE --filter=vf,pf,jp"),
        (False, True, "--vfactor=FALSE --pfactor=TRUE --filter=vf,pf,jp"),
        (False, False, "--vfactor=FALSE --pfactor=FALSE --filter=vf,pf,jp"),
    ],
)
def test_build_raw_estimates_params_modern_recipe(
    monkeypatch, vfactor_mode, pfactor_mode, expected
):
    """A 2.0.1+ recipe gets explicit --vfactor/--pfactor/--filter tokens."""
    monkeypatch.setattr(recipe_compat, "detect_has_filtering_params", lambda _dir: True)

    params = recipe_compat.build_raw_estimates_params(
        vfactor_mode=vfactor_mode, pfactor_mode=pfactor_mode
    )

    assert params == expected


def test_build_raw_estimates_params_with_filter_baseline(monkeypatch):
    """filter_baseline adds a --filterBaseline token after the --filter token."""
    monkeypatch.setattr(recipe_compat, "detect_has_filtering_params", lambda _dir: True)

    params = recipe_compat.build_raw_estimates_params(
        filter_mode="vf", filter_baseline=3
    )

    assert params == "--vfactor=TRUE --pfactor=TRUE --filter=vf --filterBaseline=3"


def test_build_raw_estimates_params_legacy_recipe_ignores_options(monkeypatch):
    """An older recipe yields no parameters at all, even with non-default options."""
    monkeypatch.setattr(
        recipe_compat, "detect_has_filtering_params", lambda _dir: False
    )

    params = recipe_compat.build_raw_estimates_params(
        vfactor_mode=False,
        pfactor_mode=True,
        filter_mode="vf",
        filter_baseline=2,
    )

    assert params == ""
