"""Tests for the `matisse diagnostic` command and its core helpers."""

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from matisse.cli.main import app
from matisse.core.diagnostic import (
    extract_timeseries,
    load_night,
    make_transfer_function_plot,
    tf_statistics,
)

DATA = Path(__file__).parent / "data"
runner = CliRunner()


@pytest.fixture
def night():
    return load_night(DATA / "test_dir_flux")


def test_load_night_filters_raw_int(night):
    catg = {d.header["HIERARCH ESO PRO CATG"] for d in night}
    assert catg <= {"CALIB_RAW_INT", "TARGET_RAW_INT"}
    assert len(night) == 4


def test_load_night_not_recursive():
    # tests/data holds sub-directories: only top-level files are read
    night = load_night(DATA)
    assert all(d.file_path.parent == DATA for d in night)
    assert not any("LAMP" in d.file_path.name for d in night)


def test_extract_meta_and_chop_filter(night):
    df = extract_timeseries(night, "VIS2", "LM")
    assert {"chop", "tpl", "config"} <= set(df.columns)
    assert set(df["chop"]) <= {"Chop", "noChop"}
    n_nochop = (df["chop"] == "noChop").sum()
    assert len(extract_timeseries(night, "VIS2", "LM", chop="nochop")) == n_nochop
    assert (
        len(extract_timeseries(night, "VIS2", "LM", chop="chop")) == len(df) - n_nochop
    )


def test_extract_tf2(night):
    df = extract_timeseries(night, "TF2", "LM")
    assert len(df) == 6  # one CAL file, 6 baselines
    assert set(df["category"]) == {"CAL"}
    assert df["baseline"].str.match(r"^U\d-U\d$").all()
    assert np.isfinite(df["value"]).all()
    assert df["time"].notna().all()


def test_extract_t3_and_vis2(night):
    t3 = extract_timeseries(night, "T3", "LM")
    assert t3["baseline"].nunique() == 4
    vis2 = extract_timeseries(night, "VIS2", "LM")
    assert set(vis2["category"]) == {"CAL", "SCI"}


def test_extract_unknown_table(night):
    with pytest.raises(ValueError):
        extract_timeseries(night, "FOO", "LM")


def test_extract_empty_window(night):
    df = extract_timeseries(night, "VIS2", "LM", wl_range=(20.0, 21.0))
    assert df.empty


def test_bcd_share_baseline_label():
    night = load_night(DATA / "test_dir_bcd")
    df = extract_timeseries(night, "TF2", "LM")
    assert df["bcd"].nunique() == 4
    # Swapped station order with BCD must map onto the same 6 labels
    assert df["baseline"].nunique() == 6


def test_tf_statistics():
    night = load_night(DATA / "test_dir_bcd")
    stats = tf_statistics(extract_timeseries(night, "TF2", "LM"))
    assert len(stats) == 6
    assert (stats["n"] == 4).all()
    # one point per (baseline, BCD) -> normalised scatter is zero
    assert np.allclose(stats["scatter_pct"], 0)


def test_make_tf_plot(night):
    df_tf = extract_timeseries(night, "TF2", "LM")
    df_vis2 = extract_timeseries(night, "VIS2", "LM")
    df_t3 = extract_timeseries(night, "T3", "LM")
    fig = make_transfer_function_plot(df_tf, df_vis2, df_t3, show_vis=True)
    assert len(fig.data) > 0
    assert "LM" in fig.layout.title.text


def test_cli_requires_a_diagnostic():
    result = runner.invoke(app, ["diagnostic", str(DATA / "test_dir_flux")])
    assert result.exit_code == 1


def test_cli_invalid_band():
    result = runner.invoke(
        app, ["diagnostic", str(DATA / "test_dir_flux"), "--tf", "-b", "K"]
    )
    assert result.exit_code == 1


def test_cli_invalid_chop():
    result = runner.invoke(
        app, ["diagnostic", str(DATA / "test_dir_flux"), "--tf", "--chop", "x"]
    )
    assert result.exit_code == 1


def test_cli_empty_dir(tmp_path):
    result = runner.invoke(app, ["diagnostic", str(tmp_path), "--tf"])
    assert result.exit_code == 1


def test_cli_tf_html(tmp_path):
    out = tmp_path / "tf.html"
    result = runner.invoke(
        app,
        [
            "diagnostic",
            str(DATA / "test_dir_flux"),
            "--tf",
            "--chop",
            "nochop",
            "--save",
            str(out),
            "--no-open",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "tf_LM.html").exists()
    assert (tmp_path / "tf_N.html").exists()
