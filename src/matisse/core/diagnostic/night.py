"""
Night-level data collection for MATISSE diagnostics.

Loads every reduced (``*_RAW_INT``) OIFITS file of a directory -- typically
the ``reduced_OIFITS/`` folder produced by ``matisse format``, with names like
``2022-07-02T041700_HD143006_K0G2D0J3_IR-LM_LOW_OUT_OUT_noChop.fits`` -- and flattens
the requested table (TF2, VIS2, T3, ...) into a tidy :class:`pandas.DataFrame`
with one row per (file, baseline/triangle, exposure). Each diagnostic plot
then consumes these DataFrames, which makes it easy to add new views
(vs. seeing, tau0, airmass, ...) without re-reading the FITS files.

Ported/rewritten from legacy ``libShowOifits.open_oi_dir`` and
``show_oi_vs_time`` (F. Millour et al.).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

from matisse.core.utils.oifits_reader import OIFitsData, OIFitsReader

logger = logging.getLogger(__name__)

#: PRO.CATG accepted by default: reduced (uncalibrated) products only.
RAW_INT_CATG = ("CALIB_RAW_INT", "TARGET_RAW_INT")

#: Default wavelength windows [µm] used to average the spectral channels.
DEFAULT_WL_RANGE: dict[str, tuple[float, float]] = {
    "LM": (3.0, 4.0),
    "N": (8.5, 10.5),
}

#: (table key in OIFitsData, data column, error column)
TABLES: dict[str, tuple[str, str, str]] = {
    "TF2": ("tf2", "TF2", "TF2ERR"),
    "VIS2": ("vis2", "VIS2", "VIS2ERR"),
    "VIS": ("vis", "VISAMP", "VISAMPERR"),
    "T3": ("t3", "CLOS", "CLOSERR"),
    "FLUX": ("flux", "FLUX", "FLUXERR"),
}

COLUMNS = [
    "mjd",
    "time",
    "value",
    "err",
    "baseline",
    "target",
    "category",
    "bcd",
    "chop",
    "tpl",
    "config",
    "band",
    "disp",
    "dit",
    "seeing",
    "tau0",
    "file",
]


def load_night(
    datadir: Path | str,
    catg: Iterable[str] | None = RAW_INT_CATG,
) -> list[OIFitsData]:
    """Read all OIFITS files of a ``reduced_OIFITS`` directory (not recursive).

    Parameters
    ----------
    datadir : Path | str
        Directory containing reduced MATISSE OIFITS files.
    catg : iterable of str or None
        Accepted ``ESO PRO CATG`` values. ``None`` accepts everything.

    Returns
    -------
    list of OIFitsData
        Sorted by observation date.
    """
    datadir = Path(datadir)
    files = sorted(
        p
        for p in datadir.iterdir()
        if p.is_file()
        and p.name.endswith((".fits", ".fits.gz"))
        and not p.name.startswith("._")
        and "LAMP" not in p.name
    )
    accepted = set(catg) if catg is not None else None

    out: list[OIFitsData] = []
    for f in files:
        data = OIFitsReader(f).read()
        if data is None:
            continue
        pro_catg = str(data.header.get("HIERARCH ESO PRO CATG", ""))
        if accepted is not None and pro_catg not in accepted:
            logger.debug(f"Skip {f.name} (PRO.CATG={pro_catg or 'none'})")
            continue
        out.append(data)

    out.sort(key=lambda d: d.date_obs)
    logger.debug(f"{len(out)} OIFITS files loaded from {datadir}")
    return out


def _file_meta(d: OIFitsData) -> dict[str, str]:
    """Observation-block metadata, same keywords as ``tidyup`` file names."""
    hdr = d.header
    stations = "".join(
        str(hdr.get(f"HIERARCH ESO ISS CONF STATION{i}", "")) for i in range(1, 5)
    )
    return {
        "chop": "Chop" if hdr.get("HIERARCH ESO ISS CHOP ST", "F") == "T" else "noChop",
        "tpl": str(hdr.get("HIERARCH ESO TPL START", "")),
        "config": stations or "noConf",
    }


def _canonical_label(sta_idx: np.ndarray, ref: dict[int, str]) -> str:
    """Station-name label independent of BCD ordering (sorted by name)."""
    names = sorted(ref.get(int(i), str(int(i))) for i in sta_idx)
    return "-".join(names)


def _robust_mean(arr: np.ndarray) -> float:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def extract_timeseries(
    night: list[OIFitsData],
    table: str,
    band: str,
    wl_range: tuple[float, float] | None = None,
    chop: str = "all",
) -> pd.DataFrame:
    """Flatten one OIFITS table over a night into a tidy DataFrame.

    Each spectral row is reduced to a single value by taking the median over
    the ``wl_range`` window (µm); errors are averaged the same way.

    Parameters
    ----------
    night : list of OIFitsData
        Output of :func:`load_night`.
    table : {"TF2", "VIS2", "VIS", "T3", "FLUX"}
        OIFITS table to extract.
    band : {"LM", "N"}
        Only files of this band are used.
    wl_range : (float, float), optional
        Wavelength window in µm. Defaults to :data:`DEFAULT_WL_RANGE`.
    chop : {"all", "chop", "nochop"}
        Keep only chopped / non-chopped exposures.

    Returns
    -------
    pandas.DataFrame
        Columns listed in :data:`COLUMNS` plus ``night`` (see
        :func:`night_label`). ``baseline`` holds station names
        (``"A0-G1"``, ``"A0-G1-J2"`` for closure phases, ``"A0"`` for flux),
        sorted so that the four BCD positions share the same label.
    """
    if table not in TABLES:
        raise ValueError(f"Unknown table {table!r}. Choose from {list(TABLES)}.")
    attr, col, errcol = TABLES[table]
    wl_min, wl_max = wl_range or DEFAULT_WL_RANGE.get(band, (0.0, np.inf))

    rows: list[dict] = []
    for d in night:
        if d.band != band:
            continue
        meta = _file_meta(d)
        if chop != "all" and meta["chop"].lower() != chop:
            continue
        tab = getattr(d, attr)
        if tab is None or col not in tab:
            continue

        wl = np.asarray(d.wavelength) * 1e6
        sel = (wl >= wl_min) & (wl <= wl_max)
        if not np.any(sel):
            logger.warning(
                f"No channel in [{wl_min}, {wl_max}] µm for {d.file_path.name}"
            )
            continue

        ref = {int(i): str(n) for i, n in zip(d.sta_index, d.sta_name, strict=False)}
        values = np.atleast_2d(tab[col])
        errors = np.atleast_2d(tab.get(errcol, np.full_like(values, np.nan)))
        if "FLAG" in tab:
            flag = np.atleast_2d(tab["FLAG"]).astype(bool)
            values = np.where(flag, np.nan, values)
            errors = np.where(flag, np.nan, errors)
        mjds = np.atleast_1d(tab["TIME"])
        sta = np.asarray(tab["STA_INDEX"])
        if sta.ndim == 1:
            sta = sta[:, None]
        bcd = f"{d.bcd1_name}-{d.bcd2_name}" if d.bcd1_name else "?"

        for i in range(values.shape[0]):
            rows.append(
                {
                    "mjd": float(mjds[i]),
                    "value": _robust_mean(values[i, sel]),
                    "err": _robust_mean(errors[i, sel]),
                    "baseline": _canonical_label(sta[i], ref),
                    "target": d.target_name.strip(),
                    "category": d.category,
                    "bcd": bcd,
                    **meta,
                    "band": d.band,
                    "disp": d.dispersion_name,
                    "dit": d.dit,
                    "seeing": d.seeing,
                    "tau0": d.tau0 * 1e3,  # s -> ms
                    "file": d.file_path.name,
                }
            )

    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "time"])
    df.insert(1, "time", _mjd_to_datetime(df["mjd"].to_numpy()))
    df.insert(2, "night", night_label(df["time"]))
    return df.sort_values("mjd", ignore_index=True)


def night_label(time: pd.Series) -> pd.Series:
    """Observing night as the ISO date of the evening (ESO convention).

    UT is shifted by -12 h so that a whole Paranal night (~22h-11h UT)
    maps onto the same label, e.g. 2022-07-02T04:17 UT -> ``"2022-07-01"``.
    """
    if time.empty:
        return pd.Series([], dtype=str)
    return (time - pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d")


def _mjd_to_datetime(mjd: np.ndarray) -> pd.Series:
    if mjd.size == 0:
        return pd.Series([], dtype="datetime64[ns]")
    return pd.Series(pd.to_datetime(Time(mjd, format="mjd").to_datetime()))


def tf_statistics(df_tf: pd.DataFrame) -> pd.DataFrame:
    """Per-baseline transfer-function stability summary.

    The TF differs between BCD positions and chopping modes, so each value
    is first normalised by the median of its (baseline, BCD, chop) group; the dispersion then only
    reflects the temporal stability of the TF (atmosphere, bad calibrator,
    wrong diameter...).

    Returns
    -------
    pandas.DataFrame
        Indexed by baseline with columns ``n`` (points), ``n_cal``
        (distinct calibrators), ``median``, ``scatter_pct`` (robust MAD-based
        relative dispersion in %), ``min`` and ``max``.
    """
    cols = ["n", "n_cal", "median", "scatter_pct", "min", "max"]
    df = df_tf[np.isfinite(df_tf["value"])]
    if df.empty:
        return pd.DataFrame(columns=cols)

    ref = df.groupby(["baseline", "bcd", "chop"])["value"].transform("median")
    df = df.assign(norm=df["value"] / ref)

    def _mad_pct(x: pd.Series) -> float:
        return float(100 * 1.4826 * np.median(np.abs(x - np.median(x))))

    g = df.groupby("baseline")
    stats = pd.DataFrame(
        {
            "n": g["value"].size(),
            "n_cal": g["target"].nunique(),
            "median": g["value"].median(),
            "scatter_pct": g["norm"].apply(_mad_pct),
            "min": g["value"].min(),
            "max": g["value"].max(),
        }
    )
    return stats[cols]
