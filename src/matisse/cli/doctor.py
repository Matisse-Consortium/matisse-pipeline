# mytool/cli/doctor.py
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import typer

app = typer.Typer(help="Run environment diagnostics for MATISSE (EsoRex).")


# ---------- result model ----------


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    fatal: bool = False


@dataclass(frozen=True)
class RecipeDirProbe:
    recipe_dir: Path
    matisse_recipes: list[str]


def _print_result(r: CheckResult) -> None:
    if r.ok:
        typer.echo(f"✅ {r.name}: {r.message}")
    else:
        icon = "❌" if r.fatal else "⚠️"
        typer.echo(f"{icon} {r.name}: {r.message}")


# ---------- subprocess helpers ----------


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip() or "(no stderr)"
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return proc


# ---------- esorex helpers ----------


def _list_esorex_recipes(recipe_dir: Path | None = None) -> list[str]:
    cmd = ["esorex"]

    if recipe_dir is not None:
        cmd += ["--recipe-dir=" + str(recipe_dir)]
    cmd += ["--recipes"]

    proc = _run(cmd)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _get_env_recipe_dirs() -> list[Path]:
    env = os.environ.get("ESOREX_PLUGIN_DIR", "").strip()
    if not env:
        return []
    return [Path(p).expanduser() for p in env.split(":") if p.strip()]


def _candidate_recipe_dirs(extra_candidates: list[Path]) -> list[Path]:
    candidates: list[Path] = []

    # 1) User environment first
    candidates.extend(_get_env_recipe_dirs())

    # 2) Known common locations (MacPorts + typical ESO installs)
    candidates.extend(
        [
            Path("/opt/local/lib/esopipes-plugins"),  # MacPorts
            Path("/usr/local/lib/esopipes-plugins"),
            Path("/usr/lib/esopipes-plugins"),
            Path("/usr/lib64/esopipes-plugins"),
        ]
    )

    # 3) User-provided extras
    candidates.extend(extra_candidates)

    # De-duplicate while keeping order
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in candidates:
        key = str(p)
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def find_matisse_recipe_dir(
    extra_candidates: list[Path],
    verbose: bool,
) -> RecipeDirProbe | None:
    for d in _candidate_recipe_dirs(extra_candidates):
        if not d.exists() or not d.is_dir():
            if verbose:
                typer.echo(f"… skip candidate (missing/not dir): {d}")
            continue

        try:
            recipes = _list_esorex_recipes(recipe_dir=d)
        except Exception as exc:
            if verbose:
                typer.echo(f"… candidate failed: {d} ({exc})")
            continue

        matisse = sorted(r for r in recipes if r.startswith("mat_"))
        if verbose:
            typer.echo(f"… candidate ok: {d} (mat_*={len(matisse)})")

        if matisse:
            return RecipeDirProbe(recipe_dir=d, matisse_recipes=matisse)

    return None


def check_esorex_in_path() -> CheckResult:
    path = shutil.which("esorex")
    if path is None:
        return CheckResult(
            name="esorex", ok=False, message="not found in PATH", fatal=True
        )
    return CheckResult(name="esorex", ok=True, message=f"found at {path}")


def check_matisse_recipes(
    recipe_dir: Path | None,
    require_any: bool,
) -> tuple[CheckResult, list[str]]:
    try:
        recipes = _list_esorex_recipes(recipe_dir=recipe_dir)
    except Exception as exc:
        return (
            CheckResult(name="MATISSE recipes", ok=False, message=str(exc), fatal=True),
            [],
        )

    matisse = sorted(r for r in recipes if r.startswith("mat_"))
    if require_any and not matisse:
        return (
            CheckResult(
                name="MATISSE recipes",
                ok=False,
                message="no mat_* recipes found",
                fatal=True,
            ),
            [],
        )

    msg = f"{len(matisse)} found."
    return (CheckResult(name="MATISSE recipes", ok=True, message=msg), matisse)


def _clickable_link(label: str, url: str) -> str:
    return f"\033]8;;{url}\033\\{label}\033]8;;\033\\"


def echo_esorex_missing() -> None:
    link_brew = "https://www.eso.org/sci/software/pipe_aem_brew.html"
    link_macports = (
        "https://www.eso.org/sci/software/pipelines/installation/macports.html"
    )

    typer.secho(
        f"\nDetected platform: {platform.system()} ({sys.platform})\n",
        fg=typer.colors.CYAN,
    )

    typer.echo("This package requires the ESO pipeline environment (esorex).\n")

    typer.secho("Recommended installation methods:\n", bold=True)

    typer.echo("  1) Homebrew (macOS and Linux)")
    typer.secho(
        f"     {_clickable_link(link_brew, link_brew)}",
        fg=typer.colors.BLUE,
    )

    typer.echo("\n  2) MacPorts (macOS)")
    typer.secho(
        f"     {_clickable_link(link_macports, link_macports)}",
        fg=typer.colors.BLUE,
    )

    typer.echo(
        "\nAfter installation, make sure the `esorex` command is available in your PATH."
    )


# ---------- typer command ----------


@app.command()
def doctor(
    recipe_dir: Path | None = typer.Option(
        None,
        "--recipe-dir",
        help="Force a recipe directory (same as esorex --recipe-dir).",
    ),
    macports_probe: bool = typer.Option(
        True,
        "--macports-probe/--no-macports-probe",
        help="Probe common install locations (incl. MacPorts) if ESOREX_PLUGIN_DIR is empty.",
    ),
    require_any: bool = typer.Option(
        True,
        "--require-any/--no-require-any",
        help="Fail if no mat_* recipes are found.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Print candidate probing details.",
    ),
) -> None:
    """
    Check that MATISSE recipes (mat_*) are visible by esorex, and report the recipe directory used/found.

    Exit codes:
      0 = OK
      2 = fatal error(s)
    """
    results: list[CheckResult] = []

    # 1) esorex in PATH
    results.append(check_esorex_in_path())
    if not results[-1].ok:
        for r in results:
            _print_result(r)
        echo_esorex_missing()
        typer.echo("\nEnvironment check failed.")
        raise typer.Exit(code=2)

    # 2) Determine recipe dir
    chosen_dir: Path | None = recipe_dir
    discovered: RecipeDirProbe | None = None

    if chosen_dir is None:
        env_dirs = _get_env_recipe_dirs()
        if env_dirs:
            # If multiple dirs in env, esorex supports colon-separated list.
            # For the doctor, we show them and let esorex default behavior handle it.
            # But for mat_* probing, we only pick the first existing dir as "representative".
            results.append(
                CheckResult(
                    name="ESOREX_PLUGIN_DIR",
                    ok=True,
                    message=":".join(str(p) for p in env_dirs),
                    fatal=False,
                )
            )
            # Pick first existing dir for explicit probing display; actual listing is still correct if user uses env.
            first_existing = next(
                (p for p in env_dirs if p.exists() and p.is_dir()), None
            )
            chosen_dir = first_existing
        elif macports_probe:
            extra: list[Path] = []
            discovered = find_matisse_recipe_dir(
                extra_candidates=extra, verbose=verbose
            )
            if discovered is not None:
                chosen_dir = discovered.recipe_dir

    # 3) Report recipe dir decision
    if recipe_dir is not None:
        results.append(
            CheckResult(name="recipe dir", ok=True, message=f"forced: {recipe_dir}")
        )
    elif discovered is not None:
        results.append(
            CheckResult(
                name="recipe dir", ok=True, message=f"found: {discovered.recipe_dir}"
            )
        )
    elif chosen_dir is not None:
        results.append(
            CheckResult(name="recipe dir", ok=True, message=f"using: {chosen_dir}")
        )
    else:
        # No explicit dir; esorex will still search defaults. We just can't report a path.
        results.append(
            CheckResult(
                name="recipe dir",
                ok=False,
                message="unknown (ESOREX_PLUGIN_DIR empty; no directory forced; probing disabled or failed)",
                fatal=False,
            )
        )

    # 4) Check MATISSE recipes visibility
    # If chosen_dir is None, we call esorex --recipes without --recipe-dir (default behavior).
    mat_check, _ = check_matisse_recipes(recipe_dir=chosen_dir, require_any=require_any)
    results.append(mat_check)

    fatal = any((not r.ok and r.fatal) for r in results)
    for r in results:
        _print_result(r)

    if fatal:
        typer.echo("\nEnvironment check failed.")
        raise typer.Exit(code=2)

    typer.echo("\nMATISSE environment looks OK.")
    raise typer.Exit(code=0)
