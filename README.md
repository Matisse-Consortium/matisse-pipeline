<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <a href="https://github.com/Matisse-Consortium/matisse-pipeline">
    <img src="https://raw.githubusercontent.com/Matisse-Consortium/matisse-pipeline/docs/logo/logo_pipeline.png" alt="MATISSE pipeline logo" width="300"/>
  </a>
</p>
<!-- markdownlint-enable MD033 MD041 -->

[![Status](https://img.shields.io/badge/status-Beta-orange.svg)](https://github.com/Matisse-Consortium/matisse-pipeline) [![Python versions](https://img.shields.io/badge/python-3.10–3.14-blue)](<https://www.python.org/>) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![CI](https://github.com/Matisse-Consortium/matisse-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/Matisse-Consortium/matisse-pipeline/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Matisse-Consortium/matisse-pipeline/branch/main/graph/badge.svg?token=N2CINYUJBI)](https://codecov.io/gh/Matisse-Consortium/matisse-pipeline)

[![Build with uv](https://img.shields.io/badge/build-uv-5A29E4.svg)](https://docs.astral.sh/uv/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/charliermarsh/ruff) [![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg)](https://docs.pytest.org/en/stable/)

Modern and modular **MATISSE** interferometric data-reduction pipeline python interface. Developed by the [MATISSE Consortium](https://www.eso.org/sci/facilities/paranal/instruments/matisse.html). It provides a user-friendly command-line interface (`matisse`) as well as backward compatibility with the original consortium scripts located in `legacy/`.

---

## 🚀 Installation (Users)

> Recommended for end-users who only need to use the pipeline.

This project uses [`uv`](https://github.com/astral-sh/uv) to manage environments and dependencies.
It’s fully compatible with `pip` but much faster and simpler to use.

### 1️⃣ Install uv

**On Linux / macOS:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2️⃣ Create and activate a virtual environment

```bash
uv venv
source .venv/bin/activate
```

### 3️⃣ Install the package

```bash
uv pip install git+https://github.com/Matisse-Consortium/matisse-pipeline.git
```

---

## 🧑‍💻 Developer installation

> For contributors or developers working on the pipeline codebase.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Matisse-Consortium/matisse-pipeline.git
cd matisse-pipeline
```

### 2️⃣ Install in editable mode with dev dependencies

```bash
uv pip install -e . --group test --group typecheck
```

This installs:

- `pytest`, `ruff`, and `pre-commit` for testing and linting
- `mypy` and `types-termcolor` for type checking

### 3️⃣ Run tests

```bash
uv run pytest
```

### 4️⃣ Lint and type check

```bash
uv run ruff check src/
uv run mypy src/
```

---

## 🧰 Legacy Scripts Compatibility

The original MATISSE reduction tools (`mat_autoPipeline.py`, etc.) are preserved in the `legacy/` folder for full backward compatibility.
They can be accessed by adding the legacy path to your environment:

```bash
export PATH="$PATH:$(python -c 'import matisse, pathlib; print(pathlib.Path(matisse.__file__).parent / "legacy")')"
```

You can add this line to your `~/.zshrc` or `~/.bashrc` to make it persistent.

Once exported, the commands will be available globally, e.g.:

```bash
mat_autoPipeline.py --dirCalib=.
```

---

## 🧩 Repository Structure

```bash
matisse-pipeline/
├── src/matisse/
│   ├── cli/                  # Main CLI entry point (`matisse`)
│   ├── legacy/               # Legacy MATISSE reduction scripts
│   ├── core/                 # Core pipeline modules
│   └── viewer/               # Viewer interface
├── tests/                    # Unit tests
├── pyproject.toml            # Project configuration
├── CHANGELOG.md              # Project follow-up and versioning
└── README.md
```

---

## 🧑‍🔬 Citation / Credits

If you use this pipeline in your research, please cite the MATISSE Consortium and the corresponding instrument papers.

> Maintained by the **MATISSE Consortium**
> Contributions welcome via pull requests.
