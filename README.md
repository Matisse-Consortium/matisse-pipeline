# MATISSE Pipeline

MATISSE Pipeline is a Python-based framework for the automated reduction and calibration of data from the **MATISSE interferometric instrument** (ESO/VLTI).
It provides a modern, user-friendly command-line interface (`matisse`) as well as backward compatibility with the original consortium scripts located in `legacy/`.

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

**On windows (PowerShell):**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
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
export PATH="$PATH:$(python -c 'import matisse_pipeline, pathlib; print(pathlib.Path(matisse_pipeline.__file__).parent / "legacy")')"
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
├── src/matisse_pipeline/
│   ├── cli.py                # Main CLI entry point (`mat`)
│   ├── legacy/               # Legacy MATISSE reduction scripts
│   ├── core/                 # Core pipeline modules
│   └── viewer/               # Viewer interface
├── tests/                    # Unit tests
├── pyproject.toml            # Project configuration (dependencies, groups, etc.)
└── README.md
```

---

## 🧑‍🔬 Citation / Credits

If you use this pipeline in your research, please cite the MATISSE Consortium and the corresponding instrument papers.

> Maintained by the **MATISSE Consortium**
> Contributions welcome via pull requests.
