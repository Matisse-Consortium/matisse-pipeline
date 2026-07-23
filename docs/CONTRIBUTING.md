# Contributing to matisse-pipeline

First off, thank you for considering contributing to the Matisse Pipeline!

## 🛠️ Local Development Setup

We use [uv](https://docs.astral.sh/uv/) for fast, reliable Python package and dependency management.

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:

    ```bash
    git clone [https://github.com/YOUR-USERNAME/matisse-pipeline.git](https://github.com/YOUR-USERNAME/matisse-pipeline.git)
    cd matisse-pipeline
    ```

3. **Create a virtual environment and install dependencies** in editable mode with development groups:

    ```bash
    uv sync --all-groups
    source .venv/bin/activate
    ```

4. **Install pre-commit hooks**:
    This ensures your code is automatically formatted and linted before every commit.

    ```bash
    uv run pre-commit install
    ```

## 🌿 Branching Policy

* Always create a new branch for your work: `git checkout -b feature/my-new-feature` or `git checkout -b fix/issue-description`.
* **Commit Safety**: We use the `no-commit-to-branch` hook to prevent accidental commits directly to `main`. This forces you to work on a feature branch.
* **Mandatory PRs**: Even with local hooks, all changes must be submitted via a Pull Request. Direct pushes to the `main` branch are restricted at the repository level.

## 📝 Commit Policy

This project follows the **[Conventional Commits](https://www.conventionalcommits.org/)** standard. This allows us to use `commitizen` to manage versioning and generate changelogs automatically.

We strongly suggest using this format for your commits to help maintainers track changes efficiently:

| Prefix | Description | Version Bump |
| :--- | :--- | :--- |
| **feat:** | A new feature | `minor` |
| **fix:** | A bug fix | `patch` |
| **feat!:** / **fix!:** | Breaking change (note the `!`) | **`major`** |
| **docs:** | Documentation only changes | `none` |
| **refactor:** | Code change that neither fixes a bug nor adds a feature | `none` |
| **test:** | Adding missing tests or correcting existing tests | `none` |
| **chore:** | Maintenance tasks (dependency updates, CI, etc.) | `none` |

## ✅ Quality Standards

Before submitting a Pull Request, please ensure:

1. **Linting & Formatting**: Run `ruff` to check for style issues.

    ```bash
    uv run ruff check .
    uv run ruff format .
    ```

2. **Tests**: Ensure all tests pass.

    ```bash
    uv run pytest
    ```

3. **Type Checking**: If applicable, verify types with `mypy`.

    ```bash
    uv run mypy src/
    ```

## 🚀 Submitting a Pull Request

1. **Push** your branch to your fork on GitHub.
2. **Open a Pull Request** against our `main` branch.
3. **Describe your changes**: Explain *what* you changed and *why*. Reference any related issues (e.g., "Closes #12").
4. **Wait for CI**: Our GitHub Actions will automatically run the test suite. If it fails, please check the logs and push the necessary fixes.

> Note: For first-time contributors, GitHub Actions will require approval from a maintainer. Once your first PR is merged, future PRs will automatically trigger the CI.

## 💬 Communication

If you have questions or want to discuss a major change before implementing it, please open an **Issue** or reach out mainteners.

---
*By contributing, you agree that your contributions will be licensed under the project's open-source license.*
