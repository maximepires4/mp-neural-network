# Code Quality & Contribution Guide

This guide outlines the standards, tools, and workflows used in `mp-neural-network` to ensure a robust, maintainable, and high-quality codebase.

---

## 1. Quality Standards & Tools

We rely on a strict set of tools to enforce quality automatically.

| Tool | Purpose | Scope |
| :--- | :--- | :--- |
| **[Ruff](https://docs.astral.sh/ruff/)** | Linter & Formatter. Replaces Flake8, Black, and Isort. | Entire project (`src`, `tests`, `examples`) |
| **[Mypy](https://mypy-lang.org/)** | Static Type Checker. | Strictly enforced on `src/mpneuralnetwork` |
| **[Pytest](https://docs.pytest.org/)** | Unit Testing framework. | `tests/` |
| **[Coverage](https://coverage.readthedocs.io/)** | Code coverage measurement. | `src/mpneuralnetwork` (must cover >90%) |

---

## 2. Local Development Workflow

### Installation

Install the project in editable mode with all development dependencies:

```bash
pip install -e .[dev,test]
pre-commit install
```

### Routine Commands

During development, you should frequently run these commands to ensure your code meets the standards.

#### **Format & Lint (Ruff)**

Fixes style issues and imports automatically.

```bash
ruff check . --fix
ruff format .
```

#### **Type Checking (Mypy)**

Verifies type safety. Configuration is loaded from `pyproject.toml`.

```bash
mypy
```

#### **Run Tests (Pytest)**

Executes the test suite and reports coverage.

```bash
coverage run -m pytest
coverage report -m
```

---

## 3. Git Workflow & Commit Convention

To automate versioning and changelog generation, we strictly adhere to the **[Conventional Commits](https://www.conventionalcommits.org/)** specification.

### Commit Message Format

```text
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Allowed Types

| Type | Meaning | Version Impact | Example |
| :--- | :--- | :--- | :--- |
| **feat** | A new feature | **MINOR** (`1.1.0`) | `feat(layer): add LSTM support` |
| **fix** | A bug fix | **PATCH** (`1.0.1`) | `fix(optim): resolve div by zero in Adam` |
| **docs** | Documentation only | None | `docs: update installation guide` |
| **style** | Formatting, missing semi-colons, etc. | None | `style: format with ruff` |
| **refactor** | Code change that is neither fix nor feat | None | `refactor: simplify activation logic` |
| **perf** | A code change that improves performance | None | `perf: vectorize loss calculation` |
| **test** | Adding or correcting tests | None | `test: add unit tests for Conv2D` |
| **chore** | Build process or aux tool changes | None | `chore: update dependencies` |

> **BREAKING CHANGES**: Adding `BREAKING CHANGE:` in the footer or appending `!` after the type/scope triggers a **MAJOR** version bump (`2.0.0`).

---

## 4. Release Process (CI/CD)

We use **[python-semantic-release](https://python-semantic-release.readthedocs.io/)** to fully automate the release cycle.

### How it works

1. **Push to Main**: You merge a Pull Request or push code to the `main` branch.
2. **CI Checks**: GitHub Actions triggers the `Tests` workflow (Lint + Types + Pytest).
3. **Release Decision**:
    * If tests **pass**: The release workflow analyzes new commits since the last tag.
    * If commits contain `feat` or `fix`: A new version is calculated.
4. **Publication**:
    * `pyproject.toml` is updated with the new version.
    * A `CHANGELOG.md` is generated or updated.
    * A Git Tag is created.
    * The package is built and uploaded to **PyPI**.
