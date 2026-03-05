# Contributing to Steno

Thanks for your interest in contributing to Steno! This document provides guidelines and instructions for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/steno.git
   cd steno
   ```
3. Install dependencies:
   ```bash
   uv sync --extra dev
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Running the app

```bash
uv run main.py
```

### Running tests

```bash
uv run pytest tests/ -v
```

All tests must pass before submitting a pull request.

### Project Conventions

- **Python**: Follow PEP 8. Use type hints where practical.
- **Frontend**: All UI code lives in `static/index.html` as a single file with inline CSS and JS. No build tools, no npm.
- **i18n**: Every user-facing string must go through the i18n system. Add keys to both `locales/en.json` and `locales/es.json`.
- **Tests**: Add tests for new functionality. Test files live in `tests/` and follow the `test_<module>.py` convention.

## Pull Request Process

1. Ensure all tests pass (`uv run pytest tests/ -v`)
2. Update documentation if your change affects usage or configuration
3. Add both English and Spanish translations for any new UI strings
4. Keep pull requests focused — one feature or fix per PR
5. Write a clear PR description explaining *what* and *why*

## Reporting Bugs

Open an issue with:

- A clear title and description
- Steps to reproduce the problem
- Expected vs actual behavior
- Your macOS version and Apple Silicon chip (M1/M2/M3/M4)
- Python version (`python --version`)

## Feature Requests

Open an issue describing:

- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
