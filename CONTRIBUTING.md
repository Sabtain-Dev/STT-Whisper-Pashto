# Contributing

Thank you for considering a contribution to STT-Whisper-Pashto.

## Before you start

- Check existing issues and pull requests to avoid duplicating work.
- For substantial changes, open an issue first to discuss the proposed approach.
- Keep changes focused and update documentation when behavior or setup changes.

## Development setup

Use Python 3.10, then install the project and development dependencies:

```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements_dev.txt
```

## Checks

Run the test suite and lint checks before opening a pull request:

```bash
pytest -v
ruff check .
```

If a change requires particular environment variables, models, audio samples, or hardware, document that in the pull request and explain which checks you were able to run.

## Pull requests

- Describe the problem and the solution, including any relevant issue links.
- Include tests for behavior changes and update relevant documentation.
- Keep commits and diffs focused; do not include unrelated formatting changes.
- Confirm that automated checks pass, or explain any failures.
