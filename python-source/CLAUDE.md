# Claude Instructions

## Package Management

- **Do NOT** install, uninstall, or modify packages or dependencies.
- **Do NOT** run `pip`, `pip3`, or any pip-related commands.
- Package and dependency management is handled manually by the user.

## Python Runtime

This project uses [uv](https://github.com/astral-sh/uv). To run Python:

- Preferred: `uv run python <script>`
- Alternative: `.venv\Scripts\python.exe <script>`
