# Agent Instructions

## Primary Focus

This workspace is for the Kaggle competition:

https://www.kaggle.com/competitions/fah-mai-the-finale-enterprise-data-agentic-showdown/

Default competition slug:

`fah-mai-the-finale-enterprise-data-agentic-showdown`

When a task mentions Kaggle, the competition, dataset files, submissions, evaluation, docs, RAG, or agent behavior without another explicit target, assume it refers to this competition.

## Kaggle MCP Usage

- Prefer the Kaggle MCP tools for reading competition data and metadata.
- Use `list_competition_data_files` or `list_competition_data_tree_files` with the default slug above before guessing file names.
- The Kaggle MCP server is configured in `C:\Users\Gunte\.codex\config.toml` as `mcp_servers.kaggle`.
- Auth is provided through the `KAGGLE_AUTH_HEADER` environment-backed header in that config. Do not print or expose the token value.
- If an in-session Kaggle MCP call returns `Unauthenticated`, the session may have loaded before the auth config changed. Start a new Codex session or run a one-off authenticated MCP call using the configured header.

## Working Style

- Keep work centered on the competition objective and its provided data.
- Read `README.md` from the competition data first when orienting.
- Treat files under `docs/` and competition-provided markdown as source material for retrieval, analysis, and answers.
- Do not fabricate competition rules, metrics, schemas, or file contents. Verify them from Kaggle data or local files.
- Avoid unrelated refactors in `fahmai-test-web`, `python-source`, or `docs` unless they directly support the competition task.

## Local Workspace

Top-level folders currently include:

- `docs`
- `fahmai-test-web`
- `python-source`
- `tools`

Use existing project structure and conventions before adding new tooling.
