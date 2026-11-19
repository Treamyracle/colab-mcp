# Contributing to `colab-mcp`

Thank you for contributing to `colab-mcp`! This guide covers local setup, testing, and codebase conventions so your pull request lands smoothly.

---

## Development Setup

```bash
git clone https://github.com/Treamyracle/colab-mcp.git
cd colab-mcp

python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

pip install -e ".[all]"
playwright install chromium  # Optional: only needed for cdp_browser tests/usage
```

---

## Running Tests & Linting

All offline tests use deterministic mocks and the local `ColabTwin` server (`tests/e2e/opaque/`), so **no Google account or Colab credits are required**:

```bash
# Full offline verification suite (unit + integration + opaque E2E)
pytest tests/unit/ tests/integration/ tests/e2e/opaque/ -q

# Optional lint check
ruff check src tests
```

Live Colab tests in `tests/e2e/test_live_*.py` require interactive authentication and are excluded from CI.

---

## Project Layout

```text
src/colab_mcp/
├── auth/       # OAuth 2.0, Chrome CDP login, SAPISIDHASH signing, OS keyring
├── runtime/    # Colab TFE client, session manager, keep-alive & watchdog
├── kernel/     # Jupyter v5.3 WebSocket bridge, correlator, MIME/ANSI pipeline
├── notebook/   # nbformat v4 loader, path guards, cell-by-cell runner
├── files/      # VM file transfer (Contents API + chunked kernel) & Google Drive v3
├── utils/      # stderr-only logging isolation, nbh generator, credential redaction
├── server.py   # FastMCP server, lifespan, stdio/sse transports, egress redaction
└── cli.py      # Typer CLI entrypoints
```

---

## Engineering Conventions

1. **Never write to `sys.stdout` in server/library code:**
   In `stdio` MCP mode, `sys.stdout` carries JSON-RPC 2.0 frames. Always use `colab_mcp.utils.logging.get_logger(__name__)`, which binds exclusively to `sys.stderr`.
2. **Never leak secrets:**
   Do not log or return OAuth access tokens, refresh tokens, proxy tokens, or session cookies. Ensure any new credential formats are covered by `colab_mcp.utils.redaction`.
3. **Keep tool schemas and responses Pydantic-typed:**
   Follow the existing pattern when adding an MCP tool:
   - Define request/response Pydantic models in the subsystem's `models.py`.
   - Implement an async `*_impl(...)` function in `tools.py` for direct unit testing.
   - Register the `@mcp.tool(...)` wrapper in `register_*_tools(mcp)`.
   - Add unit tests in `tests/unit/` and register the tool in `ALL_EXPECTED_MCP_TOOLS` in [tests/integration/test_mcp_tools.py](tests/integration/test_mcp_tools.py).
4. **Update documentation:**
   Keep the tool/CLI tables in [README.md](README.md) and [CHANGELOG.md](CHANGELOG.md) current when adding or modifying tools.

---

## Pull Request Process

1. Fork the repository and create a feature branch from `main`.
2. Add or update tests covering your changes.
3. Confirm `pytest tests/unit/ tests/integration/ tests/e2e/opaque/ -q` passes with zero failures.
4. Open a pull request against `main` using the PR template.
