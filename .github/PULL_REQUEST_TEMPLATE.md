## Summary

<!-- Describe what this PR changes and why. Link any related issue (`Fixes #...`). -->

## Checklist

- [ ] Offline test suite passes (`pytest tests/unit/ tests/integration/ tests/e2e/opaque/ -q`)
- [ ] No writes to `sys.stdout` during MCP stdio mode (all diagnostics use `sys.stderr`)
- [ ] Credentials, OAuth tokens, refresh tokens, and cookies (`SAPISID`, `OSID`) are redacted from logs/outputs
- [ ] Documentation (`README.md`, `docs/ARCHITECTURE.md`, `CHANGELOG.md`) updated if tools, CLI commands, or architecture changed
