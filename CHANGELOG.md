# Changelog

All notable changes to `colab-mcp` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-19

### Added
- **Authentication & Encrypted Secrets Vault (`colab_mcp.auth`):**
  - Google OAuth 2.0 flows (`oauth_remote`, `oauth_loopback`, `oauth_adc`), Chrome DevTools Protocol (`cdp_browser`) login with `DEVNULL` subprocess isolation, `SAPISIDHASH` request signing, and OS-native keyring storage with AES-256-GCM fallback.
  - Encrypted Secrets/Token Vault (`colab_set_secret`, `colab_list_secrets`, `colab_delete_secret`, plus CLI `set-secret`, `list-secrets`, `delete-secret`) with dynamic egress redaction and automatic `os.environ` / `google.colab.userdata` injection.
  - MCP tools: `colab_login`, `colab_get_auth_status`, `colab_logout`, `colab_set_secret`, `colab_list_secrets`, `colab_delete_secret`.
- **Runtime, Session Management & System Telemetry (`colab_mcp.runtime`):**
  - Colab Tunnel Frontend (TFE) REST client supporting hardware acceleration (`NONE`, `T4`, `L4`, `A100`, `H100`, `V5E1`, `V6E1`) and High-RAM shapes, pre-assignment re-attachment, 35s keep-alive heartbeat daemon, and 30-minute idle inactivity watchdog (preserving active `busy` kernels).
  - Live Colab TFE compatibility for camelCase `ccu-info` fields, integer enum normalization (`variant`, `machineShape`), and HTTP 204 `/unassign` handling.
  - Live VM GPU VRAM/utilization, system RAM, and disk space probe (`colab_get_system_metrics`).
  - MCP tools: `colab_list_runtime_types`, `colab_list_sessions`, `colab_connect_runtime`, `colab_get_runtime_status`, `colab_get_system_metrics`, `colab_disconnect_runtime` (including remote endpoint termination and `disconnect_all=True`).
- **Kernel WebSocket Bridge, Headless Shims & Interactive Stdin (`colab_mcp.kernel`):**
  - Jupyter Messaging Protocol v5.3 WebSocket client with `subprotocols=None`, `compression=None`, `?session_id=` support, message correlator, chronological `TextContent`/`ImageContent` MIME conversion, ECMA-48 ANSI sanitization, head-tail truncation bounds, and SIGINT / escalated kernel restart handling.
  - Headless `google.colab` popup shims (`userdata.get`, `drive.mount` workspace fallback, `auth.authenticate_user`, `files.upload`/`download`) and interactive Jupyter `stdin` (`allow_stdin=True`, `stdin_inputs` with `$SECRET:<NAME>` interpolation, `waiting_for_input` status, and `colab_send_input`).
  - MCP tools: `colab_execute_code`, `colab_get_execution_status`, `colab_get_cell_output`, `colab_send_input`, `colab_interrupt_execution`.
- **Notebook Execution Engine (`colab_mcp.notebook`):**
  - `nbformat` v4 validation, workspace path guards, sequential and selective 0-based `cell_indices` execution, `stop_on_error` policy, `stdin_inputs` / `env` forwarding, post-reset shim re-bootstrap, and atomic disk serialization.
  - MCP tool: `colab_execute_notebook`.
- **VM File Transfer & Google Drive Tools (`colab_mcp.files`):**
  - Jupyter Contents API listing/upload/download with automatic 512 KB chunked kernel transfer fallback and SHA-256 checksum verification, plus Google Drive v3 API tools.
  - MCP tools: `colab_list_files`, `colab_download_file`, `colab_upload_file`, `colab_drive_list`, `colab_drive_download`, `colab_drive_upload`.
- **Server Transports, CLI & Community Infrastructure:**
  - MCP SDK v2 (`MCPServer`) supporting `stdio` (strict `stderr` logging isolation) and `sse` transports, global egress credential redaction, and Typer CLI (`run`, `login`, `status`, `set-secret`, `list-secrets`, `delete-secret`, `list-runtimes`, `list-sessions`, `disconnect`, `logout`, `version`).
  - GitHub Actions CI matrix across Linux, Windows, and macOS (Python 3.10–3.13).
