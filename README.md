# Google Colab MCP Server (`colab-mcp`)

[![CI](https://github.com/Treamyracle/colab-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/Treamyracle/colab-mcp/actions/workflows/ci.yml)
[![MCP Version](https://img.shields.io/badge/MCP-2024--11--05-blue.svg)](https://modelcontextprotocol.io/)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

A **Model Context Protocol (MCP)** server connecting AI coding assistants (Claude Desktop, Cursor, Antigravity, Cline, Roo-Code, Windsurf) directly to **consumer Google Colab** (`colab.research.google.com`).

Authenticate with your Google account, provision CPU/GPU/TPU runtimes, execute Python snippets or full `.ipynb` Jupyter notebooks, stream cell outputs and Matplotlib plots, transfer `.zip`/dataset files to and from the VM, and manage remote sessions—all without leaving your editor or agent.

---

## 🌟 Key Features

- **Consumer Colab Automation:** Connects directly to Google Colab's Tunnel Frontend (TFE) without requiring enterprise GCP / Vertex AI configurations.
- **Hardware Acceleration on Demand:** Provision and switch between standard CPU (`NONE`), NVIDIA GPUs (`T4`, `L4`, `A100`, `H100`), and Google Cloud TPUs (`V5E1`, `V6E1`). Supports High-RAM shapes (`~51 GB`) on Pro/Pro+ accounts.
- **Remote & Local Session Management:** List all active Colab VMs on your account (`colab_list_sessions`), re-attach to compatible sessions (`reuse_existing=True`) to avoid Free Tier HTTP 412 quota errors, or terminate specific/all sessions (`colab_disconnect_runtime`).
- **Multi-Account Support:** Switch accounts cleanly via `auth_user_index` (`?authuser=N` and `X-Goog-AuthUser: N`).
- **Zero-Friction Google Authentication:**
  - **OAuth 2.0 Remote Flow:** Copy-paste authorization code for headless, remote, and agent-driven setups.
  - **Local Loopback Flow:** Browser-based OAuth callback on `127.0.0.1`.
  - **Ambient ADC:** Reuses existing `gcloud auth application-default login` credentials.
  - **Chrome CDP Fallback:** Real Chrome login via Chrome DevTools Protocol with subprocess `DEVNULL` stdout isolation and `SAPISIDHASH` request signing.
- **OS-Native Keystore Storage:** Credentials are encrypted via Windows DPAPI, macOS Keychain, or Linux Secret Service (with an AES-256-GCM `0600` file fallback).
- **Keep-Alive Daemon & 30-Minute Watchdog:** Pings `/{endpoint}/keep-alive/` every 35s and automatically deallocates idle runtimes after 30 minutes of inactivity (skipping deallocation while a kernel is actively `busy`).
- **Jupyter Messaging Protocol v5.3 WebSocket Bridge:** Intermediate kernel discovery/creation (`/api/kernels`), RFC 6455 handshake (`subprotocols=None`, `compression=None`, `?session_id=`), chronological multimodal output capture (`TextContent` + `ImageContent`), ECMA-48 ANSI sanitization, head-tail output truncation, and non-destructive SIGINT / escalated kernel restart.
- **Complete `.ipynb` Notebook Engine:** Load, validate (`nbformat` v4), execute sequentially or by 0-based `cell_indices`, configure `stop_on_error` and `reset_kernel_before_run`, and persist executed outputs back to disk.
- **VM File Transfer & Google Drive Integration:** List, upload, and download binary files (such as exported `.zip` archives, datasets, and checkpoints) via Colab's Jupyter Contents API with automatic 512 KB chunked kernel streaming and SHA-256 verification for large/out-of-root files, plus Google Drive API v3 tools.
- **Dual Transports:** Standard I/O (`stdio`) with strict `stderr` log isolation, plus Server-Sent Events (`sse`).

---

## 🏗️ Architecture

```
+-----------------------------------------------------------------------------------------+
|                                    AI AGENT HOST                                        |
|             (Claude Desktop / Cursor / Antigravity / Windsurf / VS Code Cline)          |
+-----------------------------------------------------------------------------------------+
                                             |
                                             | MCP Protocol (JSON-RPC 2.0)
                                             v
+=========================================================================================+
|                              COLAB MCP SERVER ENGINE                                    |
|                                                                                         |
|  +-----------------------------+               +-------------------------------------+  |
|  |     stdio Transport         |               |       SSE Transport (HTTP)          |  |
|  |  - JSON-RPC over stdin/out  |               |  - GET /sse (Event Stream)          |  |
|  |  - Stderr logger isolation  |               |  - POST /messages (RPC Dispatch)    |  |
|  +-----------------------------+               +-------------------------------------+  |
|                 |                                                 |                     |
|                 +------------------------+------------------------+                     |
|                                          |                                              |
|                                          v                                              |
|  +-----------------------------------------------------------------------------------+  |
|  |                             FastMCP Tool Dispatcher                               |  |
|  |   - Pydantic v2 schema validation & Global Egress Credential Redaction            |  |
|  +-----------------------------------------------------------------------------------+  |
|       |                        |                        |                        |      |
|       v                        v                        v                        v      |
|  +------------+         +--------------+         +--------------+         +----------+  |
|  |   auth/    |         |   runtime/   |         |   kernel/    |         |  files/  |  |
|  | - OAuth2   |         | - TFE Client |         | - WS Client  |         | - VM I/O |  |
|  | - CDP      |         | - Sessions   |         | - Correlator |         | - Drive  |  |
|  | - Keystore |         | - KeepAlive  |         | - MIME/ANSI  |         | - Chunks |  |
|  +------------+         +--------------+         +--------------+         +----------+  |
+=========================================================================================+
         |                       |                        |                        |
         v                       v                        v                        v
+------------------+   +--------------------+   +-------------------+   +-----------------+
| Google Accounts  |   | Colab TFE Gateway  |   | Remote Colab VM   |   | Google Drive v3 |
| / OAuth 2.0      |   | (/tun/m/...)       |   | Jupyter Kernel    |   | API (Optional)  |
+------------------+   +--------------------+   +-------------------+   +-----------------+
```

For deep protocol details, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🚀 Installation & Quick Start

### 1. Install

Run directly from GitHub with `uvx`, or install via `pip`:

```bash
# Run directly with uvx (no manual clone required)
uvx --from git+https://github.com/Treamyracle/colab-mcp.git colab-mcp run

# Or install from GitHub via pip
pip install git+https://github.com/Treamyracle/colab-mcp.git

# Or install editable for local development (with browser CDP + test dependencies)
git clone https://github.com/Treamyracle/colab-mcp.git
cd colab-mcp
pip install -e ".[all]"
playwright install chromium
```

### 2. Configure Your AI Agent Host

#### Claude Desktop
Add `colab` to your `claude_desktop_config.json`:
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux:** `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "colab": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/Treamyracle/colab-mcp.git", "colab-mcp", "run"]
    }
  }
}
```

Or using a local Python environment:

```json
{
  "mcpServers": {
    "colab": {
      "command": "python",
      "args": ["-m", "colab_mcp", "run", "--transport", "stdio"]
    }
  }
}
```

#### Cursor / Antigravity / Windsurf / Cline
Add an MCP server entry with:
- **Server Name:** `colab`
- **Transport:** `stdio`
- **Command:** `colab-mcp` (or `python`)
- **Arguments:** `["run"]` (or `["-m", "colab_mcp", "run"]`)

---

## 🔑 Authentication Guide

Authenticate once via your AI agent (`colab_login`) or from the CLI:

### Method A: Remote OAuth 2.0 Flow (Recommended for Agents & Headless CLI)
```bash
colab-mcp login --method oauth_remote
```
1. Open the generated Google consent URL in any browser and approve access.
2. Paste the authorization code back into the prompt (or pass `auth_code=...` to `colab_login`).
3. Tokens are encrypted and persisted in your OS keyring.

### Method B: Local Loopback OAuth (Desktop)
```bash
colab-mcp login --method oauth_loopback
```

### Method C: Ambient Application Default Credentials (ADC)
```bash
colab-mcp login --method oauth_adc
```

### Method D: Chrome CDP Browser Fallback
For Google Workspace accounts or environments where OAuth clients are restricted:
```bash
colab-mcp login --method cdp_browser
```
Opens a real Google Chrome window (`--remote-debugging-port=9222`) for interactive login and 2FA, then extracts and encrypts the session cookies (`SAPISID`, `SID`, `OSID`).

---

## 🛠️ Complete MCP Tool Reference

`colab-mcp` exposes **19 MCP tools** across four categories:

### 1. Authentication Tools (3)

| Tool Name | Key Parameters | Description |
|---|---|---|
| `colab_login` | `method`, `auth_user_index`, `auth_code`, `cookies`, `timeout_seconds` | Authenticate with Google Colab via OAuth 2.0 (`oauth_remote`, `oauth_loopback`, `oauth_adc`), `cdp_browser`, or `cookie_import`. |
| `colab_get_auth_status` | `auth_user_index` (default `0`), `force_refresh` (`false`) | Query active account email, token validity, Colab subscription tier, CCU balance, and active assignment count. |
| `colab_logout` | `auth_user_index` (default `0`), `terminate_active_runtimes` (`true`) | Purge stored credentials from the OS keystore and optionally terminate active runtimes. |

### 2. Runtime & Session Management Tools (5)

| Tool Name | Key Parameters | Description |
|---|---|---|
| `colab_list_runtime_types` | `auth_user_index` (default `0`) | Query live accelerator eligibility (`NONE`, `T4`, `L4`, `A100`, `H100`, `V5E1`, `V6E1`), RAM shapes, and hourly CCU rates. |
| `colab_list_sessions` | `auth_user_index` (default `0`), `probe_kernels` (`true`) | List all active remote and locally tracked Colab VM sessions (including browser or prior-run sessions) with live kernel state. |
| `colab_connect_runtime` | `accelerator`, `high_ram`, `reuse_existing` (`true`), `auth_user_index`, `timeout_seconds` | Provision a new Colab VM or re-attach to an existing matching session and initialize keep-alive heartbeat. |
| `colab_get_runtime_status` | `runtime_id` (optional) | Inspect VM health, live kernel execution state, uptime, idle timer, and keep-alive status. |
| `colab_disconnect_runtime` | `runtime_id` (optional), `disconnect_all` (`false`), `auth_user_index` (`0`) | Terminate one local/remote Colab VM by `runtime_id` or all sessions (`disconnect_all=True`) via TFE `/unassign`. |

### 3. Code & Notebook Execution Tools (5)

| Tool Name | Key Parameters | Description |
|---|---|---|
| `colab_execute_code` | `code`, `runtime_id`, `timeout_seconds`, `wait_for_completion`, `store_history`, `max_text_lines`, `max_output_bytes`, `max_images` | Execute Python code or IPython magic commands (`!nvidia-smi`, `%timeit`); returns chronological `TextContent` and `ImageContent` blocks. |
| `colab_execute_notebook` | `notebook_path` \| `notebook_content`, `reset_kernel_before_run`, `cell_indices`, `stop_on_error`, `timeout_per_cell_seconds`, `output_save_path` | Parse (`nbformat` v4) and execute `.ipynb` notebooks sequentially or selectively with structured diagnostics and disk serialization. |
| `colab_get_execution_status` | `execution_id` | Poll execution progress, elapsed time, and partial stdout tail for async jobs (`wait_for_completion=False`). |
| `colab_get_cell_output` | `execution_id`, `include_images`, `max_text_lines`, `max_output_bytes`, `max_images` | Retrieve full buffered outputs in strict chronological emission order for an execution job. |
| `colab_interrupt_execution` | `runtime_id` (optional) | Send non-destructive SIGINT first; escalates to kernel restart after 5.0s if unresponsive, reporting `state_reset` and `kernel_restarted`. |

### 4. VM File Transfer & Google Drive Tools (6)

| Tool Name | Key Parameters | Description |
|---|---|---|
| `colab_list_files` | `remote_path` (default `"/content"`), `runtime_id` (optional) | List files and directories on the connected Colab VM. |
| `colab_download_file` | `remote_path`, `local_path`, `runtime_id` (optional) | Download any file (e.g., exported `.zip`, CSV, checkpoint) from the Colab VM to local disk with SHA-256 verification and automatic chunked fallback for large files. |
| `colab_upload_file` | `local_path`, `remote_path`, `runtime_id` (optional) | Upload a local file (dataset, `.zip` bundle, script) onto the connected Colab VM. |
| `colab_drive_list` | `folder` (default `"MyDrive"`), `query`, `max_results`, `auth_user_index`, `quota_project_id` | List files and folders in Google Drive via Drive API v3. |
| `colab_drive_download` | `file_id_or_path`, `local_path`, `runtime_path`, `auth_user_index`, `runtime_id`, `quota_project_id` | Download a file from Google Drive to local disk and/or directly onto the connected Colab VM. |
| `colab_drive_upload` | `local_path` \| `runtime_path`, `drive_folder`, `file_name`, `auth_user_index`, `runtime_id`, `quota_project_id` | Upload a file from local disk or directly from the Colab VM into Google Drive. |

> **Note on Google Drive in Headless Mode:** Colab's interactive `google.colab.drive.mount('/content/drive')` relies on a browser frontend dialog and will time out over headless WebSockets. Use `colab_upload_file` / `colab_download_file` for direct VM transfers, or configure `COLAB_MCP_GCP_PROJECT` (with Drive API enabled) to use `colab_drive_*` REST tools.

---

## 💻 CLI Reference (`colab-mcp`)

| Command | Example | Description |
|---|---|---|
| `run` | `colab-mcp run --transport stdio` | Start the MCP server (`stdio` or `sse` with `--host` and `--port`). |
| `login` | `colab-mcp login --method oauth_remote` | Authenticate a Google account and save encrypted credentials. |
| `status` | `colab-mcp status --force-refresh` | Show authentication status, tier, CCU balance, and active runtime state. |
| `list-runtimes` | `colab-mcp list-runtimes` | List available hardware accelerators and hourly CCU rates. |
| `list-sessions` | `colab-mcp list-sessions` | List all active remote and locally tracked Colab sessions. |
| `disconnect` | `colab-mcp disconnect --all` | Terminate a specific runtime (`--runtime-id`) or all sessions (`--all`). |
| `logout` | `colab-mcp logout` | Terminate active sessions and purge stored credentials. |
| `version` | `colab-mcp version` | Print the installed `colab-mcp` version. |

---

## ⚡ Hardware Accelerators & Pricing

| Accelerator ID | Hardware Device | VRAM / HBM | High-RAM (~51 GB) | Free Tier | Approx. Cost |
|---|---|---|---|---|---|
| `NONE` | 2 vCPU Intel Xeon | None | Pro/Pro+ only | Yes | 0.00 CCU/hr |
| `T4` | NVIDIA Tesla T4 | 16 GB GDDR6 | Pro/Pro+ only | Yes (Dynamic) | ~1.96 CCU/hr |
| `L4` | NVIDIA L4 Tensor Core | 24 GB GDDR6 | Supported | Pro/Pro+ | ~3.84 CCU/hr |
| `A100` | NVIDIA A100-SXM4 | 40 GB / 80 GB | Supported | Pro/Pro+ | ~5.42 - 11.2 CCU/hr |
| `H100` | NVIDIA H100-SXM5 | 80 GB HBM3 | Supported | Pro+ | ~13.50 CCU/hr |
| `V5E1` | Cloud TPU v5e | 16 GB HBM | Standard Shape | Pro/Pro+ | ~1.96 CCU/hr |
| `V6E1` | Cloud TPU v6e | 32 GB HBM | Standard Shape | Pro+ | ~4.50 CCU/hr |

---

## 🛡️ Security Architecture

1. **Credential Isolation & Egress Redaction:** OAuth tokens (`ya29...`, `1//0...`), cookies (`SAPISID`, `OSID`, `SID`), and `SAPISIDHASH` signatures are stripped by a global egress redaction filter before reaching LLM context or logs.
2. **Encrypted Keystore:** Secrets use OS-native vaults (Windows DPAPI, macOS Keychain, Linux Secret Service) with an AES-256-GCM encrypted file fallback (`0600`).
3. **Cloud Execution Boundary:** All Python/notebook code executes inside Google's Colab containers—not on your local host.
4. **Compute Unit Watchdog:** Background 30-minute inactivity auto-unassign protects paid CCUs while preserving active `busy` training jobs.
5. **Path & Terminal Hardening:** Path traversal guards protect local and remote file operations, and ECMA-48 ANSI stripping sanitizes kernel tracebacks.

---

## 🤝 Contributing & Development

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

```bash
# Run offline unit, integration, and ColabTwin opaque E2E suites (no Google login required)
pytest tests/unit/ tests/integration/ tests/e2e/opaque/ -q
```

- [Architecture Reference](docs/ARCHITECTURE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)
- [Changelog](CHANGELOG.md)

---

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.
