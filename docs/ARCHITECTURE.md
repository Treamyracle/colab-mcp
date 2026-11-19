# Google Colab Model Context Protocol (MCP) Server: Technical Architecture Reference

**Document Version:** 1.0.0  
**Status:** Implemented Baseline Reference  
**Target Platform:** Consumer Google Colab (`colab.research.google.com`)  
**Specification Standards:** Model Context Protocol (MCP 2024-11-05), Jupyter Messaging Protocol v5.3/v5.4, Jupyter Notebook Format (nbformat v4)

> **Note:** This document served as the original engineering blueprint for `colab-mcp`. All core subsystems described here are implemented, with additional production capabilities (including `colab_list_sessions`, live Colab TFE/WebSocket protocol fixes, and the 6 VM file transfer & Google Drive tools in `src/colab_mcp/files/`) documented in [README.md](../README.md).

---

## Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
   - 1.1 [Executive Summary](#11-executive-summary)
   - 1.2 [Consumer Colab Operational Constraints](#12-consumer-colab-operational-constraints)
   - 1.3 [AI Agent Interaction Patterns & Motivation](#13-ai-agent-interaction-patterns--motivation)
   - 1.4 [Requirements Traceability Matrix](#14-requirements-traceability-matrix)
2. [Architecture Overview](#2-architecture-overview)
   - 2.1 [End-to-End System Architecture](#21-end-to-end-system-architecture)
   - 2.2 [Component & Data Flow Diagrams](#22-component--data-flow-diagrams)
   - 2.3 [Modular Subsystem Decomposition](#23-modular-subsystem-decomposition)
   - 2.4 [Subsystem Interface Contracts](#24-subsystem-interface-contracts)
3. [Consumer Colab Integration & Authentication](#3-consumer-colab-integration--authentication)
   - 3.1 [Authentication Challenges with Consumer Accounts](#31-authentication-challenges-with-consumer-accounts)
   - 3.2 [Primary: Google OAuth 2.0 Protocol Implementation](#32-primary-google-oauth-20-protocol-implementation)
     - 3.2.1 [OAuth Scopes & Client Credentials](#321-oauth-scopes--client-credentials)
     - 3.2.2 [Remote Copy-Paste Authorization Flow](#322-remote-copy-paste-authorization-flow)
     - 3.2.3 [Local Loopback Server Flow](#323-local-loopback-server-flow)
     - 3.2.4 [Application Default Credentials (ADC) Ingestion](#324-application-default-credentials-adc-ingestion)
     - 3.2.5 [Token Lifecycle & Automatic Refresh](#325-token-lifecycle--automatic-refresh)
   - 3.3 [Secondary: Browser-Assisted Login via Chrome DevTools Protocol (CDP)](#33-secondary-browser-assisted-login-via-chrome-devtools-protocol-cdp)
     - 3.3.1 [Bypassing Google Headless Anti-Bot Detection](#331-bypassing-google-headless-anti-bot-detection)
     - 3.3.2 [Chrome Profile Launch & Cookie Extraction Pipeline](#332-chrome-profile-launch--cookie-extraction-pipeline)
     - 3.3.3 [Google 1P API Request Signing via SAPISIDHASH](#333-google-1p-api-request-signing-via-sapisidhash)
   - 3.4 [Credential Persistence & OS Keyring Integration](#34-credential-persistence--os-keyring-integration)
     - 3.4.1 [OS-Native Keystores (DPAPI, Keychain, Secret Service)](#341-os-native-keystores-dpapi-keychain-secret-service)
     - 3.4.2 [Encrypted Headless File Fallback (AES-256-GCM)](#342-encrypted-headless-file-fallback-aes-256-gcm)
4. [Runtime Management & Colab Tunnel Frontend (TFE) Protocol](#4-runtime-management--colab-tunnel-frontend-tfe-protocol)
   - 4.1 [Tunnel Frontend (TFE) Topology & Endpoints](#41-tunnel-frontend-tfe-topology--endpoints)
   - 4.2 [Notebook Hash (`nbh`) Derivation Algorithm](#42-notebook-hash-nbh-derivation-algorithm)
   - 4.3 [Hardware Accelerator Matrix & Machine Shapes](#43-hardware-accelerator-matrix--machine-shapes)
   - 4.4 [Two-Step Assignment & Provisioning Handshake](#44-two-step-assignment--provisioning-handshake)
   - 4.5 [Session Enumeration & Compute Unit (CCU) Tracking](#45-session-enumeration--compute-unit-ccu-tracking)
   - 4.6 [Session Teardown & Resource Release (`unassign`)](#46-session-teardown--resource-release-unassign)
   - 4.7 [Keep-Alive Heartbeat Daemon & Watchdog Inactivity Auto-Unassign](#47-keep-alive-heartbeat-daemon--watchdog-inactivity-auto-unassign)
5. [Jupyter Kernel Communication & WebSocket Protocol](#5-jupyter-kernel-communication--websocket-protocol)
   - 5.1 [Colab Kernel Proxy Architecture & Kernel Discovery Handshake](#51-colab-kernel-proxy-architecture--kernel-discovery-handshake)
   - 5.2 [WebSocket Handshake, Subprotocol Fix & Protocol Compatibility](#52-websocket-handshake-subprotocol-fix--protocol-compatibility)
   - 5.3 [Jupyter Messaging Protocol v5.3 Multiplexing](#53-jupyter-messaging-protocol-v53-multiplexing)
   - 5.4 [Execution State Tracking & Message Correlator](#54-execution-state-tracking--message-correlator)
   - 5.5 [Multimodal MIME Output Processing, Chronological Ordering & ANSI Sanitization](#55-multimodal-mime-output-processing-chronological-ordering--ansi-sanitization)
   - 5.6 [Out-of-Band Execution Interruption & Kernel Restart Distinction](#56-out-of-band-execution-interruption--kernel-restart-distinction)
6. [Notebook Execution Engine](#6-notebook-execution-engine)
   - 6.1 [nbformat v4 Parsing & Schema Validation](#61-nbformat-v4-parsing--schema-validation)
   - 6.2 [Sequential & Selective Cell Index Execution](#62-sequential--selective-cell-index-execution)
   - 6.3 [Error Handling Policies & State Propagation](#63-error-handling-policies--state-propagation)
   - 6.4 [In-Memory Execution State & Disk Serialization](#64-in-memory-execution-state--disk-serialization)
7. [MCP Server Specification & Tool Schemas](#7-mcp-server-specification--tool-schemas)
   - 7.1 [Technology Stack Justification: Python MCPServer vs TypeScript](#71-technology-stack-justification-python-mcpserver-vs-typescript)
   - 7.2 [Dual Transport Engine: Stdio & Server-Sent Events (SSE)](#72-dual-transport-engine-stdio--server-sent-events-sse)
     - 7.2.1 [Stdio Transport & Absolute Stdout Isolation](#721-stdio-transport--absolute-stdout-isolation)
     - 7.2.2 [SSE Transport for Remote/Hub Deployments](#722-sse-transport-for-remotehub-deployments)
   - 7.3 [Complete JSON Schemas for 12 MCP Tools](#73-complete-json-schemas-for-12-mcp-tools)
     - 7.3.1 [Authentication Tools (colab_login, colab_get_auth_status, colab_logout)](#731-authentication-tools)
     - 7.3.2 [Runtime Tools (colab_list_runtime_types, colab_connect_runtime, colab_get_runtime_status, colab_disconnect_runtime)](#732-runtime-tools)
     - 7.3.3 [Execution Tools (colab_execute_code, colab_execute_notebook, colab_get_execution_status, colab_get_cell_output, colab_interrupt_execution)](#733-execution-tools)
8. [Security Architecture & Threat Mitigations](#8-security-architecture--threat-mitigations)
   - 8.1 [Credential Redaction & Zero-Leakage Policy](#81-credential-redaction--zero-leakage-policy)
   - 8.2 [Compute Unit Runaway Protection & Watchdog Inactivity Auto-Unassign](#82-compute-unit-runaway-protection--watchdog-inactivity-auto-unassign)
   - 8.3 [Sandbox Isolation & Local Filesystem Boundaries](#83-sandbox-isolation--local-filesystem-boundaries)
   - 8.4 [Terminal Escape Injection Sanitization](#84-terminal-escape-injection-sanitization)
9. [Phased Implementation Roadmap](#9-phased-implementation-roadmap)
   - 9.1 [Roadmap Summary & Milestone Dependencies](#91-roadmap-summary--milestone-dependencies)
   - 9.2 [Milestone M1: Authentication & Credential Storage](#92-milestone-m1-authentication--credential-storage)
   - 9.3 [Milestone M2: Runtime Lifecycle & Provisioning Engine](#93-milestone-m2-runtime-lifecycle--provisioning-engine)
   - 9.4 [Milestone M3: Kernel WebSocket Bridge & Output Pipeline](#94-milestone-m3-kernel-websocket-bridge--output-pipeline)
   - 9.5 [Milestone M4: Notebook Orchestration Engine](#95-milestone-m4-notebook-orchestration-engine)
   - 9.6 [Milestone M5: Transports, CLI Packaging & Hardening](#96-milestone-m5-transports-cli-packaging--hardening)

---

## 1. Executive Summary & Problem Statement

### 1.1 Executive Summary
The Model Context Protocol (MCP) has emerged as the open industry standard for connecting AI coding assistants (such as Claude Desktop, Cursor, Antigravity, Cline, and Roo-Code) with external tool ecosystems and execution runtimes. However, existing MCP execution servers run exclusively on the developer's local machine or within transient, unaccelerated Docker containers. Machine learning engineering, data science pipelines, deep learning research, and GPU-intensive tasks require high-performance hardware accelerators (NVIDIA T4, L4, A100, H100 GPUs and Google Cloud TPUs) that are unavailable on typical developer workstations.

Google Colab (`colab.research.google.com`) is the dominant cloud computing platform for interactive data science and deep learning, providing hosted Jupyter runtimes with complimentary and pay-as-you-go GPU/TPU accelerators. However, Google Colab is architected as a consumer web application with no documented public REST API for non-browser automation.

This document specifies the complete, production-grade technical architecture and implementation roadmap for `colabmcp`—a Google Colab Model Context Protocol Server. The server bridges autonomous AI agents with consumer Google Colab, enabling agents to authenticate, provision hardware-accelerated cloud VMs, execute Jupyter notebooks and arbitrary Python snippets, monitor execution progress in real time, and capture multimodal outputs (stdout, stderr, Matplotlib plots, error tracebacks) without manual human intervention in a browser tab.

### 1.2 Consumer Colab Operational Constraints
Interfacing with consumer Google Colab introduces fundamental challenges distinct from enterprise Google Cloud Platform (GCP) Vertex AI Workbench:
1. **Absence of Public API Gateway:** Unlike Vertex AI, consumer Colab has no gRPC/REST SDK under `cloud.google.com`. Runtimes are provisioned through Google's internal Tunnel Frontend (`colab.research.google.com/tun/m/...`).
2. **Aggressive Idle & Session Caps:** Colab VM instances automatically terminate after 90 minutes of inactivity. Additionally, free-tier runtimes enforce a 12-hour continuous hard cap (24 hours for Colab Pro/Pro+), after which all ephemeral filesystem data and in-memory state are permanently wiped.
3. **Anti-Bot & Identity Verification:** Google Accounts authentication actively detects and blocks automated headless browsers (e.g., standard Playwright/Puppeteer Chromium) with *"This browser or app may not be secure"*.
4. **Compute Unit Metering:** Colab Pro and Pay-as-you-go tiers consume Colab Compute Units (CCU) at variable hourly burn rates depending on the selected accelerator. Runaway loops or orphaned sessions rapidly deplete paid credits.
5. **Jupyter WebSocket Tunneling Quirks:** Remote kernel connectivity requires authenticating via short-lived proxy tokens (3,600-second TTL) over multiplexed WebSockets that reject standard Jupyter subprotocol negotiation if improperly configured.

### 1.3 AI Agent Interaction Patterns & Motivation
Autonomous AI coding agents operate through iterative code execution, evaluation, and repair loops. A typical interaction pattern requires:
1. **Dynamic Environment Discovery:** Querying available GPU/TPU accelerators and attaching to an appropriate runtime.
2. **Stateful Execution:** Executing sequential code blocks where variables, model weights, and imported libraries persist across cell executions.
3. **Progress Telemetry:** Monitoring long-running tasks (e.g., model fine-tuning, data preprocessing) via non-blocking status polling.
4. **Multimodal Feedback:** Inspecting generated visual outputs (loss curves, confusion matrices, data visualizations) via base64 image blocks to evaluate model convergence.
5. **Autonomous Error Recovery:** Receiving clean, ANSI-sanitized tracebacks when code fails, enabling the LLM to diagnose root causes and dispatch corrective code.

### 1.4 Requirements Traceability Matrix
This specification comprehensively fulfills all core requirements:

| ID | Requirement Name | Source | Specification Coverage |
|---|---|---|---|
| **R1** | Technical Architecture & Implementation Plan | ORIGINAL_REQUEST.md | Sections 2, 7.1, 8, 9 |
| **R1.1** | Component architecture & ASCII flowcharts | ORIGINAL_REQUEST.md | Sections 2.1, 2.2, 2.3 |
| **R1.2** | Technology stack & MCP SDK selection justification | ORIGINAL_REQUEST.md | Section 7.1 |
| **R1.3** | Security & credential management design | ORIGINAL_REQUEST.md | Sections 3.4, 8.1, 8.2 |
| **R1.4** | Phased implementation roadmap (M1 to M5) | ORIGINAL_REQUEST.md | Section 9 |
| **R2** | Consumer Colab Integration & Runtime Management | ORIGINAL_REQUEST.md | Sections 3, 4 |
| **R2.1** | Google account authentication (OAuth 2.0 & CDP fallback) | ORIGINAL_REQUEST.md | Sections 3.2, 3.3 |
| **R2.2** | Session lifecycle management & TFE assignment | ORIGINAL_REQUEST.md | Sections 4.4, 4.6 |
| **R2.3** | Hardware accelerator selection (CPU, GPU, TPU, High-RAM) | ORIGINAL_REQUEST.md | Section 4.3 |
| **R2.4** | Keep-alive heartbeat & idle timeout mitigation | ORIGINAL_REQUEST.md | Section 4.7 |
| **R3** | Notebook Execution & MCP Interface Specification | ORIGINAL_REQUEST.md | Sections 5, 6, 7 |
| **R3.1** | Load and execute `.ipynb` notebooks and single cells | ORIGINAL_REQUEST.md | Sections 6.1, 6.2, 7.3.3 |
| **R3.2** | Real-time & polling execution progress tracking | ORIGINAL_REQUEST.md | Sections 5.4, 7.3.3 |
| **R3.3** | Multimodal output capture (stdout, stderr, rich MIME, ANSI tracebacks) | ORIGINAL_REQUEST.md | Sections 5.5, 7.3.3 |
| **R3.4** | Complete JSON Schemas for 12 MCP tools | ORIGINAL_REQUEST.md | Section 7.3 |

---

## 2. Architecture Overview

### 2.1 End-to-End System Architecture
The `colabmcp` system serves as a deterministic protocol bridge between external AI agent hosts and Google Colab's distributed container infrastructure:

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
|  |   - Request schema validation (Pydantic v2 / JSON Schema Draft 7)                 |  |
|  |   - Active session store & execution task registry                                |  |
|  |   - Credential redaction filter & security boundary                               |  |
|  +-----------------------------------------------------------------------------------+  |
|            |                                |                               |           |
|            v                                v                               v           |
|  +--------------------+          +--------------------+          +--------------------+ |
|  |    Auth Module     |          |   Runtime Engine   |          |   Kernel Engine    | |
|  | - OAuth 2.0 Client |          | - TFE REST Client  |          | - WebSocket Client | |
|  | - CDP Automation   |          | - Session Manager  |          | - Jupyter v5.3 Wire| |
|  | - OS Keyring / Enc |          | - Keep-Alive Task  |          | - MIME Converter   | |
|  +--------------------+          +--------------------+          +--------------------+ |
|                                                                             |           |
|                                                                             v           |
|                                                                  +--------------------+ |
|                                                                  |  Notebook Runner   | |
|                                                                  | - nbformat v4 AST  | |
|                                                                  | - Cell Sequencer   | |
|                                                                  | - Error Propagator | |
|                                                                  +--------------------+ |
+=========================================================================================+
            |                                           |                     |
            | HTTPS Bearer / Cookies                    | HTTPS TFE REST      | WSS Proxy
            v                                           v                     v
+------------------------+                 +------------------------+   +-----------------+
|  Google Accounts /     |                 |  Colab Tunnel Frontend |   | Remote Colab VM |
|  OAuth 2.0 Endpoint    |                 |  (TFE /tun/m/...)      |   | Jupyter Kernel  |
|  (accounts.google.com) |                 |  (Allocation & Status) |   | (IPython Core)  |
+------------------------+                 +------------------------+   +-----------------+
```

### 2.2 Component & Data Flow Diagrams

#### Detailed Execution Sequence Flow:
```
AI Agent              FastMCP Server            Runtime Engine          Colab TFE Gateway         Remote Kernel
   |                         |                         |                        |                       |
   |--- 1. colab_login ----->|                         |                        |                       |
   |    (OAuth or CDP)       |--- Authenticate ------->|                        |                       |
   |<-- Auth Confirmed ------|<-- Bearer / Cookies ----|                        |                       |
   |                         |                         |                        |                       |
   |--- 2. colab_connect --->|                         |                        |                       |
   |    (acc='GPU_T4')       |--- Request VM --------->|--- GET /tun/m/assign ->|                       |
   |                         |                         |<-- XSRF Token ---------|                       |
   |                         |                         |--- POST /tun/m/assign->|                       |
   |                         |                         |<-- RuntimeProxyInfo ---|                       |
   |                         |                         |    (endpoint, token)   |                       |
   |                         |                         |                        |                       |
   |                         |--- Connect WS ---------------------------------------------------------->|
   |                         |    (X-Colab-Runtime-Proxy-Token, subprotocols=None)                      |
   |                         |<-- WS 101 Switching Protocols -------------------------------------------|
   |<-- Runtime Connected ---|                         |                        |                       |
   |    (T4, 16GB VRAM)      |                         |                        |                       |
   |                         |                         |-- (Start Keep-Alive)-> |                       |
   |                         |                         |   every 35s GET        |                       |
   |                         |                         |   /tun/m/keep-alive/   |                       |
   |                         |                         |                        |                       |
   |--- 3. execute_code ---->|                         |                        |                       |
   |    (import torch...)    |--- send execute_request (shell) ---------------------------------------->|
   |                         |                         |                        |                       |
   |                         |<-- status: busy (iopub) -------------------------------------------------|
   |                         |<-- stream: stdout (iopub) -----------------------------------------------|
   |                         |<-- display_data: image/png (iopub) --------------------------------------|
   |                         |<-- status: idle (iopub) -------------------------------------------------|
   |                         |<-- execute_reply: ok (shell) --------------------------------------------|
   |                         |                         |                        |                       |
   |                         |--- Process MIME & Clean ANSI Traceback           |                       |
   |<-- Return MCP Result ---|                         |                        |                       |
   |    [TextContent,        |                         |                        |                       |
   |     ImageContent]       |                         |                        |                       |
```

### 2.3 Modular Subsystem Decomposition
The `colabmcp` codebase is structured into five cohesive subsystems, following strict separation of concerns:

```
src/colab_mcp/
├── __init__.py                # Version metadata & public API exports
├── __main__.py                # Python execution entrypoint (`python -m colab_mcp`)
├── cli.py                     # Typer CLI application (commands: run, login, status, list-runtimes)
├── server.py                  # FastMCP server definition, lifespan manager & tool registrations
├── auth/                      # Subsystem 1: Identity, OAuth, CDP & Credential Storage
│   ├── __init__.py
│   ├── models.py              # Pydantic models: GoogleCredentials, AuthStatus, AuthMethod
│   ├── oauth.py               # Google OAuth 2.0 flows (Remote copy-paste, Loopback, ADC)
│   ├── cdp_browser.py         # Playwright Chromium CDP automation & session cookie extractor
│   ├── sapisid.py             # SAPISIDHASH request signing header calculator
│   └── keystore.py            # OS Keyring interface with AES-256-GCM encrypted file fallback
├── runtime/                   # Subsystem 2: Colab Tunnel Frontend (TFE) & VM Lifecycle
│   ├── __init__.py
│   ├── models.py              # Enums: Variant, Accelerator, Shape; Pydantic: Assignment, CCUInfo
│   ├── client.py              # TFE HTTP REST client (assign, unassign, assignments, ccu-info)
│   ├── keepalive.py           # Background asyncio keep-alive heartbeat worker
│   └── session_manager.py     # Session pool, proxy token cache, auto-reconnect coordinator
├── kernel/                    # Subsystem 3: Jupyter Kernel WebSocket Bridge & Output Engine
│   ├── __init__.py
│   ├── models.py              # Jupyter wire protocol envelopes & message types
│   ├── client.py              # Multiplexed WebSocket client (channels: shell, iopub, stdin, control)
│   ├── correlator.py          # Asynchronous msg_id tracking & event streaming coordinator
│   └── mime_converter.py      # Multimodal MIME parser (TextContent, ImageContent) & ANSI cleaner
├── notebook/                  # Subsystem 4: Notebook (.ipynb) Orchestration Engine
│   ├── __init__.py
│   ├── loader.py              # nbformat v4 parser, JSON validator, schema normalizer
│   └── runner.py              # Sequential & selective cell index runner with stop_on_error
└── utils/                     # Subsystem 5: Shared Cross-Cutting Utilities
    ├── __init__.py
    ├── logging.py             # Stderr-only logger guaranteeing zero stdout pollution
    └── nbh.py                 # 44-character web-safe base64 notebook hash (nbh) generator
```

### 2.4 Subsystem Interface Contracts

#### Contract 1: `colab_mcp.auth` ➔ `colab_mcp.runtime`
```python
class GoogleCredentials(BaseModel):
    auth_method: Literal["oauth2", "cdp_cookies", "adc"]
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expiry: Optional[datetime] = None
    cookies: Dict[str, str] = Field(default_factory=dict)
    sapisid_hash: Optional[str] = None

async def get_active_credentials() -> GoogleCredentials:
    """
    Returns valid credentials for Colab TFE requests.
    Transparently refreshes expired OAuth access tokens via refresh_token.
    Raises AuthenticationRequiredError if no valid credentials exist.
    """
```

#### Contract 2: `colab_mcp.runtime` ➔ `colab_mcp.kernel`
```python
class KernelConnectionParams(BaseModel):
    endpoint: str
    proxy_url: str              # e.g., "https://colab.research.google.com/tun/m/colab-xxxx"
    proxy_token: str            # Short-lived 3600s JWT proxy token
    token_expires_at: datetime
    kernel_id: str              # Discovered or newly initialized Jupyter kernel UUID
    client_agent: str = "colab-mcp"

async def get_kernel_connection_params(runtime_id: Optional[str] = None) -> KernelConnectionParams:
    """
    Retrieves or establishes active VM tunnel parameters and kernel binding.
    Verifies proxy_token validity; automatically queries GET /tun/m/assignments if expired.
    Discovers existing kernel via GET /api/kernels or creates one via POST /api/kernels.
    Raises RuntimeDisconnectedError if backend VM has been terminated.
    """
```

#### Contract 3: `colab_mcp.kernel` ➔ `colab_mcp.server`
```python
class ExecutionEvent(BaseModel):
    event_type: Literal["started", "stream", "display_data", "completed", "error"]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stream_name: Optional[Literal["stdout", "stderr"]] = None
    text: Optional[str] = None
    mime_type: Optional[str] = None
    data_base64: Optional[str] = None
    execution_count: Optional[int] = None
    error_name: Optional[str] = None
    error_value: Optional[str] = None
    sanitized_traceback: Optional[str] = None

async def execute_snippet(
    code: str,
    timeout_seconds: int = 300,
    store_history: bool = True
) -> AsyncIterator[ExecutionEvent]:
    """
    Asynchronously streams execution events across shell and iopub channels.
    Yields ExecutionEvent chunks in real-time.
    """
```

---

## 3. Consumer Colab Integration & Authentication

### 3.1 Authentication Challenges with Consumer Accounts
Consumer Google Colab accounts do not authenticate via GCP Service Account JSON keys. The Colab Tunnel Frontend (TFE) requires valid consumer identity credentials. Two authentication mechanisms are supported:
1. **Google OAuth 2.0 (Primary):** Clean, tokenized, non-interactive refresh support, restricted scopes.
2. **Browser-Assisted CDP Cookie Extraction (Secondary / Fallback):** For environments where OAuth client creation is restricted or enterprise Workspace policies forbid third-party OAuth apps.

### 3.2 Primary: Google OAuth 2.0 Protocol Implementation

#### 3.2.1 OAuth Scopes & Client Credentials
To interact with consumer Colab and execute code in Google's cloud containers, the OAuth token must be granted the following authoritative scopes:
```python
COLAB_OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",
]
```

- **Client ID Identification:** By default, `colabmcp` leverages Google's registered Cloud SDK public OAuth client ID (`764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com`), which has pre-approved authorization for `https://www.googleapis.com/auth/colaboratory`. Alternatively, users can configure a custom OAuth 2.0 Desktop Client ID and Secret in `~/.colabmcp/config.json`.

#### 3.2.2 Remote Copy-Paste Authorization Flow
The remote copy-paste flow is the primary authentication path for terminal-based AI agents and remote headless servers. It avoids local port binding dependencies:
```python
from google_auth_oauthlib.flow import InstalledAppFlow

REMOTE_REDIRECT_URI = "https://sdk.cloud.google.com/applicationdefaultauthcode.html"

def initiate_remote_oauth_flow(client_config: dict) -> tuple[str, InstalledAppFlow]:
    flow = InstalledAppFlow.from_client_config(
        client_config,
        scopes=COLAB_OAUTH_SCOPES,
        redirect_uri=REMOTE_REDIRECT_URI
    )
    # prompt="consent" forces Google to return a refresh_token
    # token_usage="remote" activates copy-paste authorization code landing page
    auth_url, _ = flow.authorization_url(
        prompt="consent",
        token_usage="remote",
        access_type="offline"
    )
    return auth_url, flow

def exchange_remote_auth_code(flow: InstalledAppFlow, auth_code: str) -> dict:
    flow.fetch_token(code=auth_code.strip())
    credentials = flow.credentials
    return {
        "access_token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_expiry": credentials.expiry.isoformat() if credentials.expiry else None,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "token_uri": credentials.token_uri
    }
```

#### 3.2.3 Local Loopback Server Flow
For interactive desktop environments (macOS/Windows/Linux Desktop), `colabmcp` can spawn a transient loopback webserver:
```python
def run_loopback_oauth_flow(client_config: dict) -> dict:
    flow = InstalledAppFlow.from_client_config(
        client_config,
        scopes=COLAB_OAUTH_SCOPES
    )
    # Binds to an ephemeral high-numbered port on 127.0.0.1
    credentials = flow.run_local_server(
        host="127.0.0.1",
        port=0,
        authorization_prompt_message="Opening browser for Google Colab authentication...",
        success_message="Authentication complete! You may close this tab and return to your agent.",
        open_browser=True
    )
    return {
        "access_token": credentials.token,
        "refresh_token": credentials.refresh_token,
        "token_expiry": credentials.expiry.isoformat() if credentials.expiry else None,
        "client_id": credentials.client_id,
        "client_secret": credentials.client_secret,
        "token_uri": credentials.token_uri
    }
```

#### 3.2.4 Application Default Credentials (ADC) Ingestion
If the user environment already has authenticated Google Cloud SDK credentials (`gcloud auth application-default login`), `colabmcp` automatically detects and ingests them:
```python
import google.auth
from google.auth.transport.requests import Request

def load_ambient_adc() -> Optional[dict]:
    try:
        credentials, project = google.auth.default(scopes=COLAB_OAUTH_SCOPES)
        if not credentials.valid:
            credentials.refresh(Request())
        return {
            "access_token": credentials.token,
            "refresh_token": getattr(credentials, "refresh_token", None),
            "token_expiry": credentials.expiry.isoformat() if credentials.expiry else None,
            "is_adc": True
        }
    except Exception:
        return None
```

#### 3.2.5 Token Lifecycle & Automatic Refresh
OAuth 2.0 access tokens expire after 3,600 seconds (1 hour). `colabmcp` implements proactive token refreshing with robust error handling for revoked or expired refresh tokens:
```python
from datetime import datetime, timedelta
import google.oauth2.credentials
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError

async def ensure_valid_token(creds_data: dict) -> str:
    creds = google.oauth2.credentials.Credentials(
        token=creds_data["access_token"],
        refresh_token=creds_data["refresh_token"],
        token_uri=creds_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=COLAB_OAUTH_SCOPES
    )
    
    # Proactively refresh if expiring within 5 minutes
    if not creds.valid or (creds.expiry and creds.expiry - datetime.utcnow() < timedelta(minutes=5)):
        try:
            creds.refresh(Request())
            # Update keystore with refreshed access_token and expiry
            await update_stored_credentials(creds)
        except RefreshError as e:
            # Token has been revoked, session expired, or user changed credentials
            await purge_stored_credentials()
            raise AuthenticationRequiredError(
                f"Google OAuth refresh token expired or was revoked: {e}. "
                "Please re-authenticate using colab_login."
            ) from e
        
    return creds.token
```

### 3.3 Secondary: Browser-Assisted Login via Chrome DevTools Protocol (CDP)

#### 3.3.1 Bypassing Google Headless Anti-Bot Detection
Standard headless browser automation (Playwright/Puppeteer with default headless mode) fails against `accounts.google.com` due to runtime bot detection heuristics:
- Flags `navigator.webdriver = true`.
- Evaluates missing hardware acceleration / WebGL vendor strings.
- Detects missing audio codecs and automated Chromium fingerprints.
- Outcome: Google halts login with *"This browser or app may not be secure. Learn more. Try using a different browser."*

**The Solution:** `colabmcp` uses a **Real Headed Chrome CDP Bridge**. It executes the user's installed Google Chrome binary in a dedicated user profile with remote debugging enabled (`--remote-debugging-port=9222 --user-data-dir=~/.colabmcp/chrome_profile`). Playwright attaches to this running process via `chromium.connect_over_cdp("http://127.0.0.1:9222")`. Because it is an authentic, hardware-accelerated Chrome instance, Google's bot detection does not trigger, enabling standard 2FA, SMS, and passkey authentication.

#### 3.3.2 Chrome Profile Launch & Cookie Extraction Pipeline
To prevent Chrome diagnostic, GPU, and WebRTC output from leaking into `stdout` and corrupting MCP JSON-RPC protocol framing in `stdio` mode, the browser process MUST be launched with `stdout=subprocess.DEVNULL`, `stderr=subprocess.DEVNULL`, and `close_fds=True`. Furthermore, launch race conditions are eliminated by polling the CDP JSON endpoint (`http://127.0.0.1:9222/json/version`) before attempting WebSocket attachment.

```python
import os
import sys
import shutil
import subprocess
import asyncio
import httpx
from playwright.async_api import async_playwright

REQUIRED_COLAB_COOKIES = [
    "SID", "HSID", "SSID", "APISID", "SAPISID",
    "__Secure-1PSID", "__Secure-3PSID", "OSID", "__Secure-OSID"
]

def find_chrome_executable() -> str:
    """Discovers installed Google Chrome binary across Windows, macOS, and Linux."""
    if sys.platform == "win32":
        candidates = [
            os.path.join(os.environ.get("PROGRAMFILES", "C:\\Program Files"), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)"), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
    elif sys.platform == "darwin":
        mac_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.isfile(mac_path):
            return mac_path
    else:  # Linux / Unix
        for binary in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
            found = shutil.which(binary)
            if found:
                return found

    raise FileNotFoundError("Google Chrome executable not found. Please install Chrome or authenticate via OAuth 2.0.")

async def wait_for_cdp_ready(port: int = 9222, max_retries: int = 30, interval: float = 0.2) -> bool:
    """Polls http://127.0.0.1:{port}/json/version to eliminate browser launch race conditions."""
    url = f"http://127.0.0.1:{port}/json/version"
    async with httpx.AsyncClient(timeout=1.0) as client:
        for _ in range(max_retries):
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(interval)
    return False

async def launch_and_extract_cdp_cookies(timeout_seconds: int = 180) -> dict[str, str]:
    chrome_path = find_chrome_executable()
    user_data_dir = os.path.expanduser("~/.colabmcp/chrome_profile")
    os.makedirs(user_data_dir, exist_ok=True)

    chrome_cmd = [
        chrome_path,
        "--remote-debugging-port=9222",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "https://colab.research.google.com"
    ]

    # Critical: Isolate stdout/stderr to DEVNULL to avoid corrupting stdio MCP transport
    proc = subprocess.Popen(
        chrome_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True
    )

    try:
        # Await CDP readiness prior to Playwright connection
        if not await wait_for_cdp_ready(port=9222):
            raise RuntimeError("Timed out waiting for Chrome DevTools Protocol port 9222 to become active.")

        async with async_playwright() as p:
            # Connect to real Chrome via Chrome DevTools Protocol
            browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()

            start_time = asyncio.get_event_loop().time()
            extracted_cookies = {}

            # Extraction loop: wait until navigation arrives at Colab and OSID is present
            while asyncio.get_event_loop().time() - start_time < timeout_seconds:
                # Confirm page is on Colab origin
                current_url = page.url
                if current_url.startswith("https://colab.research.google.com"):
                    cookies = await context.cookies(["https://colab.research.google.com", "https://google.com"])
                    cookie_dict = {c["name"]: c["value"] for c in cookies}

                    # Verify OSID is present alongside SAPISID and SID (required for 1P Colab origin requests)
                    if "SAPISID" in cookie_dict and "SID" in cookie_dict and "OSID" in cookie_dict:
                        extracted_cookies = cookie_dict
                        break

                await asyncio.sleep(2)

            await browser.close()

            if not extracted_cookies:
                raise TimeoutError("Login timed out before authenticated Colab cookies (including OSID) were detected.")

            return extracted_cookies
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
```

#### 3.3.3 Google 1P API Request Signing via SAPISIDHASH
When authenticating via extracted session cookies instead of OAuth Bearer tokens, Colab internal endpoints require an `Authorization` header calculated using Google's client-side `SAPISIDHASH` algorithm with unix epoch seconds (`int(time.time())`):
```python
import time
import hashlib

def generate_sapisid_hash(sapisid_cookie: str, origin: str = "https://colab.research.google.com") -> str:
    """
    Computes Google 1P authorization header using unix epoch seconds:
    Authorization: SAPISIDHASH <timestamp_sec>_<sha1(timestamp_sec + ' ' + sapisid + ' ' + origin)>
    """
    timestamp_sec = int(time.time())
    payload = f"{timestamp_sec} {sapisid_cookie} {origin}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"SAPISIDHASH {timestamp_sec}_{digest}"
```

### 3.4 Credential Persistence & OS Keyring Integration

#### 3.4.1 OS-Native Keystores (DPAPI, Keychain, Secret Service)
Credentials must never be stored in unencrypted plaintext on disk. `colabmcp` uses the Python `keyring` library to bind credentials to OS hardware-backed cryptographic vaults:
- **Windows:** Windows Credential Manager via Data Protection API (DPAPI).
- **macOS:** Apple Keychain Services (`Security.framework`).
- **Linux:** Freedesktop Secret Service API via D-Bus / GNOME Keyring / KWallet.

```python
import keyring
import json

KEYRING_SERVICE_NAME = "google-colab-mcp"
KEYRING_USERNAME = "colab_auth_credentials"

def store_credentials_in_keyring(creds: dict) -> None:
    payload = json.dumps(creds)
    keyring.set_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME, payload)

def load_credentials_from_keyring() -> Optional[dict]:
    payload = keyring.get_password(KEYRING_SERVICE_NAME, KEYRING_USERNAME)
    if payload:
        return json.loads(payload)
    return None
```

#### 3.4.2 Encrypted Headless File Fallback (AES-256-GCM)
In headless Linux server environments (e.g., Docker containers, SSH sessions, WSL without D-Bus), the OS keyring raises `keyring.errors.KeyringLocked` or fails. `colabmcp` transparently switches to an **AES-256-GCM Encrypted Fallback**:
1. A machine-unique salt and key are derived using PBKDF2-HMAC-SHA256 (100,000 iterations) from the host machine identifier (`/etc/machine-id` or Windows registry MachineGuid) and an optional user passphrase (`COLAB_MCP_KEY`).
2. Payload is encrypted with AES-256-GCM (12-byte random IV, 16-byte authentication tag).
3. Saved to `~/.colabmcp/credentials.enc` with strict POSIX permissions `0600` (read/write only by the current user).

---

## 4. Runtime Management & Colab Tunnel Frontend (TFE) Protocol

### 4.1 Tunnel Frontend (TFE) Topology & Endpoints
All consumer Colab runtime operations route through Google Colab's Tunnel Frontend (TFE) gateway:
- **Base Gateway URL:** `https://colab.research.google.com/tun/m`
- **Core TFE Endpoints:**

| Endpoint Route | HTTP Method | Purpose | Key Parameters / Headers |
|---|---|---|---|
| `/assign` | `GET` | Pre-assignment check / fetch XSRF token | Query: `authuser=0`, `nbh`, `variant`, `accelerator`, `shape` |
| `/assign` | `POST` | Allocate & claim Colab backend container VM | Header: `X-Goog-Colab-Token: <xsrf_token>` |
| `/assignments` | `GET` | Enumerate all active VM sessions for user | Query: `authuser=0` |
| `/ccu-info` | `GET` | Query remaining compute units & burn rate | Header: `Authorization: Bearer <token>` |
| `/unassign/{endpoint}` | `GET` | Query session teardown confirmation token | Path: `endpoint` |
| `/unassign/{endpoint}` | `POST` | Terminate Colab VM & release compute quota | Header: `X-Goog-Colab-Token: <token>` |
| `/{endpoint}/keep-alive/` | `GET` | Heartbeat to reset 90-min idle timeout | Header: `X-Colab-Tunnel: Google` |

### 4.2 Notebook Hash (`nbh`) Derivation Algorithm
Colab requires every runtime allocation to be bound to a unique 44-character web-safe base64 notebook hash (`nbh`). Passing a standard 36-character UUID string results in HTTP `400 Bad Request`.

```python
import uuid

def generate_notebook_hash(session_id: Optional[uuid.UUID] = None) -> str:
    """
    Generates exact 44-character web-safe base64 notebook hash (nbh).
    Algorithm verified from googlecolab/google-colab-cli:
    1. Generates UUID v4 (36 chars: e.g. 'c9a646d3-9c61-4cc9-bc77-b22e11e0563e')
    2. Replaces hyphens '-' with underscores '_'
    3. Right-pads with '.' up to exactly 44 characters (8 dots)
    """
    if session_id is None:
        session_id = uuid.uuid4()
    raw_str = str(session_id).replace("-", "_")
    padding = "." * (44 - len(raw_str))
    nbh = raw_str + padding
    assert len(nbh) == 44, f"Invalid nbh length: {len(nbh)}"
    return nbh
```

### 4.3 Hardware Accelerator Matrix & Machine Shapes
Colab supports the following accelerator and machine shape definitions:

```python
from enum import Enum

class Variant(str, Enum):
    DEFAULT = "DEFAULT"  # Standard CPU
    GPU = "GPU"          # NVIDIA CUDA GPU
    TPU = "TPU"          # Google Cloud TPU

class Accelerator(str, Enum):
    NONE = "NONE"        # CPU only
    T4 = "T4"            # NVIDIA Tesla T4 (16 GB GDDR6) - Free & Pro
    L4 = "L4"            # NVIDIA L4 (24 GB GDDR6) - Colab Pro / Pro+
    A100 = "A100"        # NVIDIA A100-SXM4 (40 GB / 80 GB HBM2) - Colab Pro / Pro+
    H100 = "H100"        # NVIDIA H100-SXM5 (80 GB HBM3) - Colab Pro+
    V5E1 = "V5E1"        # Google Cloud TPU v5e (Single chip, 16 GB HBM)
    V6E1 = "V6E1"        # Google Cloud TPU v6e (Trillium)

class Shape(str, Enum):
    STANDARD = "standard"  # ~12.7 GB System RAM (query param omitted or shape=0)
    HIGH_RAM = "hm"        # ~51.0 GB System RAM (query param shape=hm)
```

#### Hardware Tier & Compute Unit (CCU) Cost Reference:
| Accelerator Tier | Device Specification | VRAM / HBM | High-RAM Supported | Free Tier Available | Approx. CCU Cost / Hr |
|---|---|---|---|---|---|
| `NONE` (CPU) | 2 vCPU, Intel Xeon | None | Yes (Pro/Pro+ only — **NOT available on Free Tier**) | Yes | 0.00 |
| `T4` (GPU) | NVIDIA Tesla T4 | 16 GB GDDR6 | Yes (Pro/Pro+ only — **NOT available on Free Tier**) | Yes (dynamic quota) | ~1.96 |
| `L4` (GPU) | NVIDIA L4 Tensor Core | 24 GB GDDR6 | Yes (`hm`) | No (Pro/Pro+ only) | ~3.84 |
| `A100` (GPU) | NVIDIA A100-SXM4 | 40 GB / 80 GB | Yes (`hm`) | No (Pro/Pro+ only) | ~5.42 - 11.2 |
| `H100` (GPU) | NVIDIA H100-SXM5 | 80 GB HBM3 | Yes (`hm`) | No (Pro+ only) | ~13.50 |
| `V5E1` (TPU) | Cloud TPU v5e (Lite) | 16 GB HBM | No | No (Pro/Pro+ only) | ~1.96 |
| `V6E1` (TPU) | Cloud TPU v6e | 32 GB HBM | No | No (Pro+ only) | ~4.50 |

> **Important Constraint on Free Tier Machine Shapes:** High-RAM (`shape=hm`, ~51.0 GB RAM) is strictly restricted to paid subscriptions (Colab Pro / Pro+). Free Tier accounts attempting to provision `shape=hm` are rejected by Google's TFE gateway with an HTTP 400 Bad Request. Only standard RAM (~12.7 GB) is supported on the Free Tier.

### 4.4 Two-Step Assignment & Provisioning Handshake
Provisioning an accelerated cloud container requires a synchronized two-step handshake:

```
[colabmcp Runtime Client]                                       [Colab TFE Gateway]
           |                                                            |
           |--- 1. GET /tun/m/assign?authuser=0&nbh={nbh}&variant=GPU...|
           |    Headers: Authorization: Bearer <token>                  |
           |             X-Colab-Client-Agent: colab-mcp                |
           |             X-Goog-AuthUser: 0                             |
           |                                                            |
           |<-- Status 200: GetAssignmentResponse ----------------------|
           |    { "token": "AB-xxxx_XSRF_TOKEN", "nbh": "...", ... }    |
           |    (OR: returns Assignment if already running)             |
           |                                                            |
           |--- 2. POST /tun/m/assign?authuser=0&nbh={nbh}... ---------|
           |    Headers: Authorization: Bearer <token>                  |
           |             X-Goog-Colab-Token: AB-xxxx_XSRF_TOKEN         |
           |             X-Goog-AuthUser: 0                             |
           |                                                            |
           |<-- Status 200: PostAssignmentResponse ---------------------|
           |    {                                                       |
           |      "endpoint": "colab-f918e7c2-901a-4bc8-812b-...",      |
           |      "accelerator": "T4",                                  |
           |      "runtimeProxyInfo": {                                 |
           |        "url": "https://colab.research.google.com/tun/m/...",|
           |        "token": "ya29.c.b0xxxx...",                        |
           |        "tokenExpiresInSeconds": 3600                       |
           |      }                                                     |
           |    }                                                       |
```

#### Pre-Assignment Session Enumeration & Error Disambiguation Architecture:
1. **Re-attachment Check (`reuse_existing=True`):** Before initiating a new provisioning cycle, `colabmcp` queries `GET /tun/m/assignments?authuser={index}`. If an active VM container already exists that matches the requested accelerator, the client immediately re-attaches to its `endpoint` and `runtimeProxyInfo`. This avoids unnecessary allocations and prevents HTTP 412 quota rejection on Free Tier accounts (which permit only one active VM).
2. **Multi-Account Support (`auth_user_index`):** To support users logged into multiple Google Accounts simultaneously, `auth_user_index` (default: 0) is parameterized across all requests, sent as both `?authuser={index}` query parameter and `X-Goog-AuthUser: {index}` HTTP header.
3. **HTTP 400 Handling (High-RAM on Free Tier):** If High-RAM (`hm`) is requested on an account without a paid Colab Pro/Pro+ subscription, TFE returns HTTP 400. `colabmcp` catches this and raises `HighRamNotAvailableError` with explicit upgrade instructions.
4. **HTTP 412 Disambiguation:** TFE returns HTTP 412 Precondition Failed for distinct quota conditions. `colabmcp` parses the JSON error payload to distinguish:
   - `TOO_MANY_ASSIGNMENTS`: User has reached the maximum concurrent VM limit for their account tier.
   - `CCU_EXHAUSTED`: Paid Compute Unit balance is depleted (0.00 CCU remaining).
   - `FREE_TIER_QUOTA_EXCEEDED`: Free tier dynamic GPU quota exhausted; suggests fallback to CPU or waiting.

#### Python Implementation of Provisioning Protocol:
```python
import json
import httpx
from urllib.parse import urlencode
from typing import Optional, List, Dict, Any

class HighRamNotAvailableError(Exception):
    """Raised when High-RAM is requested on an account tier that does not support it."""
    pass

class TooManyAssignmentsError(Exception):
    """Raised when the user exceeds maximum allowed concurrent Colab sessions."""
    pass

class ComputeUnitsExhaustedError(Exception):
    """Raised when paid Colab Compute Units balance is exhausted."""
    pass

class FreeTierQuotaExceededError(Exception):
    """Raised when dynamic Free Tier GPU allocation quota is exhausted."""
    pass

class ColabTFEClient:
    BASE_URL = "https://colab.research.google.com/tun/m"

    def __init__(self, auth_token: str, auth_user_index: int = 0, client_agent: str = "colab-mcp"):
        self.auth_token = auth_token
        self.auth_user_index = auth_user_index
        self.client_agent = client_agent
        self.client = httpx.AsyncClient(timeout=30.0)

    def _get_base_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Accept": "application/json",
            "X-Colab-Client-Agent": self.client_agent,
            "X-Goog-AuthUser": str(self.auth_user_index),
        }

    def _build_assign_url(self, nbh: str, variant: Variant, accelerator: Accelerator, shape: Shape) -> str:
        params = {
            "authuser": str(self.auth_user_index),
            "nbh": nbh,
            "variant": variant.value,
            "accelerator": accelerator.value,
        }
        if shape == Shape.HIGH_RAM:
            params["shape"] = "hm"
        return f"{self.BASE_URL}/assign?{urlencode(params)}"

    async def list_assignments(self) -> List[Dict[str, Any]]:
        """Queries active VM container assignments for the authenticated user index."""
        url = f"{self.BASE_URL}/assignments?authuser={self.auth_user_index}"
        resp = await self.client.get(url, headers=self._get_base_headers())
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else data.get("assignments", [])

    def _parse_412_error(self, response_text: str):
        """Parses TFE HTTP 412 response body to disambiguate quota constraints."""
        try:
            err_data = json.loads(response_text)
            err_code = err_data.get("error", {}).get("code") or err_data.get("code", "")
            err_msg = err_data.get("error", {}).get("message") or err_data.get("message", "")
        except Exception:
            err_code, err_msg = "", response_text

        if "TOO_MANY_ASSIGNMENTS" in err_code or "TOO_MANY_ASSIGNMENTS" in err_msg:
            raise TooManyAssignmentsError(
                "Maximum concurrent Colab runtimes reached for this account. "
                "Disconnect active sessions or upgrade your Colab tier."
            )
        elif "CCU_EXHAUSTED" in err_code or "CCU_EXHAUSTED" in err_msg or "compute units" in err_msg.lower():
            raise ComputeUnitsExhaustedError(
                "Colab Compute Unit (CCU) balance is exhausted. "
                "Please purchase additional compute units at colab.research.google.com/signup."
            )
        elif "FREE_TIER" in err_code or "quota" in err_msg.lower():
            raise FreeTierQuotaExceededError(
                "Colab Free Tier GPU usage quota reached. "
                "Google Colab limits free GPU access dynamically. Switch to CPU or retry later."
            )
        else:
            raise RuntimeError(f"Colab allocation rejected (HTTP 412): {err_msg or response_text}")

    async def assign_runtime(
        self,
        nbh: str,
        variant: Variant = Variant.GPU,
        accelerator: Accelerator = Accelerator.T4,
        shape: Shape = Shape.STANDARD,
        reuse_existing: bool = True
    ) -> dict:
        # Step 0: Pre-Assignment Enumeration & Re-attachment
        if reuse_existing:
            active_sessions = await self.list_assignments()
            for session in active_sessions:
                # Match requested accelerator or compatible assignment
                session_acc = session.get("accelerator", "").upper()
                if session_acc == accelerator.value or (accelerator == Accelerator.NONE and session.get("variant") == "DEFAULT"):
                    if "endpoint" in session and "runtimeProxyInfo" in session:
                        return session

        url = self._build_assign_url(nbh, variant, accelerator, shape)
        headers = self._get_base_headers()

        # Step 1: Pre-assignment query & XSRF token fetch
        resp = await self.client.get(url, headers=headers)
        if resp.status_code == 400 and shape == Shape.HIGH_RAM:
            raise HighRamNotAvailableError(
                "High-RAM shape is not available on Google Colab Free Tier. "
                "Please upgrade to Colab Pro/Pro+ or request standard RAM (high_ram=False)."
            )
        elif resp.status_code == 412:
            self._parse_412_error(resp.text)
        resp.raise_for_status()
        data = resp.json()

        # If already assigned, return active assignment
        if "endpoint" in data and "runtimeProxyInfo" in data:
            return data

        xsrf_token = data.get("token")
        if not xsrf_token:
            raise ValueError("Colab TFE failed to provide an assignment XSRF token.")

        # Step 2: Post allocation request with XSRF token
        post_headers = {**headers, "X-Goog-Colab-Token": xsrf_token}
        alloc_resp = await self.client.post(url, headers=post_headers)
        if alloc_resp.status_code == 400 and shape == Shape.HIGH_RAM:
            raise HighRamNotAvailableError(
                "High-RAM shape is not available on Google Colab Free Tier. "
                "Please upgrade to Colab Pro/Pro+ or request standard RAM (high_ram=False)."
            )
        elif alloc_resp.status_code == 412:
            self._parse_412_error(alloc_resp.text)
        elif alloc_resp.status_code in (502, 503, 504):
            raise RuntimeError(f"Accelerator {accelerator.value} is currently out of capacity in Google Cloud data centers.")
        alloc_resp.raise_for_status()
        return alloc_resp.json()
```

### 4.5 Session Enumeration & Compute Unit (CCU) Tracking
- **Enumerating Active Sessions:** `GET https://colab.research.google.com/tun/m/assignments?authuser={auth_user_index}` with header `X-Goog-AuthUser: {auth_user_index}` returns all active VM endpoints attached to the user account index.
- **Compute Unit (CCU) Metrics:** `GET https://colab.research.google.com/tun/m/ccu-info?authuser={auth_user_index}` returns:
```json
{
  "user_tier": "PRO",
  "compute_units_remaining": 84.25,
  "consumption_rate_per_hour": 1.96,
  "active_assignments_count": 1,
  "auto_refill_enabled": true
}
```

### 4.6 Session Teardown & Resource Release (`unassign`)
To prevent burning compute units when work is completed, `colabmcp` executes a two-step unassign sequence passing the active `authuser` context:
```python
async def unassign_runtime(self, endpoint: str) -> bool:
    unassign_url = f"{self.BASE_URL}/unassign/{endpoint}?authuser={self.auth_user_index}"
    headers = self._get_base_headers()

    # Step 1: Query unassign verification token
    get_resp = await self.client.get(unassign_url, headers=headers)
    if get_resp.status_code == 404:
        return True # Already released
    get_resp.raise_for_status()
    xsrf_token = get_resp.json().get("token")

    # Step 2: Confirm deallocation
    post_headers = {**headers, "X-Goog-Colab-Token": xsrf_token}
    post_resp = await self.client.post(unassign_url, headers=post_headers)
    return post_resp.status_code == 200
```

### 4.7 Keep-Alive Heartbeat Daemon & Watchdog Inactivity Auto-Unassign
Colab monitors kernel activity. If no HTTP/WebSocket traffic occurs within **90 minutes**, the VM is automatically reclaimed by Google.
- **Keep-Alive Protocol:** Periodic GET request to `https://colab.research.google.com/tun/m/{endpoint}/keep-alive/?authuser={auth_user_index}`.
- **Headers:** `X-Colab-Tunnel: Google`, `X-Colab-Client-Agent: colab-mcp`, `X-Goog-AuthUser: {auth_user_index}`, `Authorization: Bearer <token>`.
- **Cadence:** Dispatched every 35 to 45 seconds in an `asyncio.Task`.
- **Fault-Tolerance Rule:** The Google TFE gateway frequently holds keep-alive requests until socket timeout. As proven in `googlecolab/google-colab-cli`, an HTTP `ReadTimeout` MUST be caught and treated as a **successful heartbeat**, because the TCP connection itself resets the backend activity timer.
- **Watchdog Inactivity Integration (Runaway Prevention):**
  To prevent unattended agents or forgotten sessions from burning expensive compute units (e.g. A100 at 5.42 CCU/hr), the keep-alive daemon is strictly coupled to an in-memory `last_activity_time` timestamp.
  - Every time an execution request (`colab_execute_code`, `colab_execute_notebook`) or output polling occurs, `last_activity_time` is updated to `time.time()`.
  - The heartbeat loop evaluates idle duration: `idle_seconds = time.time() - last_activity_time`.
  - **30-Minute Idle Auto-Termination:** If no execution occurs for **30 minutes (1,800 seconds)**, the daemon automatically:
    1. Suspends keep-alive heartbeat pings.
    2. Dispatches `unassign_runtime(endpoint)` to proactively release the cloud VM.
    3. Emits an audit log warning to `sys.stderr`: `[colab-mcp watchdog] Runtime {endpoint} was automatically deallocated after 30 minutes of inactivity to protect compute unit balance.`
    4. Sets the local session status to `IDLE_TIMEOUT_DISCONNECTED`.

---

## 5. Jupyter Kernel Communication & WebSocket Protocol

### 5.1 Colab Kernel Proxy Architecture & Kernel Discovery Handshake
Once a VM container is assigned, code execution occurs via Jupyter IPython kernel channels tunneled through Colab's secure reverse proxy.
- **Tunnel WebSocket URL:**
  ```
  wss://colab.research.google.com/tun/m/{endpoint}/api/kernels/{kernel_id}/channels
  ```

#### Intermediate Kernel Discovery & Creation Handshake:
Before establishing the WebSocket channel, the client must resolve an active `kernel_id`. When a container VM is provisioned, a kernel may already be initialized, or the container may require starting a new Python 3 kernel. The client performs an intermediate REST handshake via the Colab proxy:

1. **Query Active Kernels:**
   Issue `GET {proxy_url}/api/kernels` with header `X-Colab-Runtime-Proxy-Token: <token>`.
   If the returned list contains one or more active kernel objects, extract the kernel UUID from `resp.json()[0]["id"]`.
2. **Create Kernel (if none active):**
   If the kernel list is empty (`[]`), dispatch `POST {proxy_url}/api/kernels` with JSON body `{"name": "python3"}` and header `X-Colab-Runtime-Proxy-Token: <token>`. Extract the newly instantiated kernel ID from `resp.json()["id"]`.

```python
import httpx

async def discover_or_create_kernel(proxy_url: str, proxy_token: str) -> str:
    """
    Discovers an existing Jupyter kernel or starts a new Python 3 kernel via Colab proxy.
    """
    kernels_url = f"{proxy_url.rstrip('/')}/api/kernels"
    headers = {
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Step 1: Query active kernels
        resp = await client.get(kernels_url, headers=headers)
        resp.raise_for_status()
        kernels = resp.json()
        if kernels and len(kernels) > 0:
            return kernels[0]["id"]

        # Step 2: Create new python3 kernel if none active
        create_resp = await client.post(kernels_url, headers=headers, json={"name": "python3"})
        create_resp.raise_for_status()
        return create_resp.json()["id"]
```

### 5.2 WebSocket Handshake, Subprotocol Fix & Protocol Compatibility
In standard Jupyter deployments, clients send the header `Sec-WebSocket-Protocol: v1.kernel.websocket.jupyter.org`.
- **The Colab Bug:** Google Colab's Tunnel Frontend proxy completes the HTTP 101 Switching Protocols handshake but does **not** echo back the `Sec-WebSocket-Protocol` header.
- **The Empty Subprotocol Pitfall:** Setting `subprotocols=[]` in Python `websockets` causes the client to send an empty `Sec-WebSocket-Protocol: ` header, which RFC 6455 compliant servers and reverse proxies reject with **HTTP 400 Bad Request**.
- **The Authoritative Fix (`subprotocols=None`):** Setting `subprotocols=None` (omitted) ensures that Python `websockets` does NOT send any `Sec-WebSocket-Protocol` header at all, allowing the Colab TFE proxy to accept the connection cleanly.
- **Modern `websockets` Library Dynamic Compatibility (v13+ / v16.0):** In modern `websockets` versions (v13.0 through v16.0+), the handshake headers argument was renamed from `extra_headers` to `additional_headers`. `colabmcp` uses runtime signature inspection to support both legacy and modern `websockets` releases without deprecation warnings or `TypeError`.

```python
import inspect
import websockets

async def connect_colab_kernel_ws(proxy_url: str, proxy_token: str, kernel_id: str):
    """
    Establishes resilient WebSocket tunnel to Colab Jupyter kernel with
    RFC 6455 compliant headers and cross-version websockets compatibility.
    """
    ws_url = proxy_url.replace("https://", "wss://").rstrip("/") + f"/api/kernels/{kernel_id}/channels"
    headers = {
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "X-Colab-Client-Agent": "colab-mcp",
    }

    # Dynamic parameter inspection supporting websockets <13 (extra_headers) and >=13 (additional_headers)
    ws_kwargs = {
        "subprotocols": None,      # Prevents empty Sec-WebSocket-Protocol header rejection
        "ping_interval": 20,
        "ping_timeout": 20,
        "max_size": 50 * 1024 * 1024  # Support up to 50MB plot/data frames
    }

    if "additional_headers" in inspect.signature(websockets.connect).parameters:
        ws_kwargs["additional_headers"] = headers
    else:
        ws_kwargs["extra_headers"] = headers

    return await websockets.connect(ws_url, **ws_kwargs)
```

### 5.3 Jupyter Messaging Protocol v5.3 Multiplexing
All communication over the WebSocket channel consists of JSON-serialized frames adhering to the Jupyter Messaging Protocol (v5.3/v5.4):

```json
{
  "header": {
    "msg_id": "c8f921a4-921b-410a-ba02-918239018234",
    "username": "colab-mcp",
    "session": "a901f421-4123-4122-b231-102938475612",
    "msg_type": "execute_request",
    "version": "5.3",
    "date": "2026-11-16T08:15:30.123456Z"
  },
  "parent_header": {},
  "metadata": {},
  "content": {
    "code": "import torch\nprint('CUDA Available:', torch.cuda.is_available())",
    "silent": false,
    "store_history": true,
    "user_expressions": {},
    "allow_stdin": false,
    "stop_on_error": true
  },
  "channel": "shell",
  "buffers": []
}
```

#### Channel Multiplexing Matrix:
| Channel | Direction | Key Message Types | Purpose |
|---|---|---|---|
| `shell` | Bidirectional | `execute_request`, `execute_reply` | Code submission, execution status return, execution count |
| `iopub` | Kernel ➔ Client | `status`, `execute_input`, `stream`, `display_data`, `execute_result`, `error` | Live stdout/stderr chunks, generated plots/images, tracebacks |
| `stdin` | Kernel ➔ Client | `input_request`, `input_reply` | Interactive user input prompts (disabled by default in agent mode) |
| `control` | Bidirectional | `interrupt_request`, `interrupt_reply`, `shutdown_request` | High-priority out-of-band execution interruption (SIGINT) |

### 5.4 Execution State Tracking & Message Correlator
Because the WebSocket connection is asynchronous and multiplexed, incoming `iopub` messages from parallel tasks or background threads must be correlated with the originating execution request.
- Every `execute_request` generates a unique UUID `msg_id`.
- The kernel injects this UUID into the `parent_header.msg_id` of all resulting `iopub` and `shell` messages.
- The `colabmcp` Correlator buffers messages matching the active `msg_id` until the `status: idle` and `execute_reply` frames are received.

```python
class KernelMessageCorrelator:
    def __init__(self):
        self.active_executions: dict[str, asyncio.Queue] = {}

    def register_execution(self, msg_id: str) -> asyncio.Queue:
        queue = asyncio.Queue()
        self.active_executions[msg_id] = queue
        return queue

    def route_incoming_message(self, message: dict) -> None:
        parent_id = message.get("parent_header", {}).get("msg_id")
        if parent_id in self.active_executions:
            self.active_executions[parent_id].put_nowait(message)

    def complete_execution(self, msg_id: str) -> None:
        self.active_executions.pop(msg_id, None)
```

### 5.5 Multimodal MIME Output Processing, Chronological Ordering & ANSI Sanitization

#### MIME Type to MCP Content Block Translation & Chronological Interleaving:
Rather than segregating all text outputs prior to images, `colabmcp` strictly preserves **chronological emission order** as messages arrive from the kernel's `iopub` channel. If user code emits stdout, generates a plot, and then prints downstream metrics, the MCP tool returns `[TextContent, ImageContent, TextContent]` in exact chronological sequence.

```
+------------------------------------+-----------------------------------------------------+
| Jupyter iopub Message Payload      | MCP Content Block Translation                       |
+------------------------------------+-----------------------------------------------------+
| stream (content.name == "stdout")  | TextContent(type="text", text=content.text)         |
| stream (content.name == "stderr")  | TextContent(type="text", text="[stderr] " + text)   |
| display_data / execute_result:     |                                                     |
|   - "image/png" (base64 string)    | ImageContent(type="image", data=b64, mime="image/png")
|   - "image/jpeg" (base64 string)   | ImageContent(type="image", data=b64, mime="image/jpeg")
|   - "text/plain" (string)          | TextContent(type="text", text=data["text/plain"])   |
|   - "text/html" (HTML string)      | Converted to Markdown or raw TextContent            |
| error:                             |                                                     |
|   - ename, evalue, traceback[]     | TextContent(type="text", text=clean_traceback)      |
|                                    | Flag response with isError: true                    |
+------------------------------------+-----------------------------------------------------+
```

#### ECMA-48 Compliant ANSI Escape Sanitization Regex:
Python tracebacks emitted by IPython include intense ANSI 24-bit color escapes, terminal title codes, and Fe control sequences. These corrupt LLM prompt interpretation and waste context window tokens. `colabmcp` filters tracebacks and stream text using an ECMA-48 compliant sanitization pattern:

```python
import re

ANSI_ESCAPE_PATTERN = re.compile(
    r'\x1b(?:'
    r'\[[0-?]*[ -/]*[@-~]'           # CSI sequences (colors, cursor modes, line clears)
    r'|(?s:\].*?(?:\x1b\\|\x07))'    # OSC sequences (hyperlinks, window titles)
    r'|[\(\)][A-Z0-9]'               # VT100 character set designations
    r'|[@-Z\\-_]'                    # 2-byte Fe control sequences
    r')'
)

def sanitize_ansi_traceback(raw_traceback_lines: list[str]) -> str:
    """Strips all ECMA-48 terminal color codes, cursor sequences, and OSC payloads."""
    cleaned_lines = [ANSI_ESCAPE_PATTERN.sub('', line) for line in raw_traceback_lines]
    return "\n".join(cleaned_lines)
```

#### Output Safety Bounds & Head-Tail Truncation:
To protect LLM context windows against runaway loops (e.g. `while True: print(...)`) or giant datasets, `colabmcp` enforces strict safety bounds on execution output:
- **`max_text_lines`:** Maximum text lines retained (default: 1,000 lines).
- **`max_output_bytes`:** Maximum raw text payload size (default: 102,400 bytes / 100 KB).
- **`max_images`:** Maximum visual plots returned per execution (default: 5 images).
- **`max_images` Enforcement Rule:** If the number of emitted images exceeds `max_images` (default: 5), the output pipeline retains the first `max_images` visual blocks chronologically, omits subsequent images, and appends a notice `TextContent(type="text", text="[Notice: N additional plot images omitted due to max_images limit]")`.

When output exceeds `max_text_lines` or `max_output_bytes`, the output processor applies **head-tail truncation**, preserving the initial portion (setup/progress) and final portion (summary/stacktrace) with a prominent omission banner. To guarantee strict ceilings on both line count and byte volume without slice duplication when long lines overflow byte thresholds, a verified two-stage truncation algorithm is enforced:

```python
def apply_head_tail_truncation(
    text: str,
    max_lines: int = 1000,
    max_bytes: int = 102400
) -> str:
    """
    Preserves first half and last half of output with truncation warning if limits exceeded.
    Guarantees strict ceilings on both line count and byte volume without slice overlap.
    """
    encoded = text.encode("utf-8")
    lines = text.splitlines()

    if len(lines) <= max_lines and len(encoded) <= max_bytes:
        return text

    # Stage 1: Line-based truncation if line count exceeds threshold
    omitted_lines = 0
    if len(lines) > max_lines:
        head_line_count = max_lines // 2
        tail_line_count = max_lines - head_line_count
        head_lines = lines[:head_line_count]
        tail_lines = lines[-tail_line_count:] if tail_line_count > 0 else []
        omitted_lines = len(lines) - (head_line_count + tail_line_count)
        text = "\n".join(head_lines) + "\n... [TRUNCATED] ...\n" + "\n".join(tail_lines)
        encoded = text.encode("utf-8")

    # Stage 2: Byte-based truncation if payload still exceeds byte ceiling
    if len(encoded) > max_bytes:
        half_bytes = max_bytes // 2
        head_bytes = encoded[:half_bytes].decode("utf-8", errors="ignore")
        tail_bytes = encoded[-half_bytes:].decode("utf-8", errors="ignore")
        omitted_bytes = len(encoded) - (len(head_bytes.encode("utf-8")) + len(tail_bytes.encode("utf-8")))

        banner = (
            f"\n\n... [TRUNCATED: "
            + (f"{omitted_lines} lines / " if omitted_lines > 0 else "")
            + f"{omitted_bytes} bytes omitted to preserve LLM context window] ...\n\n"
        )
        return head_bytes + banner + tail_bytes

    # Line-only truncation banner
    head_line_count = max_lines // 2
    tail_line_count = max_lines - head_line_count
    banner = f"\n\n... [TRUNCATED: {omitted_lines} lines omitted to preserve LLM context window] ...\n\n"
    return "\n".join(lines[:head_line_count]) + banner + "\n".join(lines[-tail_line_count:])
```

### 5.6 Out-of-Band Execution Interruption & Kernel Restart Distinction
When long-running or runaway code hangs, `colabmcp` executes a tiered abort procedure with strict status distinction:

1. **Tier 1: Non-Destructive SIGINT Interrupt:**
   - **Method A (REST API Bypass):** Dispatches HTTP `POST {proxy_url}/api/kernels/{kernel_id}/interrupt` with `X-Colab-Runtime-Proxy-Token`. Returns HTTP `204 No Content`.
   - **Method B (WebSocket Control Channel):** Dispatches `interrupt_request` on `channel: "control"`.
   - **Preservation:** SIGINT halts execution **without clearing in-memory state**. All previously declared variables, imported modules, trained model weights, and history are preserved (`state_reset: false`, `kernel_restarted: false`).

2. **Tier 2: Escalated Kernel Restart (Destructive):**
   - If the kernel fails to return `status: idle` within 5.0 seconds of SIGINT dispatch (e.g. unresponsive native CUDA C-extensions, un-interruptible I/O loops), `colabmcp` escalates to `POST {proxy_url}/api/kernels/{kernel_id}/restart`.
   - **State Destruction Warning:** Kernel restart kills the underlying IPython process. **All in-memory variables and objects are completely lost** (`state_reset: true`, `kernel_restarted: true`).
   - The response clearly flags this to the agent so prerequisite cells can be re-run.

#### WebSocket Client Auto-Reconnection State Machine:
When a kernel is restarted, the existing WebSocket connection drops or receives `status: "restarting"`. `colabmcp` implements a 6-stage auto-reconnection state machine:
```
[CONNECTED] --(restart initiated / WS close)--> [DISCONNECTED]
      |                                              |
      v                                              v
[RECONNECTING] <---(Poll GET /api/kernels)--- [BACKOFF_SLEEP]
      |
      +---> (Kernel Ready) ---> [ESTABLISHING_WS] ---> [RECONNECTED / IDLE]
```
1. **Detection:** Receives `status: "restarting"` on `iopub` or abnormal WebSocket termination (code 1006).
2. **Teardown:** Cleanly closes stale WebSocket connection without raising unhandled exceptions.
3. **Kernel Re-Discovery:** Polls `GET {proxy_url}/api/kernels` with exponential backoff (0.5s, 1.0s, 2.0s) until the newly spawned kernel reports `execution_state: "idle"`.
4. **WebSocket Re-Handshake:** Re-invokes `connect_colab_kernel_ws()` with refreshed socket parameters.
5. **State Synchronization:** Flushes stale correlated message queues and re-registers listeners.
6. **Notification:** Resumes `CONNECTED` status and emits audit event to `sys.stderr`.

---

## 6. Notebook Execution Engine

### 6.1 nbformat v4 Parsing & Schema Validation
Notebook files (`.ipynb`) are validated against the official Jupyter `nbformat` v4 schema prior to transmission. The loader explicitly accepts either a filesystem path or raw in-memory JSON content:

```python
from typing import Optional
import nbformat
from nbformat.validator import validate, ValidationError

def load_and_validate_notebook(
    notebook_path: Optional[str] = None,
    notebook_content: Optional[str] = None
) -> nbformat.NotebookNode:
    """
    Loads and validates a Jupyter notebook against nbformat v4 schema.
    Accepts either a filesystem path (notebook_path) or raw JSON content (notebook_content).
    """
    if notebook_path is None and notebook_content is None:
        raise ValueError("Either notebook_path or notebook_content must be provided.")
    if notebook_path is not None and notebook_content is not None:
        raise ValueError("Provide either notebook_path OR notebook_content, not both.")

    if notebook_content is not None:
        nb = nbformat.reads(notebook_content, as_version=4)
    else:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

    # Validates structure, cell types, and schema compliance
    validate(nb)
    return nb
```

### 6.2 Sequential & Selective Cell Index Execution
The notebook runner supports executing all cells sequentially or selectively targeting indices (e.g. `cell_indices=[0, 1, 4]`):
1. **0-Based Absolute Indexing (`nb.cells`):** The `cell_indices` parameter refers strictly to the 0-based absolute index within `nb.cells`. Non-code cells (markdown, raw text) situated at those indices are automatically bypassed during code execution while their positions and metadata remain unaltered in the document tree.
2. **Context Persistence & Optional Kernel Reset:** Variables, functions, and imported modules persist in the kernel session across cells. To ensure reproducible benchmark runs, Tool 9 provides `reset_kernel_before_run: bool = true`, which issues a clean kernel restart prior to executing the notebook.
3. **Execution Count Tracking:** The engine increments and attaches `execution_count` to each executed code cell node.

### 6.3 Error Handling Policies & State Propagation
When executing multi-cell workflows, the `stop_on_error` parameter controls fault propagation:
- `stop_on_error = True` (Default): If cell $N$ raises an unhandled exception, execution halts immediately. Cells $N+1 \dots M$ are marked as `skipped`, and the partial notebook state is preserved.
- `stop_on_error = False`: The runner captures the error for cell $N$ and continues executing cell $N+1$.
- **Detailed Error Capture:** For any failed cell, `colabmcp` extracts and populates `error_name`, `error_value`, and the sanitized `traceback` within the resulting `cell_summaries` array, allowing agents to diagnose and repair failures without re-executing.

### 6.4 In-Memory Execution State & Disk Serialization
Outputs gathered from `iopub` are converted back into standard `nbformat` output nodes:
- `StreamNode` (`output_type: "stream"`, `name: "stdout"|"stderr"`, `text: "..."`)
- `DisplayDataNode` (`output_type: "display_data"`, `data: {"image/png": "...", "text/plain": "..."}`)
- `ExecuteResultNode` (`output_type: "execute_result"`, `execution_count: N`, `data: {...}`)
- `ErrorNode` (`output_type: "error"`, `ename: "...", evalue: "...", traceback: [...]`)

The updated notebook tree is atomically written to `output_save_path` using `nbformat.write(nb, f)`.

---

## 7. MCP Server Specification & Tool Schemas

### 7.1 Technology Stack Justification: Python MCPServer vs TypeScript
The implementation technology was evaluated across Python and TypeScript MCP SDKs:

| Dimension | Python (`mcp.server.mcpserver.MCPServer`) | TypeScript (`@modelcontextprotocol/sdk`) | Architectural Decision |
|---|---|---|---|
| **Ecosystem Synergy** | **Native**. First-party integration with `nbformat`, `jupyter_client`, and PyData tools. | **Complex**. Requires `@jupyterlab/services` and Node.js-based polyfills. | **Python** provides 100% native compatibility with Jupyter notebook ASTs. |
| **Schema Generation** | Automatic generation from Pydantic models & type annotations. | Zod schemas with manual JSON-RPC adapters. | **Python** reduces schema divergence and maintenance overhead. |
| **Multimodal Handling** | Native `ImageContent` and `TextContent` primitives. | Manual JSON object construction. | **Tie** (Both comply with MCP 2024-11-05). |
| **OS Keystore Security** | `keyring` library supports Windows DPAPI, macOS Keychain, Linux Secret Service out-of-the-box. | Requires `keytar` native C++ bindings, which frequently break across platforms. | **Python** provides seamless cross-platform keystore persistence. |
| **Deployment Ergonomics** | Run instantly via `uvx colab-mcp` or `pipx run colab-mcp`. Matches Python data science environments. | Requires Node.js runtime installed alongside Python. | **Python** matches the target user and agent runtime environment. |

### 7.2 Dual Transport Engine: Stdio & Server-Sent Events (SSE)

#### 7.2.1 Stdio Transport & Absolute Stdout Isolation
For local AI assistants (Claude Desktop, Cursor, Antigravity), `stdio` is the standard transport.
- **The Core Risk:** MCP `stdio` communication relies on single-line JSON-RPC 2.0 messages delimited by `\n` on `sys.stdout`. If any dependent library (e.g., `requests`, `playwright`, `httpx`, or an accidental `print()`) writes a raw string to `sys.stdout`, the JSON-RPC framing is instantly corrupted, terminating the agent connection.
- **Guaranteed Isolation Architecture:**
  1. `colabmcp` intercepts `sys.stdout` immediately upon process startup.
  2. All standard logging handlers (`logging.StreamHandler`) are explicitly bound to `stream=sys.stderr`.
  3. `MCPServer` takes exclusive ownership of the underlying raw stdout stream for JSON-RPC frame delivery.

#### 7.2.2 SSE Transport for Remote/Hub Deployments
For remote deployment, Docker containers, or multi-agent hubs, `colabmcp` provides an HTTP Server-Sent Events (SSE) transport:
- Invoked via: `colab-mcp run --transport sse --host 0.0.0.0 --port 8000`
- `GET /sse`: Persistent event-stream establishing the client session.
- `POST /messages?session_id=<UUID>`: JSON-RPC request ingestion endpoint.

---

### 7.3 Complete JSON Schemas for 12 MCP Tools

```
================================================================================
CATEGORY 1: AUTHENTICATION TOOLS
================================================================================
```

#### Tool 1: `colab_login`
- **Description:** Authenticate with Google Colab. Supports Google OAuth 2.0 (remote copy-paste authorization URL, local loopback browser listener, or ambient Application Default Credentials) and browser-assisted cookie extraction via Chrome DevTools Protocol (CDP).
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabLoginInput",
  "type": "object",
  "properties": {
    "method": {
      "type": "string",
      "enum": ["oauth_remote", "oauth_loopback", "oauth_adc", "cdp_browser", "cookie_import"],
      "default": "oauth_remote",
      "description": "Authentication mechanism: 'oauth_remote' generates a Google authorization URL for terminal copy-paste; 'oauth_loopback' opens local browser with loopback listener; 'oauth_adc' uses ambient gcloud credentials; 'cdp_browser' launches Chrome with debugging port to capture cookies; 'cookie_import' accepts manual session cookies."
    },
    "auth_user_index": {
      "type": "integer",
      "default": 0,
      "minimum": 0,
      "description": "Google multi-account user index (maps to ?authuser=N and X-Goog-AuthUser: N, default 0)."
    },
    "auth_code": {
      "type": "string",
      "description": "The authorization code obtained from the Google OAuth consent landing page. Required if method is 'oauth_remote' and completing exchange."
    },
    "cookies": {
      "type": "object",
      "description": "Cookie key-value mapping required if method is 'cookie_import'. Must contain at least 'SAPISID', 'SID', and 'OSID'.",
      "properties": {
        "SID": { "type": "string" },
        "HSID": { "type": "string" },
        "SSID": { "type": "string" },
        "SAPISID": { "type": "string" },
        "__Secure-1PSID": { "type": "string" },
        "__Secure-3PSID": { "type": "string" },
        "OSID": { "type": "string" }
      },
      "required": ["SAPISID", "SID"]
    },
    "timeout_seconds": {
      "type": "integer",
      "default": 180,
      "minimum": 30,
      "maximum": 600,
      "description": "Maximum seconds to await interactive browser login before raising timeout."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "status": "authenticated",
  "auth_method": "oauth_remote",
  "auth_user_index": 0,
  "user_email": "researcher@example.com",
  "expires_at": "2026-11-16T12:00:00Z",
  "has_refresh_token": true,
  "auth_url": null,
  "message": "Successfully authenticated with Google Colab via OAuth 2.0."
}
```

---

#### Tool 2: `colab_get_auth_status`
- **Description:** Query the current authentication status, active account identity, token/cookie expiration times, and available Colab Compute Units (CCU).
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabGetAuthStatusInput",
  "type": "object",
  "properties": {
    "auth_user_index": {
      "type": "integer",
      "default": 0,
      "minimum": 0,
      "description": "Google multi-account user index to inspect (default 0)."
    },
    "force_refresh": {
      "type": "boolean",
      "default": false,
      "description": "Whether to perform an active network check against Colab TFE to verify token validity."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "is_authenticated": true,
  "auth_method": "oauth2",
  "auth_user_index": 0,
  "user_email": "researcher@example.com",
  "token_valid": true,
  "expires_in_seconds": 3540,
  "account_tier": "PRO",
  "compute_units_remaining": 84.25,
  "active_sessions_count": 1
}
```

---

#### Tool 3: `colab_logout`
- **Description:** Purge cached Google authentication credentials from OS keystore and local encrypted cache, and optionally terminate all active Colab cloud runtimes.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabLogoutInput",
  "type": "object",
  "properties": {
    "auth_user_index": {
      "type": "integer",
      "default": 0,
      "minimum": 0,
      "description": "Google multi-account user index to log out (default 0)."
    },
    "terminate_active_runtimes": {
      "type": "boolean",
      "default": true,
      "description": "Whether to actively release all provisioned Colab cloud VMs to stop compute unit consumption before logging out."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "status": "logged_out",
  "auth_user_index": 0,
  "keystore_purged": true,
  "terminated_runtimes_count": 1,
  "message": "Authentication credentials securely wiped and active runtimes deallocated."
}
```

---

```
================================================================================
CATEGORY 2: RUNTIME MANAGEMENT TOOLS
================================================================================
```

#### Tool 4: `colab_list_runtime_types`
- **Description:** Query available Google Colab hardware accelerators (CPU, T4, L4, A100, H100 GPUs, TPU v5e/v6e), high-RAM configurations, current availability, and hourly compute unit consumption rates.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabListRuntimeTypesInput",
  "type": "object",
  "properties": {
    "auth_user_index": {
      "type": "integer",
      "default": 0,
      "minimum": 0,
      "description": "Google multi-account user index (default 0)."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "account_tier": "PRO",
  "available_accelerators": [
    {
      "accelerator": "NONE",
      "variant": "DEFAULT",
      "display_name": "Standard CPU",
      "vram": null,
      "system_ram": "12.7 GB (Standard) / 51.0 GB (High-RAM — Pro/Pro+ only)",
      "ccu_cost_per_hour": 0.0,
      "available": true
    },
    {
      "accelerator": "T4",
      "variant": "GPU",
      "display_name": "NVIDIA Tesla T4",
      "vram": "16 GB GDDR6",
      "system_ram": "12.7 GB (Standard) / 51.0 GB (High-RAM — Pro/Pro+ only)",
      "ccu_cost_per_hour": 1.96,
      "available": true
    },
    {
      "accelerator": "L4",
      "variant": "GPU",
      "display_name": "NVIDIA L4 Tensor Core",
      "vram": "24 GB GDDR6",
      "system_ram": "51.0 GB",
      "ccu_cost_per_hour": 3.84,
      "available": true
    },
    {
      "accelerator": "A100",
      "variant": "GPU",
      "display_name": "NVIDIA A100-SXM4",
      "vram": "40 GB / 80 GB HBM2",
      "system_ram": "83.5 GB",
      "ccu_cost_per_hour": 5.42,
      "available": true
    },
    {
      "accelerator": "V5E1",
      "variant": "TPU",
      "display_name": "Google Cloud TPU v5e",
      "vram": "16 GB HBM",
      "system_ram": "Single-chip pod slice",
      "ccu_cost_per_hour": 1.96,
      "available": true
    }
  ]
}
```

---

#### Tool 5: `colab_connect_runtime`
- **Description:** Provision a new Colab container VM or attach to an existing session with requested accelerator (CPU, T4, L4, A100, TPU) and RAM shape (Standard vs High-RAM), establishing the secure Jupyter WebSocket bridge. When `reuse_existing=True`, the tool queries `GET /tun/m/assignments?authuser={auth_user_index}` and re-attaches to any matching running VM, preventing HTTP 412 quota rejection on Free Tier accounts. Note that High-RAM (`high_ram=True`) is not supported on the Free Tier and will raise `HighRamNotAvailableError`.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabConnectRuntimeInput",
  "type": "object",
  "properties": {
    "accelerator": {
      "type": "string",
      "enum": ["NONE", "T4", "L4", "A100", "H100", "V5E1", "V6E1"],
      "default": "NONE",
      "description": "Hardware accelerator to allocate."
    },
    "high_ram": {
      "type": "boolean",
      "default": false,
      "description": "Request high-memory container shape (~51 GB RAM). Strictly requires Colab Pro/Pro+; NOT available on Free Tier (raises HighRamNotAvailableError)."
    },
    "reuse_existing": {
      "type": "boolean",
      "default": true,
      "description": "If an active VM matching the requested accelerator already exists, re-attach to it via GET /tun/m/assignments rather than provisioning a new VM."
    },
    "auth_user_index": {
      "type": "integer",
      "default": 0,
      "minimum": 0,
      "description": "Google multi-account user index (maps to ?authuser=N and X-Goog-AuthUser: N, default 0)."
    },
    "timeout_seconds": {
      "type": "integer",
      "default": 120,
      "minimum": 30,
      "maximum": 300,
      "description": "Timeout in seconds to wait for cloud VM allocation and WebSocket handshake."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "runtime_id": "colab-f918e7c2-901a-4bc8-812b-2019482019ab",
  "status": "connected",
  "accelerator": "T4",
  "variant": "GPU",
  "high_ram": false,
  "reused_existing": true,
  "resources": {
    "cpu_model": "Intel(R) Xeon(R) CPU @ 2.20GHz",
    "cpu_count": 2,
    "system_ram_gb": 12.67,
    "gpu_name": "Tesla T4",
    "gpu_vram_gb": 15.0,
    "disk_space_total_gb": 107.7
  },
  "kernel_id": "8f3b149a-281a-49a7-9b24-118492048201",
  "kernel_state": "idle",
  "heartbeat_interval_seconds": 35
}
```

---

#### Tool 6: `colab_get_runtime_status`
- **Description:** Inspect active VM health, live resource consumption (CPU, RAM, GPU VRAM, disk), kernel execution state (`idle`, `busy`), and remaining idle timer countdown.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabGetRuntimeStatusInput",
  "type": "object",
  "properties": {
    "runtime_id": {
      "type": "string",
      "description": "Optional runtime identifier. Defaults to the active runtime session."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "runtime_id": "colab-f918e7c2-901a-4bc8-812b-2019482019ab",
  "status": "active",
  "kernel_state": "idle",
  "uptime_seconds": 2450,
  "idle_seconds": 84,
  "metrics": {
    "cpu_utilization_percent": 12.4,
    "ram_used_gb": 3.8,
    "ram_total_gb": 12.67,
    "gpu_utilization_percent": 45.0,
    "gpu_vram_used_gb": 4.2,
    "gpu_vram_total_gb": 15.0,
    "disk_used_gb": 31.2,
    "disk_total_gb": 107.7
  },
  "keep_alive_healthy": true
}
```

---

#### Tool 7: `colab_disconnect_runtime`
- **Description:** Release and terminate an allocated Colab container VM via the TFE `/unassign` route, stopping cloud compute unit consumption.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabDisconnectRuntimeInput",
  "type": "object",
  "properties": {
    "runtime_id": {
      "type": "string",
      "description": "Identifier of the runtime session to release. Defaults to active runtime."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "runtime_id": "colab-f918e7c2-901a-4bc8-812b-2019482019ab",
  "status": "terminated",
  "compute_resources_released": true,
  "message": "Backend VM successfully deallocated and compute units halted."
}
```

---

```
================================================================================
CATEGORY 3: EXECUTION & OUTPUT MONITORING TOOLS
================================================================================
```

#### Tool 8: `colab_execute_code`
- **Description:** Execute a Python snippet or IPython magic command (`!pip`, `%timeit`) on the connected Colab kernel. Returns captured stdout, stderr, rich multimodal MIME outputs (e.g. Matplotlib charts as ImageContent), and sanitized error tracebacks in strict chronological order. Enforces output safety bounds (max text lines, max bytes, max images) with automatic head-tail truncation to preserve LLM context windows.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabExecuteCodeInput",
  "type": "object",
  "properties": {
    "code": {
      "type": "string",
      "description": "Python source code or IPython shell commands to execute in the remote kernel."
    },
    "runtime_id": {
      "type": "string",
      "description": "Optional runtime identifier. Defaults to active runtime."
    },
    "timeout_seconds": {
      "type": "integer",
      "default": 300,
      "minimum": 5,
      "maximum": 3600,
      "description": "Execution timeout in seconds before sending SIGINT interrupt."
    },
    "wait_for_completion": {
      "type": "boolean",
      "default": true,
      "description": "If true, blocks until execution completes and returns full outputs. If false, dispatches asynchronously and immediately returns an execution_id for polling."
    },
    "store_history": {
      "type": "boolean",
      "default": true,
      "description": "Whether to increment execution_count and record inputs in IPython history."
    },
    "max_text_lines": {
      "type": "integer",
      "default": 1000,
      "minimum": 10,
      "maximum": 10000,
      "description": "Maximum text lines returned for stdout/stderr before applying head-tail truncation."
    },
    "max_output_bytes": {
      "type": "integer",
      "default": 102400,
      "minimum": 1024,
      "maximum": 10485760,
      "description": "Maximum raw output bytes (default: 100 KB) before applying truncation."
    },
    "max_images": {
      "type": "integer",
      "default": 5,
      "minimum": 1,
      "maximum": 50,
      "description": "Maximum visual plot images returned per execution."
    }
  },
  "required": ["code"]
}
```
- **Output Return Format:** Array of MCP Content Blocks strictly preserving chronological kernel emission order:
  ```json
  [
    {
      "type": "text",
      "text": "[Execution 1] Status: ok (Duration: 2.14s)\nstdout:\nEpoch 1/5 - loss: 0.421\nGenerating training loss curve..."
    },
    {
      "type": "image",
      "data": "iVBORw0KGgoAAAANSUhEUgAA...",
      "mimeType": "image/png"
    },
    {
      "type": "text",
      "text": "Epoch 5/5 complete. Final validation accuracy: 96.4%."
    }
  ]
  ```
  *(If an unhandled exception occurs, an error `TextContent` block with ECMA-48 sanitized traceback is appended, and `isError: true` is set).*

---

#### Tool 9: `colab_execute_notebook`
- **Description:** Load, validate, and execute an entire `.ipynb` Jupyter notebook file cell-by-cell on the active Colab runtime. Supports selective cell execution by 0-based absolute indices in `nb.cells`, optional kernel reset before run, error interruption policies, and writing outputs back to disk.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabExecuteNotebookInput",
  "type": "object",
  "properties": {
    "notebook_path": {
      "type": "string",
      "description": "Absolute or relative file path to the local .ipynb notebook. Provide either notebook_path OR notebook_content."
    },
    "notebook_content": {
      "type": "string",
      "description": "Raw JSON string of the notebook. Provide either notebook_path OR notebook_content."
    },
    "reset_kernel_before_run": {
      "type": "boolean",
      "default": true,
      "description": "Whether to restart the kernel prior to execution to guarantee a clean environment and reproducible run."
    },
    "cell_indices": {
      "type": "array",
      "items": { "type": "integer" },
      "description": "Optional list of 0-based absolute indices in nb.cells to execute (e.g. [0, 1, 3]). Non-code cells situated at specified indices are automatically skipped. If omitted, executes all code cells in sequence."
    },
    "stop_on_error": {
      "type": "boolean",
      "default": true,
      "description": "Whether to stop executing downstream cells if an unhandled exception occurs."
    },
    "timeout_per_cell_seconds": {
      "type": "integer",
      "default": 300,
      "minimum": 5,
      "maximum": 3600,
      "description": "Maximum execution duration per cell."
    },
    "output_save_path": {
      "type": "string",
      "description": "Optional file path to write the executed notebook with populated cell outputs."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
```json
{
  "total_cells": 8,
  "executed_cells": 3,
  "successful_cells": 2,
  "failed_cells": 1,
  "skipped_cells": 5,
  "total_duration_seconds": 16.5,
  "kernel_reset_performed": true,
  "cell_summaries": [
    {
      "cell_index": 0,
      "execution_count": 1,
      "status": "success",
      "duration_seconds": 1.1,
      "has_plots": false
    },
    {
      "cell_index": 1,
      "execution_count": 2,
      "status": "success",
      "duration_seconds": 14.2,
      "has_plots": true,
      "plot_count": 1
    },
    {
      "cell_index": 2,
      "execution_count": 3,
      "status": "failed",
      "duration_seconds": 1.2,
      "has_plots": false,
      "error_name": "ZeroDivisionError",
      "error_value": "division by zero",
      "traceback": [
        "Traceback (most recent call last):",
        "  File \"<ipython-input-3-abc>\", line 2, in <module>",
        "    result = 10 / 0",
        "ZeroDivisionError: division by zero"
      ]
    }
  ],
  "saved_to": "c:\\Users\\tream\\OneDrive\\Documents\\1DEV\\colabmcp\\results.ipynb"
}
```

---

#### Tool 10: `colab_get_execution_status`
- **Description:** Query execution progress, elapsed time, and partial stdout of an asynchronous cell job dispatched with `wait_for_completion=false`.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabGetExecutionStatusInput",
  "type": "object",
  "properties": {
    "execution_id": {
      "type": "string",
      "description": "Unique asynchronous execution ID returned by colab_execute_code."
    }
  },
  "required": ["execution_id"]
}
```
- **Output Return Format (JSON):**
```json
{
  "execution_id": "exec-a901284-9102",
  "status": "running",
  "elapsed_seconds": 45.2,
  "partial_stdout_tail": "Epoch 12/50 - loss: 0.0412 - accuracy: 0.9841",
  "has_error": false
}
```

---

#### Tool 11: `colab_get_cell_output`
- **Description:** Retrieve full buffered outputs (stdout, stderr, and generated plots as ImageContent blocks) for an asynchronous or completed execution job in strict chronological order.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabGetCellOutputInput",
  "type": "object",
  "properties": {
    "execution_id": {
      "type": "string",
      "description": "Unique execution identifier."
    },
    "include_images": {
      "type": "boolean",
      "default": true,
      "description": "Whether to return rendered matplotlib/seaborn plots as ImageContent blocks."
    },
    "max_text_lines": {
      "type": "integer",
      "default": 1000,
      "minimum": 10,
      "maximum": 10000,
      "description": "Maximum lines of stdout/stderr to include before applying head-tail truncation."
    },
    "max_output_bytes": {
      "type": "integer",
      "default": 102400,
      "minimum": 1024,
      "maximum": 10485760,
      "description": "Maximum raw output bytes (default: 100 KB) before applying truncation."
    },
    "max_images": {
      "type": "integer",
      "default": 5,
      "minimum": 1,
      "maximum": 50,
      "description": "Maximum visual plot images returned per execution."
    }
  },
  "required": ["execution_id"]
}
```
- **Output Return Format:** Array of MCP Content Blocks strictly preserving chronological kernel emission order (`[TextContent, ImageContent, TextContent]`).

---

#### Tool 12: `colab_interrupt_execution`
- **Description:** Cancel and abort the currently executing cell or snippet on the Colab kernel. Dispatches a non-destructive SIGINT interrupt first (preserving in-memory variables and state). If the kernel is unresponsive after 5.0 seconds, escalates to a kernel restart (destroying in-memory variables and re-establishing the WebSocket tunnel). The output explicitly communicates whether variables were preserved or reset.
- **Input Schema:**
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "ColabInterruptExecutionInput",
  "type": "object",
  "properties": {
    "runtime_id": {
      "type": "string",
      "description": "Optional runtime identifier. Defaults to active runtime."
    }
  },
  "required": []
}
```
- **Output Return Format (JSON):**
  - Case 1: Non-Destructive SIGINT Interrupt (Variables Preserved):
    ```json
    {
      "runtime_id": "colab-f918e7c2-901a-4bc8-812b-2019482019ab",
      "status": "interrupted",
      "state_reset": false,
      "kernel_restarted": false,
      "kernel_state": "idle",
      "warning": null,
      "message": "SIGINT interrupt signal dispatched; execution halted while preserving in-memory state."
    }
    ```
  - Case 2: Escalated Kernel Restart (Variables Cleared):
    ```json
    {
      "runtime_id": "colab-f918e7c2-901a-4bc8-812b-2019482019ab",
      "status": "restarted",
      "state_reset": true,
      "kernel_restarted": true,
      "kernel_state": "idle",
      "warning": "SIGINT did not halt kernel within 5.0s. Kernel was forcefully restarted; all in-memory variables and imported modules have been reset.",
      "message": "Kernel restarted and WebSocket bridge automatically reconnected."
    }
    ```

---

## 8. Security Architecture & Threat Mitigations

### 8.1 Credential Redaction & Zero-Leakage Policy
- **Threat:** Accidental exposure of Google account cookies or OAuth refresh tokens in MCP tool output blocks, system prompts, or debug logs would allow third-party observers to compromise the user's Google Account.
- **Mitigation:**
  1. *Egress Filter:* A global response interceptor inspects all tool return payloads and log streams. Any string matching OAuth access tokens, OAuth refresh tokens, or full Google session cookies is redacted to `[REDACTED]`. The regex covers:
     ```python
     CREDENTIAL_REDACTION_PATTERN = re.compile(
         r'(?:ya29\.[a-zA-Z0-9_\-]+'                                                   # OAuth 2.0 access tokens
         r'|1//0[a-zA-Z0-9_\-]+'                                                       # Google OAuth refresh tokens
         r'|(?:SAPISID|APISID|HSID|SSID|SID|__Secure-[13]PSID|OSID)=[a-zA-Z0-9_\-]+)' # Full Google session cookies
     )

     def redact_credentials(text: str) -> str:
         """Redacts sensitive Google credentials and tokens from text before returning to LLM."""
         return CREDENTIAL_REDACTION_PATTERN.sub('[REDACTED]', text)
     ```
  2. *Strict Object Isolation:* `colab_get_auth_status` exposes only booleans (`is_authenticated: true`) and sanitized metadata (`user_email`). Raw secrets are never returned to the LLM agent.
  3. *Volatile Memory Scrubbing:* On `colab_logout`, in-memory credential objects are overwritten and dereferenced before garbage collection.

### 8.2 Compute Unit Runaway Protection & Watchdog Inactivity Auto-Unassign
- **Threat:** An autonomous agent could trigger an infinite loop or dispatch long model training runs on an A100 GPU (costing ~5.42 CCU/hour), depleting the user's paid Colab balance.
- **Mitigation:**
  1. *Mandatory Execution Timeouts:* `colab_execute_code` enforces a configurable timeout with a hard ceiling of 3,600 seconds (default: 300s). When the timer expires, the server automatically fires an out-of-band `interrupt_request`.
  2. *Watchdog Inactivity Auto-Unassign (30-Minute Idle Ceiling):* The keep-alive daemon tracks `last_activity_time` updated on every cell submission and output retrieval. If no execution occurs for **30 minutes (1,800 seconds)**, the watchdog automatically suspends keep-alive heartbeat pings and invokes `unassign_runtime(endpoint)` to deallocate the cloud VM, completely cutting off CCU consumption.
  3. *Cost Transparency:* The `colab_list_runtime_types` tool explicitly lists hourly CCU burn rates so the agent can inform the user before provisioning high-cost accelerators.

### 8.3 Sandbox Isolation & Local Filesystem Boundaries
- **Threat:** Code generated by the AI agent could attempt to read or modify sensitive files on the user's host machine.
- **Mitigation:**
  - *Cloud-Only Execution:* Code submitted via `colab_execute_code` executes **exclusively inside Google's cloud container VM** in Google Cloud Platform data centers. The local machine hosting `colabmcp` never evaluates the code.
  - *Path Restriction:* `colab_execute_notebook` validates that input and output file paths reside within permitted workspace boundaries, preventing directory traversal attacks (`../../etc/passwd`).

### 8.4 Terminal Escape Injection Sanitization
- **Threat:** Malicious code or adversarial tracebacks could output ANSI terminal escape sequences designed to exploit vulnerabilities in host terminal emulators.
- **Mitigation:** All `stream` and `error` payloads pass through `sanitize_ansi_traceback()` with the ECMA-48 compliant regex before being wrapped into MCP `TextContent` blocks, stripping all CSI sequences, OSC hyperlink payloads (matching across newlines via `(?s:\].*?(?:\x1b\\|\x07))`), VT100 character sets, and 2-byte Fe control sequences:

```python
ANSI_ESCAPE_PATTERN = re.compile(
    r'\x1b(?:'
    r'\[[0-?]*[ -/]*[@-~]'           # CSI sequences (colors, cursor modes, line clears)
    r'|(?s:\].*?(?:\x1b\\|\x07))'    # OSC sequences with DOTALL (hyperlinks, window titles)
    r'|[\(\)][A-Z0-9]'               # VT100 character set designations
    r'|[@-Z\\-_]'                    # 2-byte Fe control sequences
    r')'
)
```

---

## 9. Phased Implementation Roadmap

### 9.1 Roadmap Summary & Milestone Dependencies

```
[M1: Auth & Keystore]
         |
         v
[M2: Runtime Provisioning & Keep-Alive]
         |
         v
[M3: Kernel WebSocket & MIME Pipeline]
         |
         v
[M4: Notebook Execution Engine]
         |
         v
[M5: Dual Transports, CLI & Hardening]
```

### 9.2 Milestone M1: Authentication & Credential Storage
- **Duration:** 2 Weeks
- **Deliverables:**
  1. `colab_mcp.auth.oauth`: Google OAuth 2.0 flows (remote copy-paste with `token_usage="remote"`, local loopback listener, and ambient ADC loader).
  2. `colab_mcp.auth.cdp_browser`: Playwright Chrome CDP connector for headed browser session and cookie extraction.
  3. `colab_mcp.auth.keystore`: OS-native keyring storage (Windows DPAPI, macOS Keychain, Linux Secret Service) with AES-256-GCM file fallback.
  4. MCP Tools: `colab_login`, `colab_get_auth_status`, `colab_logout`.
- **Exit Criteria:**
  - Running `colab_login(method="oauth_remote")` generates a valid Google consent URL.
  - Submitting authorization code exchanges tokens and persists them into OS Keyring.
  - `colab_get_auth_status` accurately verifies token validity and user email.

### 9.3 Milestone M2: Runtime Lifecycle & Provisioning Engine
- **Duration:** 2 Weeks
- **Dependencies:** M1
- **Deliverables:**
  1. `colab_mcp.utils.nbh`: Deterministic 44-character web-safe base64 notebook hash generator.
  2. `colab_mcp.runtime.client`: TFE HTTP REST client handling two-step `/assign`, pre-assignment enumeration re-attachment (`/assignments`), multi-account support (`auth_user_index`, `X-Goog-AuthUser`), and `/unassign` sequences with Free-Tier quota / High-RAM error handling.
  3. `colab_mcp.runtime.keepalive`: Background asyncio heartbeat daemon sending `GET /keep-alive/` every 35s with read timeout tolerance and 30-minute idle watchdog auto-unassign.
  4. MCP Tools: `colab_list_runtime_types`, `colab_connect_runtime`, `colab_get_runtime_status`, `colab_disconnect_runtime`.
- **Exit Criteria:**
  - `colab_connect_runtime(accelerator="T4")` successfully claims a Colab VM and returns valid `runtimeProxyInfo`.
  - Heartbeat daemon keeps session alive past 90-minute standard idle threshold, and automatically disconnects on 30 minutes of inactivity.
  - `colab_disconnect_runtime` cleanly releases the VM container.

### 9.4 Milestone M3: Kernel WebSocket Bridge & Output Pipeline
- **Duration:** 2 Weeks
- **Dependencies:** M2
- **Deliverables:**
  1. `colab_mcp.kernel.client`: WebSocket multiplexer connecting to Colab kernel tunnel with intermediate kernel discovery/creation handshake (`GET`/`POST /api/kernels`), `subprotocols=None`, and dynamic `additional_headers`/`extra_headers` fallback.
  2. `colab_mcp.kernel.correlator`: Message correlator matching `execute_request` msg_id with incoming `iopub` streams.
  3. `colab_mcp.kernel.mime_converter`: Chronological MIME pipeline parsing Matplotlib PNGs into `ImageContent`, output bounds with head-tail truncation, and cleaning ANSI tracebacks with ECMA-48 regex into `TextContent`.
  4. MCP Tools: `colab_execute_code`, `colab_get_execution_status`, `colab_get_cell_output`, `colab_interrupt_execution` (distinguishing SIGINT from escalated restart).
- **Exit Criteria:**
  - Executing `print("Hello Colab")` returns standard output in `TextContent`.
  - Executing Matplotlib plotting snippet returns interleaved chronological `[TextContent, ImageContent, TextContent]`.
  - Interrupting an infinite loop halts execution within 5 seconds without crashing the kernel, or escalates to restart with clear status distinction.

### 9.5 Milestone M4: Notebook Orchestration Engine
- **Duration:** 2 Weeks
- **Dependencies:** M3
- **Deliverables:**
  1. `colab_mcp.notebook.loader`: `nbformat` v4 parser and JSON schema validator supporting explicit `notebook_path` and `notebook_content`.
  2. `colab_mcp.notebook.runner`: Sequential and selective cell index runner (0-based `nb.cells`), optional kernel reset before run (`reset_kernel_before_run`), and `stop_on_error` controls.
  3. Notebook AST serializer writing updated execution counts, structured errors (`error_name`, `error_value`, `traceback`), and outputs back to disk.
  4. MCP Tool: `colab_execute_notebook`.
- **Exit Criteria:**
  - Successfully parses, executes, and updates a 10-cell `.ipynb` notebook.
  - Halts execution at failing cell when `stop_on_error: true`.
  - Persists valid `.ipynb` file to disk verifiable via JupyterLab/VS Code.

### 9.6 Milestone M5: Transports, CLI Packaging & Hardening
- **Duration:** 2 Weeks
- **Dependencies:** M4
- **Deliverables:**
  1. `colab_mcp.server`: FastMCP server supporting dual `stdio` and `sse` transports.
  2. `colab_mcp.utils.logging`: Strict stdout isolation redirecting all diagnostic logging to `sys.stderr`.
  3. `colab_mcp.cli`: Typer CLI application (`colab-mcp run`, `login`, `status`).
  4. PyPI packaging via `hatchling` supporting execution via `uvx colab-mcp`.
  5. Security auditing, credential redaction filters, and end-to-end integration test suite.
- **Exit Criteria:**
  - Claude Desktop and Cursor connect to `colabmcp` via `stdio` configuration without JSON-RPC corruption.
  - MCP Inspector (`npx @modelcontextprotocol/inspector`) passes all tool calls cleanly.
  - End-to-end test suite executes across Windows, macOS, and Linux runners.
