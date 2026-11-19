# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| `0.1.x` | Yes |

---

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities, credential-leak bugs, or token-handling flaws.

Instead, report vulnerabilities privately via GitHub's **Security Advisories** ("Report a vulnerability") on [`Treamyracle/colab-mcp`](https://github.com/Treamyracle/colab-mcp/security).

Please include:
- A clear description of the vulnerability and potential impact.
- Minimal steps to reproduce (with any tokens, cookies, or personal identifiers redacted).
- Affected version and OS/Python environment.

---

## Security Architecture Summary

- **Credential Storage:** OAuth tokens and Google session cookies are stored in the host OS keystore (Windows DPAPI, macOS Keychain, Linux Secret Service) or an AES-256-GCM encrypted file with `0600` permissions.
- **Egress Redaction:** `colab_mcp.server` installs an egress filter across tool responses and exceptions, and `colab_mcp.utils.logging` filters diagnostic logs before output.
- **Execution Isolation:** Code and notebooks execute inside remote Google Colab VM containers, never locally on the host.
- **Path Traversal Protection:** Both local notebook/file operations and remote VM file paths reject null bytes and `..` traversal segments.
