"""
Headless ``google.colab`` shims & runtime system-metrics probes injected over Jupyter.

Prevents Colab browser modals (``userdata.get``, ``drive.mount``,
``auth.authenticate_user``, ``files.upload`` / ``files.download``) from
hanging headless MCP executions.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


def build_colab_bootstrap_code(
    *,
    secrets: Optional[Dict[str, str]] = None,
    env_overrides: Optional[Dict[str, str]] = None,
    removed_secrets: Optional[Iterable[str]] = None,
    access_token: Optional[str] = None,
) -> str:
    """
    Build an idempotent Python snippet that configures ``os.environ`` and installs
    headless ``google.colab`` shims on the remote kernel without polluting output.
    """
    secrets_repr = repr(dict(secrets or {}))
    env_repr = repr(dict(env_overrides or {}))
    removed_repr = repr(list(removed_secrets or []))
    token_repr = repr(access_token)

    return f"""
def _colab_mcp_install_shims():
    import importlib
    import importlib.util
    import os
    import sys

    secrets = {secrets_repr}
    env_overrides = {env_repr}
    removed = {removed_repr}
    access_token = {token_repr}

    store = globals().setdefault("_COLAB_MCP_SECRETS", {{}})
    for r_key in removed:
        store.pop(r_key, None)
        if os.environ.get(r_key) is not None:
            os.environ.pop(r_key, None)

    for s_key, s_val in secrets.items():
        store[s_key] = s_val
        os.environ[s_key] = s_val

    for e_key, e_val in env_overrides.items():
        os.environ[e_key] = e_val

    if access_token:
        globals()["_COLAB_MCP_ACCESS_TOKEN"] = access_token
        os.environ.setdefault("GOOGLE_OAUTH_ACCESS_TOKEN", access_token)
        os.environ.setdefault("CLOUDSDK_AUTH_ACCESS_TOKEN", access_token)

    has_colab = ("google.colab" in sys.modules) or (importlib.util.find_spec("google.colab") is not None)
    if not has_colab:
        return

    # 1. google.colab.userdata shim
    try:
        userdata_mod = sys.modules.get("google.colab.userdata") or importlib.import_module("google.colab.userdata")
        secret_not_found_cls = getattr(userdata_mod, "SecretNotFoundError", LookupError)

        def _headless_userdata_get(key):
            k = str(key)
            val = globals().get("_COLAB_MCP_SECRETS", {{}}).get(k)
            if val is None:
                val = os.environ.get(k)
            if val is None:
                raise secret_not_found_cls(
                    f"Secret {{k!r}} not found in colab-mcp secret vault. "
                    f"Store it first via colab_set_secret(name={{k!r}}, value=...)."
                )
            return val

        userdata_mod.get = _headless_userdata_get
    except Exception:
        pass

    # 2. google.colab.drive shim (Option A: seamless local /content/drive/MyDrive workspace)
    try:
        drive_mod = sys.modules.get("google.colab.drive") or importlib.import_module("google.colab.drive")

        def _headless_drive_mount(mountpoint="/content/drive", force_remount=False, timeout_ms=120000, readonly=False):
            mp = os.path.abspath(str(mountpoint))
            mydrive = os.path.join(mp, "MyDrive")
            os.makedirs(mydrive, exist_ok=True)
            my_drive_space = os.path.join(mp, "My Drive")
            if not os.path.exists(my_drive_space) and hasattr(os, "symlink"):
                try:
                    os.symlink(mydrive, my_drive_space)
                except OSError:
                    pass
            os.environ["COLAB_MCP_DRIVE_MOUNTPOINT"] = mp
            if os.path.ismount(mp):
                return
            print(
                f"[colab-mcp] Headless Drive mount active at {{mydrive}} "
                "(use colab_drive_upload / colab_drive_download to sync with Google Drive)"
            )

        def _headless_drive_unmount(*args, **kwargs):
            print("[colab-mcp] Headless Drive workspace synced.")

        drive_mod.mount = _headless_drive_mount
        drive_mod.flush_and_unmount = _headless_drive_unmount
    except Exception:
        pass

    # 3. google.colab.auth shim
    try:
        auth_mod = sys.modules.get("google.colab.auth") or importlib.import_module("google.colab.auth")

        def _headless_authenticate_user(clear_output=True, project_id=None):
            tok = globals().get("_COLAB_MCP_ACCESS_TOKEN") or os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
            if not tok:
                raise RuntimeError(
                    "No active Google OAuth token in colab-mcp session. Call colab_login() first."
                )
            os.environ["GOOGLE_OAUTH_ACCESS_TOKEN"] = tok
            os.environ["CLOUDSDK_AUTH_ACCESS_TOKEN"] = tok
            if project_id:
                os.environ["GOOGLE_CLOUD_PROJECT"] = str(project_id)
                os.environ["CLOUDSDK_CORE_PROJECT"] = str(project_id)
            print("[colab-mcp] Headless Google authentication active via colab-mcp OAuth session.")

        auth_mod.authenticate_user = _headless_authenticate_user
    except Exception:
        pass

    # 4. google.colab.files shim
    try:
        files_mod = sys.modules.get("google.colab.files") or importlib.import_module("google.colab.files")

        def _headless_files_upload(**kwargs):
            upload_dir = "/content/.colab_mcp_uploads"
            loaded = {{}}
            if os.path.isdir(upload_dir):
                for fname in sorted(os.listdir(upload_dir)):
                    fpath = os.path.join(upload_dir, fname)
                    if os.path.isfile(fpath):
                        with open(fpath, "rb") as fh:
                            loaded[fname] = fh.read()
            if loaded:
                return loaded
            raw_paths = input(
                "[colab-mcp] files.upload() requested file(s). Enter VM path(s) (comma-separated): "
            ).strip()
            if not raw_paths:
                return {{}}
            for raw_p in raw_paths.split(","):
                p = os.path.expanduser(raw_p.strip())
                if not p:
                    continue
                if not os.path.isfile(p):
                    raise FileNotFoundError(f"Upload file not found on VM: {{p}}")
                with open(p, "rb") as fh:
                    loaded[os.path.basename(p)] = fh.read()
            return loaded

        def _headless_files_download(filename):
            abs_path = os.path.abspath(str(filename))
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Cannot download missing file: {{abs_path}}")
            print(
                f"[colab-mcp] File ready for download: {{abs_path}} "
                f"(use colab_download_file(remote_path={{abs_path!r}}, local_path=...))"
            )

        files_mod.upload = _headless_files_upload
        files_mod.download = _headless_files_download
    except Exception:
        pass

_colab_mcp_install_shims()
del _colab_mcp_install_shims
""".lstrip()


def build_system_metrics_probe_code() -> str:
    """
    Build a silent kernel snippet that prints one JSON line with GPU, RAM,
    and disk usage metrics on the connected Colab VM.
    """
    return """
def _colab_mcp_collect_metrics():
    import json
    import os
    import shutil
    import subprocess

    metrics = {
        "gpus": [],
        "ram_total_mb": None,
        "ram_used_mb": None,
        "ram_free_mb": None,
        "ram_utilization_percent": None,
        "disk_path": "/content" if os.path.exists("/content") else os.getcwd(),
        "disk_total_gb": None,
        "disk_used_gb": None,
        "disk_free_gb": None,
        "disk_utilization_percent": None,
    }

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                mem_used = float(parts[3]) if parts[3] not in ("", "N/A", "[N/A]") else None
                mem_total = float(parts[4]) if parts[4] not in ("", "N/A", "[N/A]") else None
                gpu_util = float(parts[2]) if parts[2] not in ("", "N/A", "[N/A]") else None
                temp_c = (
                    float(parts[5])
                    if len(parts) > 5 and parts[5] not in ("", "N/A", "[N/A]")
                    else None
                )
                metrics["gpus"].append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "gpu_utilization_percent": gpu_util,
                    "vram_used_mb": mem_used,
                    "vram_total_mb": mem_total,
                    "temperature_celsius": temp_c,
                })
    except Exception:
        pass

    try:
        import psutil
        vm = psutil.virtual_memory()
        metrics["ram_total_mb"] = round(vm.total / (1024 * 1024), 1)
        metrics["ram_used_mb"] = round(vm.used / (1024 * 1024), 1)
        metrics["ram_free_mb"] = round(vm.available / (1024 * 1024), 1)
        metrics["ram_utilization_percent"] = round(float(vm.percent), 1)
    except Exception:
        try:
            if os.path.isfile("/proc/meminfo"):
                meminfo = {}
                with open("/proc/meminfo", "r", encoding="utf-8") as f:
                    for ln in f:
                        if ":" in ln:
                            k, v = ln.split(":", 1)
                            tok = v.strip().split()[0]
                            if tok.isdigit():
                                meminfo[k.strip()] = int(tok)
                total_kb = meminfo.get("MemTotal", 0)
                avail_kb = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
                if total_kb > 0:
                    used_kb = max(0, total_kb - avail_kb)
                    metrics["ram_total_mb"] = round(total_kb / 1024.0, 1)
                    metrics["ram_used_mb"] = round(used_kb / 1024.0, 1)
                    metrics["ram_free_mb"] = round(avail_kb / 1024.0, 1)
                    metrics["ram_utilization_percent"] = round((used_kb / total_kb) * 100.0, 1)
        except Exception:
            pass

    try:
        du = shutil.disk_usage(metrics["disk_path"])
        gb = 1024.0 ** 3
        metrics["disk_total_gb"] = round(du.total / gb, 2)
        metrics["disk_used_gb"] = round(du.used / gb, 2)
        metrics["disk_free_gb"] = round(du.free / gb, 2)
        metrics["disk_utilization_percent"] = round((du.used / du.total) * 100.0, 1) if du.total > 0 else 0.0
    except Exception:
        pass

    print(json.dumps(metrics))

_colab_mcp_collect_metrics()
del _colab_mcp_collect_metrics
""".lstrip()


__all__ = [
    "build_colab_bootstrap_code",
    "build_system_metrics_probe_code",
]
