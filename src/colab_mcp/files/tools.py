"""
FastMCP tools for Colab VM file transfer (list/upload/download) and Google Drive access.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import posixpath
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx
from mcp.server.mcpserver import MCPServer

from colab_mcp.auth.oauth import get_valid_auth_headers
from colab_mcp.files.models import (
    DeleteFileResponse,
    DownloadFileResponse,
    DriveFileEntry,
    DriveListResponse,
    DriveTransferResponse,
    ListFilesResponse,
    RemoteFileEntry,
    UploadFileResponse,
)
from colab_mcp.kernel.tools import get_active_kernel_client
from colab_mcp.runtime.session_manager import get_session_manager
from colab_mcp.utils.logging import get_logger

logger = get_logger(__name__)

_CONTENTS_API_MAX_BYTES = 15 * 1024 * 1024  # 15 MB single-shot threshold
_KERNEL_CHUNK_BYTES = 512 * 1024  # 512 KB base64 chunk size
_PROTECTED_REMOTE_DELETE_PATHS = frozenset({"/", "/content"})
_DRIVE_FILES_URL = "https://www.googleapis.com/drive/v3/files"
_DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"
_DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"


def _validate_local_path(path_str: str) -> Path:
    """Validate a local filesystem path against empty strings, null bytes, and '..' segments."""
    if not isinstance(path_str, str) or not path_str.strip():
        raise ValueError("local_path must be a non-empty string.")
    if "\x00" in path_str:
        raise ValueError("local_path must not contain null bytes.")
    raw = Path(path_str)
    if ".." in raw.parts:
        raise ValueError(f"Path traversal ('..') is not permitted: '{path_str}'.")
    return raw.expanduser().resolve()


def _normalize_remote_posix(remote_path: str) -> str:
    """Validate and normalize a remote POSIX path on the Colab VM."""
    if not isinstance(remote_path, str) or not remote_path.strip():
        raise ValueError("remote_path must be a non-empty string.")
    if "\x00" in remote_path:
        raise ValueError("remote_path must not contain null bytes.")
    cleaned = remote_path.strip().replace("\\", "/")
    parts = [p for p in cleaned.split("/") if p]
    if ".." in parts:
        raise ValueError(f"Path traversal ('..') is not permitted in remote_path: '{remote_path}'.")
    if not cleaned.startswith("/"):
        return posixpath.normpath(posixpath.join("/content", cleaned))
    return posixpath.normpath(cleaned)


def _to_contents_rel_path(abs_remote: str) -> str:
    """Return POSIX path relative to Jupyter server root '/' for /api/contents/."""
    return abs_remote.lstrip("/")


def _proxy_headers(proxy_token: str, client_agent: str = "colab-mcp") -> Dict[str, str]:
    return {
        "X-Colab-Runtime-Proxy-Token": proxy_token,
        "X-Colab-Client-Agent": client_agent,
        "Accept": "application/json",
    }


def _resolve_quota_project(quota_project_id: Optional[str]) -> Optional[str]:
    if quota_project_id is not None:
        cleaned = quota_project_id.strip()
        if cleaned:
            return cleaned
    env_proj = os.environ.get("COLAB_MCP_GCP_PROJECT", "").strip()
    if env_proj:
        return env_proj
    cfg_path = os.path.expanduser("~/.colabmcp/config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            proj = str(data.get("quota_project_id") or "").strip()
            if proj:
                return proj
        except Exception:
            pass
    return None


async def _drive_headers(
    auth_user_index: int,
    quota_project_id: Optional[str] = None,
) -> Dict[str, str]:
    headers = dict(await get_valid_auth_headers(auth_user_index))
    headers["Accept"] = "application/json"
    proj = _resolve_quota_project(quota_project_id)
    if proj:
        headers["X-Goog-User-Project"] = proj
    return headers


def _raise_for_drive_error(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    text = resp.text or ""
    lowered = text.lower()
    if resp.status_code == 403 and "quota project" in lowered:
        raise ValueError(
            "Google Drive API requires a GCP quota project (HTTP 403). "
            "Pass quota_project_id='<your-gcp-project-id>' or set COLAB_MCP_GCP_PROJECT, "
            "or transfer files directly between your Colab VM and local disk using "
            "colab_download_file / colab_upload_file."
        )
    if resp.status_code == 403 and ("insufficient" in lowered or "scope" in lowered or "permission" in lowered):
        raise ValueError(
            "Google Drive API denied access due to insufficient OAuth scope (HTTP 403). "
            "Google's public OAuth client grants `drive.file` access rather than full `drive` read/write; "
            "upload files first with `colab_drive_upload`, configure your own OAuth client in "
            "`~/.colabmcp/config.json`, or transfer directly via `colab_download_file` / `colab_upload_file`."
        )
    resp.raise_for_status()


async def _ensure_contents_dirs(
    http: httpx.AsyncClient,
    proxy_url: str,
    headers: Dict[str, str],
    rel_dir: str,
) -> None:
    """Create intermediate directories under /content via Jupyter Contents API."""
    if not rel_dir:
        return
    accum = ""
    for part in rel_dir.split("/"):
        if not part:
            continue
        accum = f"{accum}/{part}" if accum else part
        url = f"{proxy_url.rstrip('/')}/api/contents/{quote(accum, safe='/')}"
        r = await http.get(url, headers=headers)
        if r.status_code == 404:
            put_resp = await http.put(url, headers=headers, json={"type": "directory"})
            put_resp.raise_for_status()


async def _kernel_exec_json(code: str, runtime_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute a small JSON-emitting helper snippet on the active Colab kernel."""
    client = await get_active_kernel_client(runtime_id=runtime_id)
    raw = await client.execute_raw(code=code, timeout_seconds=120, store_history=False)
    if raw.has_error:
        raise RuntimeError(f"Kernel file operation failed: {raw.error_name}: {raw.error_value}")
    stdout_parts = [
        str(msg.get("content", {}).get("text", ""))
        for msg in raw.iopub_messages
        if (msg.get("header", {}).get("msg_type") or msg.get("msg_type")) == "stream"
        and msg.get("content", {}).get("name", "stdout") == "stdout"
    ]
    joined = "".join(stdout_parts).strip()
    if not joined:
        raise RuntimeError("Kernel file operation produced no output.")
    return json.loads(joined.splitlines()[-1])


async def list_files_impl(
    remote_path: str = "/content",
    runtime_id: Optional[str] = None,
) -> ListFilesResponse:
    """Core logic for colab_list_files."""
    abs_remote = _normalize_remote_posix(remote_path)
    mgr = get_session_manager()
    params = await mgr.get_kernel_connection_params(runtime_id=runtime_id)
    rel = _to_contents_rel_path(abs_remote)

    entries: List[RemoteFileEntry] = []
    if rel is not None:
        url = f"{params.proxy_url.rstrip('/')}/api/contents/{quote(rel, safe='/')}?content=1"
        headers = _proxy_headers(params.proxy_token, params.client_agent)
        async with httpx.AsyncClient(timeout=20.0) as http:
            resp = await http.get(url, headers=headers)
            if resp.status_code == 404:
                raise FileNotFoundError(f"Remote path not found: {abs_remote}")
            resp.raise_for_status()
            payload = resp.json()

        if payload.get("type") != "directory":
            entries.append(
                RemoteFileEntry(
                    name=str(payload.get("name") or posixpath.basename(abs_remote)),
                    path=abs_remote,
                    type=str(payload.get("type") or "file"),
                    size_bytes=payload.get("size"),
                    last_modified=payload.get("last_modified"),
                    mimetype=payload.get("mimetype"),
                )
            )
        else:
            for item in payload.get("content") or []:
                item_name = str(item.get("name", ""))
                entries.append(
                    RemoteFileEntry(
                        name=item_name,
                        path=posixpath.join(abs_remote, item_name),
                        type=str(item.get("type") or "file"),
                        size_bytes=item.get("size"),
                        last_modified=item.get("last_modified"),
                        mimetype=item.get("mimetype"),
                    )
                )
    else:
        script = (
            "import json, os\n"
            f"_p = {abs_remote!r}\n"
            "if not os.path.exists(_p):\n"
            "    raise FileNotFoundError(_p)\n"
            "if os.path.isdir(_p):\n"
            "    _out = [{'name': e.name, 'path': os.path.join(_p, e.name), "
            "'type': 'directory' if e.is_dir() else 'file', "
            "'size_bytes': None if e.is_dir() else e.stat().st_size} "
            "for e in sorted(os.scandir(_p), key=lambda x: x.name)]\n"
            "else:\n"
            "    _out = [{'name': os.path.basename(_p), 'path': _p, 'type': 'file', "
            "'size_bytes': os.stat(_p).st_size}]\n"
            "print(json.dumps({'entries': _out}))\n"
        )
        data = await _kernel_exec_json(script, runtime_id=params.endpoint)
        entries = [RemoteFileEntry(**item) for item in data.get("entries", [])]

    return ListFilesResponse(
        runtime_id=params.endpoint,
        remote_path=abs_remote,
        entries=entries,
        total_entries=len(entries),
    )


async def _download_via_kernel_chunks(
    abs_remote: str,
    dest: Path,
    runtime_id: str,
) -> tuple[int, str]:
    """Stream a remote file in 512 KB base64 chunks through the Jupyter kernel."""
    meta = await _kernel_exec_json(
        "import hashlib, json, os\n"
        f"_p = {abs_remote!r}\n"
        "if not os.path.isfile(_p):\n"
        "    raise FileNotFoundError(_p)\n"
        "h = hashlib.sha256()\n"
        "sz = 0\n"
        "with open(_p, 'rb') as fh:\n"
        "    for chunk in iter(lambda: fh.read(1048576), b''):\n"
        "        h.update(chunk)\n"
        "        sz += len(chunk)\n"
        "print(json.dumps({'size': sz, 'sha256': h.hexdigest()}))\n",
        runtime_id=runtime_id,
    )
    total_size = int(meta["size"])
    expected_sha = str(meta["sha256"])

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + ".tmp")
    hasher = hashlib.sha256()
    written = 0
    try:
        with tmp_path.open("wb") as out_f:
            offset = 0
            while offset < total_size or total_size == 0:
                chunk_res = await _kernel_exec_json(
                    "import base64, json\n"
                    f"with open({abs_remote!r}, 'rb') as fh:\n"
                    f"    fh.seek({offset})\n"
                    f"    _b = fh.read({_KERNEL_CHUNK_BYTES})\n"
                    "print(json.dumps({'b64': base64.b64encode(_b).decode('ascii')}))\n",
                    runtime_id=runtime_id,
                )
                raw_bytes = base64.b64decode(chunk_res["b64"])
                out_f.write(raw_bytes)
                hasher.update(raw_bytes)
                written += len(raw_bytes)
                offset += len(raw_bytes)
                if not raw_bytes or total_size == 0:
                    break
        if hasher.hexdigest() != expected_sha:
            raise IOError(f"SHA-256 mismatch downloading {abs_remote}.")
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    return written, expected_sha


async def download_file_impl(
    remote_path: str,
    local_path: str,
    runtime_id: Optional[str] = None,
) -> DownloadFileResponse:
    """Core logic for colab_download_file."""
    abs_remote = _normalize_remote_posix(remote_path)
    dest = _validate_local_path(local_path)
    if dest.exists() and dest.is_dir():
        raise ValueError(f"local_path points to an existing directory: '{dest}'.")

    mgr = get_session_manager()
    params = await mgr.get_kernel_connection_params(runtime_id=runtime_id)
    rel = _to_contents_rel_path(abs_remote)

    if rel is not None and rel != "":
        meta_url = f"{params.proxy_url.rstrip('/')}/api/contents/{quote(rel, safe='/')}?content=0"
        headers = _proxy_headers(params.proxy_token, params.client_agent)
        async with httpx.AsyncClient(timeout=30.0) as http:
            meta_resp = await http.get(meta_url, headers=headers)
            if meta_resp.status_code == 404:
                raise FileNotFoundError(f"Remote file not found: {abs_remote}")
            meta_resp.raise_for_status()
            meta = meta_resp.json()
            if meta.get("type") == "directory":
                raise ValueError(f"remote_path '{abs_remote}' is a directory, not a file.")
            size_hint = meta.get("size")
            if size_hint is None or int(size_hint) <= _CONTENTS_API_MAX_BYTES:
                content_url = (
                    f"{params.proxy_url.rstrip('/')}/api/contents/{quote(rel, safe='/')}?content=1"
                )
                dl_resp = await http.get(content_url, headers=headers)
                dl_resp.raise_for_status()
                payload = dl_resp.json()
                fmt = payload.get("format")
                raw_content = payload.get("content") or ""
                if fmt == "base64":
                    data_bytes = base64.b64decode(raw_content)
                elif fmt == "json":
                    data_bytes = json.dumps(raw_content, indent=2).encode("utf-8")
                else:
                    data_bytes = str(raw_content).encode("utf-8")

                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = dest.with_name(dest.name + ".tmp")
                tmp_path.write_bytes(data_bytes)
                os.replace(tmp_path, dest)
                sha = hashlib.sha256(data_bytes).hexdigest()
                return DownloadFileResponse(
                    runtime_id=params.endpoint,
                    remote_path=abs_remote,
                    local_path=str(dest),
                    size_bytes=len(data_bytes),
                    sha256=sha,
                    transfer_method="contents_api",
                )

    size_bytes, sha = await _download_via_kernel_chunks(abs_remote, dest, params.endpoint)
    return DownloadFileResponse(
        runtime_id=params.endpoint,
        remote_path=abs_remote,
        local_path=str(dest),
        size_bytes=size_bytes,
        sha256=sha,
        transfer_method="kernel_chunked",
    )


async def upload_file_impl(
    local_path: str,
    remote_path: str,
    runtime_id: Optional[str] = None,
) -> UploadFileResponse:
    """Core logic for colab_upload_file."""
    src = _validate_local_path(local_path)
    if not src.is_file():
        raise FileNotFoundError(f"Local file not found: '{src}'.")
    abs_remote = _normalize_remote_posix(remote_path)

    data_bytes = src.read_bytes()
    sha = hashlib.sha256(data_bytes).hexdigest()
    size_bytes = len(data_bytes)

    mgr = get_session_manager()
    params = await mgr.get_kernel_connection_params(runtime_id=runtime_id)
    rel = _to_contents_rel_path(abs_remote)

    if rel is not None and rel != "" and size_bytes <= _CONTENTS_API_MAX_BYTES:
        headers = _proxy_headers(params.proxy_token, params.client_agent)
        async with httpx.AsyncClient(timeout=30.0) as http:
            rel_dir = posixpath.dirname(rel)
            await _ensure_contents_dirs(http, params.proxy_url, headers, rel_dir)
            url = f"{params.proxy_url.rstrip('/')}/api/contents/{quote(rel, safe='/')}"
            put_resp = await http.put(
                url,
                headers=headers,
                json={
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(data_bytes).decode("ascii"),
                },
            )
            put_resp.raise_for_status()
        return UploadFileResponse(
            runtime_id=params.endpoint,
            local_path=str(src),
            remote_path=abs_remote,
            size_bytes=size_bytes,
            sha256=sha,
            transfer_method="contents_api",
        )

    tmp_remote = f"{abs_remote}.tmp"
    await _kernel_exec_json(
        "import json, os\n"
        f"os.makedirs(os.path.dirname({tmp_remote!r}) or '.', exist_ok=True)\n"
        f"open({tmp_remote!r}, 'wb').close()\n"
        "print(json.dumps({'ok': True}))\n",
        runtime_id=params.endpoint,
    )
    for offset in range(0, size_bytes, _KERNEL_CHUNK_BYTES):
        chunk_b64 = base64.b64encode(data_bytes[offset : offset + _KERNEL_CHUNK_BYTES]).decode("ascii")
        await _kernel_exec_json(
            "import base64, json\n"
            f"with open({tmp_remote!r}, 'ab') as fh:\n"
            f"    fh.write(base64.b64decode({chunk_b64!r}))\n"
            "print(json.dumps({'ok': True}))\n",
            runtime_id=params.endpoint,
        )
    verify = await _kernel_exec_json(
        "import hashlib, json, os\n"
        "h = hashlib.sha256()\n"
        f"with open({tmp_remote!r}, 'rb') as fh:\n"
        "    for b in iter(lambda: fh.read(1048576), b''):\n"
        "        h.update(b)\n"
        f"if h.hexdigest() != {sha!r}:\n"
        "    raise ValueError('Upload SHA-256 mismatch')\n"
        f"os.replace({tmp_remote!r}, {abs_remote!r})\n"
        "print(json.dumps({'ok': True}))\n",
        runtime_id=params.endpoint,
    )
    _ = verify
    return UploadFileResponse(
        runtime_id=params.endpoint,
        local_path=str(src),
        remote_path=abs_remote,
        size_bytes=size_bytes,
        sha256=sha,
        transfer_method="kernel_chunked",
    )


async def delete_file_impl(
    remote_path: str,
    recursive: bool = False,
    runtime_id: Optional[str] = None,
) -> DeleteFileResponse:
    """Core logic for colab_delete_file."""
    abs_remote = _normalize_remote_posix(remote_path)
    if abs_remote in _PROTECTED_REMOTE_DELETE_PATHS:
        raise ValueError(f"Refusing to delete protected root directory: '{abs_remote}'.")

    mgr = get_session_manager()
    params = await mgr.get_kernel_connection_params(runtime_id=runtime_id)
    rel = _to_contents_rel_path(abs_remote)

    if rel:
        url = f"{params.proxy_url.rstrip('/')}/api/contents/{quote(rel, safe='/')}"
        headers = _proxy_headers(params.proxy_token, params.client_agent)
        async with httpx.AsyncClient(timeout=30.0) as http:
            del_resp = await http.delete(url, headers=headers)
            if del_resp.status_code == 404:
                raise FileNotFoundError(f"Remote path not found: {abs_remote}")
            if del_resp.status_code in (200, 204):
                return DeleteFileResponse(
                    status="deleted",
                    runtime_id=params.endpoint,
                    remote_path=abs_remote,
                    recursive=recursive,
                    delete_method="contents_api",
                )
            if del_resp.status_code == 400:
                if not recursive:
                    raise ValueError(
                        f"Remote directory '{abs_remote}' is not empty. "
                        "Pass recursive=True to remove it recursively."
                    )
            else:
                del_resp.raise_for_status()

    script = (
        "import json, os, shutil\n"
        f"_p = {abs_remote!r}\n"
        f"_rec = {bool(recursive)!r}\n"
        "if not os.path.lexists(_p):\n"
        "    raise FileNotFoundError(_p)\n"
        "if os.path.isdir(_p) and not os.path.islink(_p):\n"
        "    if _rec:\n"
        "        shutil.rmtree(_p)\n"
        "    else:\n"
        "        os.rmdir(_p)\n"
        "else:\n"
        "    os.unlink(_p)\n"
        "print(json.dumps({'deleted': _p}))\n"
    )
    await _kernel_exec_json(script, runtime_id=params.endpoint)
    return DeleteFileResponse(
        status="deleted",
        runtime_id=params.endpoint,
        remote_path=abs_remote,
        recursive=recursive,
        delete_method="kernel_recursive" if recursive else "kernel_unlink",
    )


async def _resolve_drive_folder_id(
    http: httpx.AsyncClient,
    headers: Dict[str, str],
    folder_path: str,
    create_missing: bool = False,
) -> str:
    """Resolve a Drive path like 'MyDrive/SKRIPSI_NER_BAB4' to its Drive folder ID."""
    cleaned = folder_path.strip().strip("/")
    if not cleaned or cleaned in ("MyDrive", "My Drive", "root"):
        return "root"
    parts = [p for p in cleaned.split("/") if p]
    if parts and parts[0] in ("MyDrive", "My Drive", "root"):
        parts = parts[1:]

    parent_id = "root"
    for seg in parts:
        safe_seg = seg.replace("'", "\\'")
        q = (
            f"name = '{safe_seg}' and '{parent_id}' in parents "
            f"and mimeType = '{_DRIVE_FOLDER_MIME}' and trashed = false"
        )
        resp = await http.get(
            _DRIVE_FILES_URL,
            params={"q": q, "fields": "files(id,name)", "pageSize": 5},
            headers=headers,
        )
        _raise_for_drive_error(resp)
        files = resp.json().get("files", [])
        if files:
            parent_id = str(files[0]["id"])
        elif create_missing:
            create_resp = await http.post(
                _DRIVE_FILES_URL,
                headers=headers,
                json={"name": seg, "mimeType": _DRIVE_FOLDER_MIME, "parents": [parent_id]},
            )
            _raise_for_drive_error(create_resp)
            parent_id = str(create_resp.json()["id"])
        else:
            raise FileNotFoundError(f"Google Drive folder not found: '{folder_path}' (missing '{seg}').")
    return parent_id


async def drive_list_impl(
    folder: str = "MyDrive",
    query: Optional[str] = None,
    max_results: int = 50,
    auth_user_index: int = 0,
    quota_project_id: Optional[str] = None,
) -> DriveListResponse:
    """Core logic for colab_drive_list."""
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not (1 <= max_results <= 200):
        raise ValueError("max_results must be an integer between 1 and 200.")
    headers = await _drive_headers(auth_user_index, quota_project_id)
    async with httpx.AsyncClient(timeout=20.0) as http:
        folder_id = await _resolve_drive_folder_id(http, headers, folder, create_missing=False)
        q_parts = [f"'{folder_id}' in parents", "trashed = false"]
        if query and query.strip():
            safe_q = query.strip().replace("'", "\\'")
            q_parts.append(f"name contains '{safe_q}'")
        resp = await http.get(
            _DRIVE_FILES_URL,
            params={
                "q": " and ".join(q_parts),
                "pageSize": max_results,
                "fields": "files(id,name,mimeType,size,modifiedTime)",
                "orderBy": "folder,modifiedTime desc",
            },
            headers=headers,
        )
        _raise_for_drive_error(resp)
        items = resp.json().get("files", [])

    entries = [
        DriveFileEntry(
            id=str(item["id"]),
            name=str(item["name"]),
            mime_type=str(item.get("mimeType", "")),
            size_bytes=int(item["size"]) if item.get("size") is not None else None,
            modified_time=item.get("modifiedTime"),
            is_folder=(item.get("mimeType") == _DRIVE_FOLDER_MIME),
        )
        for item in items
    ]
    return DriveListResponse(folder=folder, files=entries, total_files=len(entries))


async def drive_download_impl(
    file_id_or_path: str,
    local_path: Optional[str] = None,
    runtime_path: Optional[str] = None,
    auth_user_index: int = 0,
    runtime_id: Optional[str] = None,
    quota_project_id: Optional[str] = None,
) -> DriveTransferResponse:
    """Download a file from Google Drive to local disk and/or the connected Colab VM."""
    if not local_path and not runtime_path:
        raise ValueError("Specify at least one destination: local_path or runtime_path.")

    headers = await _drive_headers(auth_user_index, quota_project_id)
    async with httpx.AsyncClient(timeout=120.0) as http:
        target_id = file_id_or_path.strip()
        if "/" in target_id:
            folder_part, file_name = target_id.rsplit("/", 1)
            folder_id = await _resolve_drive_folder_id(http, headers, folder_part, create_missing=False)
            safe_name = file_name.replace("'", "\\'")
            search = await http.get(
                _DRIVE_FILES_URL,
                params={
                    "q": f"name = '{safe_name}' and '{folder_id}' in parents and trashed = false",
                    "fields": "files(id,name)",
                    "pageSize": 1,
                },
                headers=headers,
            )
            _raise_for_drive_error(search)
            matches = search.json().get("files", [])
            if not matches:
                raise FileNotFoundError(f"Drive file not found: '{file_id_or_path}'.")
            target_id = str(matches[0]["id"])

        meta_resp = await http.get(
            f"{_DRIVE_FILES_URL}/{quote(target_id, safe='')}",
            params={"fields": "id,name,mimeType,size"},
            headers=headers,
        )
        _raise_for_drive_error(meta_resp)
        meta = meta_resp.json()
        file_name = str(meta["name"])

        dl_resp = await http.get(
            f"{_DRIVE_FILES_URL}/{quote(target_id, safe='')}?alt=media",
            headers=headers,
        )
        _raise_for_drive_error(dl_resp)
        data_bytes = dl_resp.content

    sha = hashlib.sha256(data_bytes).hexdigest()
    saved_local: Optional[str] = None
    saved_remote: Optional[str] = None

    if local_path:
        dest = _validate_local_path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(data_bytes)
        os.replace(tmp, dest)
        saved_local = str(dest)

    if runtime_path:
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(data_bytes)
            temp_src = tf.name
        try:
            up_res = await upload_file_impl(temp_src, runtime_path, runtime_id=runtime_id)
            saved_remote = up_res.remote_path
        finally:
            Path(temp_src).unlink(missing_ok=True)

    return DriveTransferResponse(
        status="downloaded",
        file_id=target_id,
        file_name=file_name,
        size_bytes=len(data_bytes),
        sha256=sha,
        local_path=saved_local,
        runtime_path=saved_remote,
    )


async def drive_upload_impl(
    local_path: Optional[str] = None,
    runtime_path: Optional[str] = None,
    drive_folder: str = "MyDrive",
    file_name: Optional[str] = None,
    auth_user_index: int = 0,
    runtime_id: Optional[str] = None,
    quota_project_id: Optional[str] = None,
) -> DriveTransferResponse:
    """Upload a file from local disk or the connected Colab VM directly to Google Drive."""
    if not local_path and not runtime_path:
        raise ValueError("Specify either local_path or runtime_path as the upload source.")
    if local_path and runtime_path:
        raise ValueError("Provide either local_path OR runtime_path, not both.")

    src_local: Optional[Path] = None
    cleanup_tmp = False
    if local_path:
        src_local = _validate_local_path(local_path)
        if not src_local.is_file():
            raise FileNotFoundError(f"Local file not found: '{src_local}'.")
        resolved_name = file_name or src_local.name
        data_bytes = src_local.read_bytes()
    else:
        import tempfile

        assert runtime_path is not None
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            temp_dl = tf.name
        cleanup_tmp = True
        try:
            dl_res = await download_file_impl(runtime_path, temp_dl, runtime_id=runtime_id)
            data_bytes = Path(dl_res.local_path).read_bytes()
            resolved_name = file_name or posixpath.basename(dl_res.remote_path)
        finally:
            if cleanup_tmp:
                Path(temp_dl).unlink(missing_ok=True)

    sha = hashlib.sha256(data_bytes).hexdigest()
    headers = await _drive_headers(auth_user_index, quota_project_id)
    async with httpx.AsyncClient(timeout=120.0) as http:
        folder_id = await _resolve_drive_folder_id(http, headers, drive_folder, create_missing=True)
        boundary = f"colab_mcp_boundary_{os.urandom(8).hex()}"
        metadata_part = json.dumps({"name": resolved_name, "parents": [folder_id]}).encode("utf-8")
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode("utf-8")
            + metadata_part
            + f"\r\n--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n".encode("utf-8")
            + data_bytes
            + f"\r\n--{boundary}--\r\n".encode("utf-8")
        )
        up_headers = {
            **headers,
            "Content-Type": f"multipart/related; boundary={boundary}",
        }
        up_resp = await http.post(
            f"{_DRIVE_UPLOAD_URL}?uploadType=multipart&fields=id,name,size",
            headers=up_headers,
            content=body,
        )
        _raise_for_drive_error(up_resp)
        created = up_resp.json()

    return DriveTransferResponse(
        status="uploaded",
        file_id=str(created["id"]),
        file_name=str(created.get("name", resolved_name)),
        size_bytes=len(data_bytes),
        sha256=sha,
        local_path=str(src_local) if src_local else None,
        runtime_path=runtime_path,
    )


def register_file_tools(mcp: MCPServer) -> None:
    """Register Colab VM file transfer and Google Drive MCP tools on MCPServer."""

    @mcp.tool(
        name="colab_list_files",
        description="List files and directories on the connected Colab VM (e.g. '/content' or '/tmp').",
    )
    async def colab_list_files(
        remote_path: str = "/content",
        runtime_id: Optional[str] = None,
    ) -> ListFilesResponse:
        return await list_files_impl(remote_path=remote_path, runtime_id=runtime_id)

    @mcp.tool(
        name="colab_download_file",
        description=(
            "Download a file (such as an exported .zip, CSV, plot, or model checkpoint) "
            "from the connected Colab VM directly to a local file path."
        ),
    )
    async def colab_download_file(
        remote_path: str,
        local_path: str,
        runtime_id: Optional[str] = None,
    ) -> DownloadFileResponse:
        return await download_file_impl(
            remote_path=remote_path,
            local_path=local_path,
            runtime_id=runtime_id,
        )

    @mcp.tool(
        name="colab_upload_file",
        description="Upload a local file (dataset, archive, or script) to the connected Colab VM.",
    )
    async def colab_upload_file(
        local_path: str,
        remote_path: str,
        runtime_id: Optional[str] = None,
    ) -> UploadFileResponse:
        return await upload_file_impl(
            local_path=local_path,
            remote_path=remote_path,
            runtime_id=runtime_id,
        )

    @mcp.tool(
        name="colab_delete_file",
        description=(
            "Delete a file or directory on the connected Colab VM (pass recursive=True to "
            "remove non-empty directories)."
        ),
    )
    async def colab_delete_file(
        remote_path: str,
        recursive: bool = False,
        runtime_id: Optional[str] = None,
    ) -> DeleteFileResponse:
        return await delete_file_impl(
            remote_path=remote_path,
            recursive=recursive,
            runtime_id=runtime_id,
        )

    @mcp.tool(
        name="colab_drive_list",
        description=(
            "List files and folders in Google Drive (e.g. 'MyDrive' or 'MyDrive/SKRIPSI_NER_BAB4'). "
            "Supports optional quota_project_id if using default OAuth credentials."
        ),
    )
    async def colab_drive_list(
        folder: str = "MyDrive",
        query: Optional[str] = None,
        max_results: int = 50,
        auth_user_index: int = 0,
        quota_project_id: Optional[str] = None,
    ) -> DriveListResponse:
        return await drive_list_impl(
            folder=folder,
            query=query,
            max_results=max_results,
            auth_user_index=auth_user_index,
            quota_project_id=quota_project_id,
        )

    @mcp.tool(
        name="colab_drive_download",
        description=(
            "Download a file from Google Drive (by Drive file ID or 'MyDrive/.../file' path) "
            "to local_path and/or directly onto the connected Colab VM at runtime_path."
        ),
    )
    async def colab_drive_download(
        file_id_or_path: str,
        local_path: Optional[str] = None,
        runtime_path: Optional[str] = None,
        auth_user_index: int = 0,
        runtime_id: Optional[str] = None,
        quota_project_id: Optional[str] = None,
    ) -> DriveTransferResponse:
        return await drive_download_impl(
            file_id_or_path=file_id_or_path,
            local_path=local_path,
            runtime_path=runtime_path,
            auth_user_index=auth_user_index,
            runtime_id=runtime_id,
            quota_project_id=quota_project_id,
        )

    @mcp.tool(
        name="colab_drive_upload",
        description=(
            "Upload a file from local_path or directly from the connected Colab VM (runtime_path) "
            "into Google Drive."
        ),
    )
    async def colab_drive_upload(
        local_path: Optional[str] = None,
        runtime_path: Optional[str] = None,
        drive_folder: str = "MyDrive",
        file_name: Optional[str] = None,
        auth_user_index: int = 0,
        runtime_id: Optional[str] = None,
        quota_project_id: Optional[str] = None,
    ) -> DriveTransferResponse:
        return await drive_upload_impl(
            local_path=local_path,
            runtime_path=runtime_path,
            drive_folder=drive_folder,
            file_name=file_name,
            auth_user_index=auth_user_index,
            runtime_id=runtime_id,
            quota_project_id=quota_project_id,
        )
