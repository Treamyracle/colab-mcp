"""Unit tests for colab_mcp.files (VM file transfer and Google Drive tools)."""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

from colab_mcp.files.tools import (
    download_file_impl,
    drive_download_impl,
    drive_list_impl,
    drive_upload_impl,
    list_files_impl,
    upload_file_impl,
)
from colab_mcp.runtime.models import KernelConnectionParams


def _fake_params() -> KernelConnectionParams:
    return KernelConnectionParams(
        endpoint="gpu-l4-test",
        proxy_url="https://proxy.colab.test",
        proxy_token="proxy-tok",
        token_expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        kernel_id="kern-1",
    )


@pytest.mark.asyncio
async def test_list_upload_download_contents_api(tmp_path: Path, monkeypatch):
    stored_files: dict[str, bytes] = {}
    created_dirs: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert path.startswith("/api/contents")
        rel = path[len("/api/contents"):].lstrip("/")
        if request.method == "GET":
            if rel == "content":
                items = [
                    {"name": "out.zip", "type": "file", "size": len(stored_files.get("content/out.zip", b""))}
                ]
                return httpx.Response(200, json={"type": "directory", "name": "content", "content": items})
            if rel in created_dirs:
                return httpx.Response(200, json={"type": "directory", "name": rel, "content": []})
            if rel in stored_files:
                data = stored_files[rel]
                if "content=0" in str(request.url):
                    return httpx.Response(200, json={"type": "file", "name": rel, "size": len(data)})
                return httpx.Response(
                    200,
                    json={
                        "type": "file",
                        "name": rel,
                        "format": "base64",
                        "size": len(data),
                        "content": base64.b64encode(data).decode("ascii"),
                    },
                )
            return httpx.Response(404)
        if request.method == "PUT":
            body = json.loads(request.content.decode("utf-8"))
            if body.get("type") == "directory":
                created_dirs.add(rel)
                return httpx.Response(201, json={"type": "directory", "name": rel})
            stored_files[rel] = base64.b64decode(body["content"])
            return httpx.Response(201, json={"type": "file", "name": rel})
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: orig_async_client(*args, **{**kwargs, "transport": transport}),
    )

    fake_mgr = AsyncMock()
    fake_mgr.get_kernel_connection_params = AsyncMock(return_value=_fake_params())
    monkeypatch.setattr("colab_mcp.files.tools.get_session_manager", lambda: fake_mgr)

    src = tmp_path / "dataset.zip"
    payload = b"PK\x03\x04hello-colab-zip-bytes"
    src.write_bytes(payload)

    up = await upload_file_impl(str(src), "/content/data/dataset.zip")
    assert up.transfer_method == "contents_api"
    assert up.size_bytes == len(payload)
    assert up.sha256 == hashlib.sha256(payload).hexdigest()
    assert "content/data" in created_dirs
    assert stored_files["content/data/dataset.zip"] == payload

    stored_files["content/out.zip"] = b"PK\x03\x04results-archive"
    listed = await list_files_impl("/content")
    assert listed.total_entries == 1
    assert listed.entries[0].name == "out.zip"

    dst = tmp_path / "downloaded" / "out.zip"
    dl = await download_file_impl("/content/out.zip", str(dst))
    assert dl.transfer_method == "contents_api"
    assert dst.read_bytes() == b"PK\x03\x04results-archive"
    assert dl.sha256 == hashlib.sha256(b"PK\x03\x04results-archive").hexdigest()


@pytest.mark.asyncio
async def test_download_large_or_fallback_uses_kernel_chunks(tmp_path: Path, monkeypatch):
    fake_mgr = AsyncMock()
    fake_mgr.get_kernel_connection_params = AsyncMock(return_value=_fake_params())
    monkeypatch.setattr("colab_mcp.files.tools.get_session_manager", lambda: fake_mgr)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"type": "file", "name": "a.zip", "size": 25 * 1024 * 1024})

    transport = httpx.MockTransport(handler)
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: orig_async_client(*args, **{**kwargs, "transport": transport}),
    )

    raw_data = b"large-or-tmp-zip-content-1234567890"
    expected_sha = hashlib.sha256(raw_data).hexdigest()
    calls = {"n": 0}

    async def fake_kernel_exec(code: str, runtime_id=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"size": len(raw_data), "sha256": expected_sha}
        return {"b64": base64.b64encode(raw_data).decode("ascii")}

    monkeypatch.setattr("colab_mcp.files.tools._kernel_exec_json", fake_kernel_exec)

    dst = tmp_path / "a.zip"
    res = await download_file_impl("/tmp/a.zip", str(dst))
    assert res.transfer_method == "kernel_chunked"
    assert dst.read_bytes() == raw_data
    assert res.sha256 == expected_sha


@pytest.mark.asyncio
async def test_drive_list_upload_download_and_quota_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "colab_mcp.files.tools.get_valid_auth_headers",
        AsyncMock(return_value={"Authorization": "Bearer ya29.test"}),
    )

    drive_blobs: dict[str, bytes] = {"file-1": b"drive-zip-payload"}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.headers.get("X-Goog-User-Project") == "missing-proj-test":
            return httpx.Response(403, json={"error": {"message": "requires a quota project"}})
        if request.url.path == "/drive/v3/files" and request.method == "GET":
            q = request.url.params.get("q", "")
            if "SKRIPSI_NER_BAB4" in q:
                return httpx.Response(200, json={"files": [{"id": "folder-skripsi", "name": "SKRIPSI_NER_BAB4"}]})
            return httpx.Response(
                200,
                json={
                    "files": [
                        {
                            "id": "file-1",
                            "name": "results.zip",
                            "mimeType": "application/zip",
                            "size": str(len(drive_blobs["file-1"])),
                        }
                    ]
                },
            )
        if request.url.path == "/drive/v3/files/file-1":
            if "alt=media" in url:
                return httpx.Response(200, content=drive_blobs["file-1"])
            return httpx.Response(
                200,
                json={"id": "file-1", "name": "results.zip", "mimeType": "application/zip"},
            )
        if request.url.path == "/upload/drive/v3/files" and request.method == "POST":
            return httpx.Response(200, json={"id": "file-up-99", "name": "uploaded.zip", "size": "12"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    orig_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: orig_async_client(*args, **{**kwargs, "transport": transport}),
    )

    listed = await drive_list_impl(folder="MyDrive/SKRIPSI_NER_BAB4", quota_project_id="my-gcp-proj")
    assert listed.total_files == 1
    assert listed.files[0].name == "results.zip"

    dst = tmp_path / "from_drive.zip"
    dl = await drive_download_impl("file-1", local_path=str(dst), quota_project_id="my-gcp-proj")
    assert dst.read_bytes() == b"drive-zip-payload"
    assert dl.file_name == "results.zip"

    src = tmp_path / "to_drive.zip"
    src.write_bytes(b"upload-bytes")
    up = await drive_upload_impl(local_path=str(src), drive_folder="MyDrive", quota_project_id="my-gcp-proj")
    assert up.file_id == "file-up-99"

    with pytest.raises(ValueError, match="quota project"):
        await drive_list_impl(folder="MyDrive", quota_project_id="missing-proj-test")
