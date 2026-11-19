"""
Pydantic response models for Colab VM file transfer and Google Drive tools.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RemoteFileEntry(BaseModel):
    """Metadata for a file or directory on a connected Colab runtime."""

    name: str
    path: str
    type: str = "file"
    size_bytes: Optional[int] = None
    last_modified: Optional[str] = None
    mimetype: Optional[str] = None


class ListFilesResponse(BaseModel):
    """Response schema for colab_list_files."""

    runtime_id: str
    remote_path: str
    entries: List[RemoteFileEntry] = Field(default_factory=list)
    total_entries: int = 0


class DownloadFileResponse(BaseModel):
    """Response schema for colab_download_file."""

    status: str = "downloaded"
    runtime_id: str
    remote_path: str
    local_path: str
    size_bytes: int
    sha256: str
    transfer_method: str


class UploadFileResponse(BaseModel):
    """Response schema for colab_upload_file."""

    status: str = "uploaded"
    runtime_id: str
    local_path: str
    remote_path: str
    size_bytes: int
    sha256: str
    transfer_method: str


class DriveFileEntry(BaseModel):
    """Metadata for a file or folder in Google Drive."""

    id: str
    name: str
    mime_type: str
    size_bytes: Optional[int] = None
    modified_time: Optional[str] = None
    is_folder: bool = False


class DriveListResponse(BaseModel):
    """Response schema for colab_drive_list."""

    folder: str
    files: List[DriveFileEntry] = Field(default_factory=list)
    total_files: int = 0


class DriveTransferResponse(BaseModel):
    """Response schema for colab_drive_download and colab_drive_upload."""

    status: str
    file_id: str
    file_name: str
    size_bytes: int
    sha256: Optional[str] = None
    local_path: Optional[str] = None
    runtime_path: Optional[str] = None
