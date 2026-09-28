"""Settings API routes for PaperReading.

Provides endpoints for managing API configuration and other settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from starlette.background import BackgroundTask

from app.core.config import settings
from app.services.api_config import (
    APIConfig,
    CONFIG_FILE,
    default_protocol_for,
    fetch_models,
    get_mineru_model_versions,
    get_provider_info,
    load_config,
    save_config,
    test_connection,
    test_mineru_connection,
    update_mineru_config,
)
from app.services.backup_service import (
    build_full_backup,
    build_papers_export,
    estimate_backup_sizes,
    restore_full_backup,
)

router = APIRouter(prefix="/settings", tags=["settings"])


def _dir_size_bytes(path: Path) -> int:
    """Recursively compute total size (bytes) of all files under *path*."""
    if not path.exists():
        return 0
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def _file_size_bytes(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _count_paper_storage_dirs(storage_dir: Path) -> int:
    """Count subdirectories representing individual paper storage."""
    if not storage_dir.exists():
        return 0
    return sum(1 for entry in storage_dir.iterdir() if entry.is_dir())


def _format_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    units = ["KB", "MB", "GB", "TB"]
    size = float(size_bytes) / 1024.0
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_bytes} B"


class APIConfigUpdate(BaseModel):
    provider: str = "deepseek"
    # 调用协议：'openai'（OpenAI 兼容）或 'anthropic'（Claude）。
    # 留空则按所选厂商的预设自动决定。
    protocol: str = ""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"


class ModelsFetchRequest(BaseModel):
    """拉取模型列表的请求体（用未保存的配置试探）。"""
    provider: str = "deepseek"
    protocol: str = ""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"


class MinerUConfigUpdate(BaseModel):
    token: str = ""
    model_version: str = "vlm"
    base_url: str = "https://mineru.net"


class TestResultResponse(BaseModel):
    success: bool
    message: str


@router.get("/api-config")
def get_api_config() -> dict:
    """Get current API configuration (excluding sensitive fields for display)."""
    config = load_config()
    return {
        "provider": config.provider,
        "protocol": config.protocol,
        "api_key": config.api_key,
        "base_url": config.base_url,
        "model": config.model,
        "is_configured": config.is_configured,
        "mineru": {
            "token": config.mineru.token,
            "model_version": config.mineru.model_version,
            "base_url": config.mineru.base_url,
            "is_configured": config.mineru_is_configured,
        },
        "mineru_model_versions": get_mineru_model_versions(),
        "provider_info": get_provider_info(),
    }


@router.put("/api-config")
def update_api_config(payload: APIConfigUpdate) -> dict:
    """Update API configuration."""
    config = load_config()

    # 协议：显式给了就用它，否则按厂商预设决定。
    # 不再校验模型是否在预设列表里——模型可以从接口拉取，也允许自定义。
    protocol = payload.protocol.strip() or default_protocol_for(payload.provider)

    new_config = APIConfig(
        provider=payload.provider,
        protocol=protocol,
        api_key=payload.api_key.strip(),
        base_url=payload.base_url.strip(),
        model=payload.model.strip(),
        mineru=config.mineru,  # Preserve MinerU config
    )
    saved = save_config(new_config)

    # Reset settings to force reload from new config file
    settings.reset()

    return {
        "provider": saved.provider,
        "protocol": saved.protocol,
        "api_key": saved.api_key,
        "base_url": saved.base_url,
        "model": saved.model,
        "is_configured": saved.is_configured,
        "mineru": {
            "token": saved.mineru.token,
            "model_version": saved.mineru.model_version,
            "base_url": saved.mineru.base_url,
            "is_configured": saved.mineru_is_configured,
        },
    }


@router.put("/mineru-config")
def update_mineru_config_api(payload: MinerUConfigUpdate) -> dict:
    """Update MinerU API configuration."""
    # Validate model version
    model_versions = get_mineru_model_versions()
    if payload.model_version not in model_versions:
        raise HTTPException(
            status_code=400,
            detail=f"MinerU 模型版本 '{payload.model_version}' 不受支持，可选: {', '.join(model_versions)}"
        )

    updated = update_mineru_config(
        token=payload.token.strip(),
        model_version=payload.model_version,
        base_url=payload.base_url.strip(),
    )

    # Reset settings to force reload
    settings.reset()

    return {
        "token": updated.token,
        "model_version": updated.model_version,
        "base_url": updated.base_url,
        "is_configured": updated.is_configured,
        "available_model_versions": model_versions,
    }


@router.post("/mineru-config/test", response_model=TestResultResponse)
def test_mineru_config_api(payload: MinerUConfigUpdate) -> TestResultResponse:
    """Test MinerU API connection (without saving)."""
    result = test_mineru_connection(
        token=payload.token.strip(),
        model_version=payload.model_version,
        base_url=payload.base_url.strip(),
    )
    return TestResultResponse(**result)


@router.post("/api-config/test", response_model=TestResultResponse)
def test_api_config(payload: APIConfigUpdate) -> TestResultResponse:
    """Test API connection with the given configuration (without saving)."""
    config = APIConfig(
        provider=payload.provider,
        protocol=payload.protocol.strip() or default_protocol_for(payload.provider),
        api_key=payload.api_key.strip(),
        base_url=payload.base_url.strip(),
        model=payload.model.strip(),
    )
    result = test_connection(config)
    return TestResultResponse(**result)


@router.post("/api-config/models")
def fetch_api_models(payload: ModelsFetchRequest) -> dict:
    """从所选厂商的接口拉取可用模型列表（不保存配置）。"""
    return fetch_models(
        api_key=payload.api_key.strip(),
        base_url=payload.base_url.strip(),
        protocol=payload.protocol.strip() or default_protocol_for(payload.provider),
    )


@router.get("/providers")
def get_providers() -> dict:
    """Get available providers and their models."""
    return get_provider_info()


@router.get("/mineru-model-versions")
def get_mineru_versions() -> dict:
    """Get available MinerU model versions."""
    return {
        "model_versions": get_mineru_model_versions(),
        "default": "vlm",
        "descriptions": {
            "pipeline": "传统管道模型，速度快，适合简单文档",
            "vlm": "视觉语言模型（推荐），精度高，支持复杂文档",
            "MinerU-HTML": "专为HTML文件解析优化",
        },
    }


@router.get("/storage-info")
def get_storage_info() -> dict:
    """Get storage usage breakdown for the workspace.

    Splits total usage into three categories:
    - papers:  workspace/storage/  (original PDFs + MinerU markdown per paper)
    - logs:    workspace/task_logs/ + workspace/debug_logs/
    - system:  paperreading.db + api_config.json
    """
    workspace = settings.workspace_dir
    storage_dir = workspace / "storage"
    task_logs_dir = workspace / "task_logs"
    debug_logs_dir = workspace / "debug_logs"
    db_path = settings.db_path
    api_config_path = CONFIG_FILE

    # Papers storage (original.pdf + mineru/ per paper_id)
    papers_size = _dir_size_bytes(storage_dir)
    papers_count = _count_paper_storage_dirs(storage_dir)

    # Log files
    task_logs_size = _dir_size_bytes(task_logs_dir)
    debug_logs_size = _dir_size_bytes(debug_logs_dir)
    logs_size = task_logs_size + debug_logs_size

    # System files
    db_size = _file_size_bytes(db_path)
    api_config_size = _file_size_bytes(api_config_path)
    system_size = db_size + api_config_size

    total_size = papers_size + logs_size + system_size

    return {
        "papers": {
            "size_bytes": papers_size,
            "size_display": _format_size(papers_size),
            "count": papers_count,
            "path": str(storage_dir),
        },
        "logs": {
            "size_bytes": logs_size,
            "size_display": _format_size(logs_size),
        },
        "system": {
            "size_bytes": system_size,
            "size_display": _format_size(system_size),
        },
        "total": {
            "size_bytes": total_size,
            "size_display": _format_size(total_size),
        },
        "workspace_path": str(workspace),
    }


def _count_and_delete_files_in_dirs(paths: List[Path], pattern: str) -> Tuple[int, int]:
    """Delete files matching *pattern* within each directory path without recursion.

    Returns (count_deleted, bytes_freed).
    """
    count = 0
    freed = 0
    for p in paths:
        if not p.exists() or not p.is_dir():
            continue
        for entry in p.glob(pattern):
            if not entry.is_file():
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            try:
                entry.unlink()
                count += 1
                freed += size
            except OSError:
                continue
    return count, freed


def _delete_mineru_cache(storage_dir: Path) -> tuple[int, int]:
    """Delete MinerU intermediate cache (mineru/ sub-dirs per paper).

    Returns (count_deleted, bytes_freed).
    """
    if not storage_dir.exists():
        return 0, 0
    count = 0
    freed = 0
    for paper_dir in storage_dir.iterdir():
        if not paper_dir.is_dir():
            continue
        mineru_dir = paper_dir / "mineru"
        if not mineru_dir.exists() or not mineru_dir.is_dir():
            continue
        # Recursively sum size and delete files inside
        for entry in mineru_dir.rglob("*"):
            if entry.is_file():
                try:
                    freed += entry.stat().st_size
                except OSError:
                    pass
                try:
                    entry.unlink()
                    count += 1
                except OSError:
                    pass
        # Remove now-empty mineru dir
        try:
            # Delete nested empty subdirs
            for sub in sorted(mineru_dir.rglob("*"), key=lambda s: -len(str(s))):
                if sub.is_dir():
                    try:
                        sub.rmdir()
                    except OSError:
                        pass
            mineru_dir.rmdir()
        except OSError:
            pass
    return count, freed


@router.post("/storage/clear-logs")
def clear_storage_logs() -> dict:
    """Clear task_logs and debug_logs under the workspace."""
    workspace = settings.workspace_dir
    task_logs_dir = workspace / "task_logs"
    debug_logs_dir = workspace / "debug_logs"
    count, freed = _count_and_delete_files_in_dirs([task_logs_dir, debug_logs_dir], "*.jsonl")
    return {
        "success": True,
        "count_deleted": count,
        "size_display": _format_size(freed),
    }


@router.post("/storage/clear-cache")
def clear_storage_cache() -> dict:
    """Clear MinerU intermediate cache under workspace/storage/<paper_id>/mineru.

    Original PDFs are preserved.
    """
    workspace = settings.workspace_dir
    storage_dir = workspace / "storage"
    count, freed = _delete_mineru_cache(storage_dir)
    return {
        "success": True,
        "count_deleted": count,
        "size_display": _format_size(freed),
    }


# ========== Backup & Restore ==========


@router.get("/backup/info")
def get_backup_info() -> dict:
    """Estimate backup sizes for full backup and papers-only export."""
    return estimate_backup_sizes()


@router.post("/backup/full")
def create_full_backup() -> Response:
    """Build and stream a complete workspace backup ZIP."""
    try:
        zip_path, filename, _manifest = build_full_backup()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"生成全量备份失败：{exc}") from exc
    # FileResponse 流式读取磁盘文件，并负责在发送完成后删除临时包
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=filename,
        background=BackgroundTask(zip_path.unlink, missing_ok=True),
    )


@router.post("/backup/papers")
def create_papers_export() -> Response:
    """Build and stream a ZIP of original PDFs preserving folder hierarchy."""
    try:
        zip_path, filename, manifest = build_papers_export()
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"导出文献失败：{exc}") from exc
    # Stash a small summary in headers so the UI can show skipped papers count
    # without parsing the binary ZIP.
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=filename,
        headers={
            "X-Exported-Count": str(manifest.get("exported_count", 0)),
            "X-Skipped-Count": str(manifest.get("skipped_count", 0)),
        },
        background=BackgroundTask(zip_path.unlink, missing_ok=True),
    )


@router.post("/restore")
def restore_backup(file: UploadFile = File(...)) -> dict:
    """Validate and apply an uploaded full-backup ZIP in place.

    同步端点：由 FastAPI 放入线程池执行，避免恢复期间阻塞事件循环；
    直接透传上传流，不把整包读进内存。
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="请上传 .zip 格式的备份文件")
    try:
        summary = restore_full_backup(file.file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"恢复备份失败：{exc}") from exc
    return {"success": True, **summary}