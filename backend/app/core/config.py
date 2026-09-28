"""Central configuration for the PaperReading backend.

Keep all runtime paths in one place so the demo remains easy to reason about
and future refactors do not scatter filesystem assumptions across modules.

API configuration is loaded from the workspace/api_config.json file,
with fallback to environment variables for backward compatibility.

部署相关：运行时数据目录（数据库、上传的 PDF、日志、api_config.json）默认在
仓库根的 ``workspace/``，可用环境变量 ``PAPERPILOT_WORKSPACE_DIR`` 覆盖——
容器/云平台（如 Railway）把持久化卷挂到别处时必须用它。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    project_root: Path = Path(__file__).resolve().parents[3]
    data_dir: Path = project_root / "Data"
    workspace_dir: Path = project_root / "workspace"
    db_path: Path = workspace_dir / "paperreading.db"
    schema_path: Path = data_dir / "schema.sql"
    seed_path: Path = data_dir / "seed.sql"

    # Lazy-loaded from API config file or environment
    _api_key: str | None = None
    _base_url: str | None = None
    _model: str | None = None
    _provider: str | None = None
    _protocol: str | None = None
    _mineru_token: str | None = None
    _mineru_model_version: str | None = None
    _mineru_base_url: str | None = None

    def __post_init__(self) -> None:
        # 允许把运行时数据目录指到别处（云平台挂载持久化卷时必需）。
        # 注意：db_path 的默认值是在类定义时按「默认 workspace_dir」算出来的，
        # 所以这里必须一并重算，否则数据库仍会落在默认位置。
        override = os.getenv("PAPERPILOT_WORKSPACE_DIR", "").strip()
        if override:
            self.workspace_dir = Path(override).expanduser().resolve()
            self.db_path = self.workspace_dir / "paperreading.db"

    def _load_api_config(self) -> None:
        """Load API configuration from the config file."""
        if self._api_key is not None:
            return  # Already loaded

        try:
            from app.services.api_config import load_config
            config = load_config()
            self._provider = config.provider
            self._protocol = getattr(config, "protocol", "") or ""
            self._api_key = config.api_key or os.getenv("DEEPSEEK_API_KEY", "")
            self._base_url = config.base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self._model = config.model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
            # MinerU config
            self._mineru_token = config.mineru.token or os.getenv("MINERU_API_TOKEN", "")
            self._mineru_model_version = config.mineru.model_version or os.getenv("MINERU_MODEL_VERSION", "vlm")
            self._mineru_base_url = config.mineru.base_url or os.getenv("MINERU_BASE_URL", "https://mineru.net")
        except Exception:
            # Fallback to environment variables
            self._provider = os.getenv("DEEPSEEK_PROVIDER", "deepseek")
            self._protocol = os.getenv("LLM_PROTOCOL", "")
            self._api_key = os.getenv("DEEPSEEK_API_KEY", "")
            self._base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            self._model = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
            self._mineru_token = os.getenv("MINERU_API_TOKEN", "")
            self._mineru_model_version = os.getenv("MINERU_MODEL_VERSION", "vlm")
            self._mineru_base_url = os.getenv("MINERU_BASE_URL", "https://mineru.net")

    @property
    def provider(self) -> str:
        self._load_api_config()
        return self._provider or "deepseek"

    # ===== 通用大模型配置（支持多家厂商 / 两种协议）=====
    # 业务代码统一用 llm_*；deepseek_* 保留为别名，避免大范围改名。

    @property
    def llm_api_key(self) -> str:
        self._load_api_config()
        return self._api_key or ""

    @property
    def llm_base_url(self) -> str:
        self._load_api_config()
        return self._base_url or "https://api.deepseek.com"

    @property
    def llm_model(self) -> str:
        self._load_api_config()
        return self._model or "deepseek-flash"

    @property
    def llm_protocol(self) -> str:
        """调用协议：'openai'（OpenAI 兼容）或 'anthropic'（Claude）。"""
        self._load_api_config()
        return self._protocol or "openai"

    @property
    def deepseek_api_key(self) -> str:
        return self.llm_api_key

    @property
    def deepseek_base_url(self) -> str:
        return self.llm_base_url

    @property
    def deepseek_model(self) -> str:
        return self.llm_model

    @property
    def mineru_token(self) -> str:
        self._load_api_config()
        return self._mineru_token or ""

    @property
    def mineru_model_version(self) -> str:
        self._load_api_config()
        return self._mineru_model_version or "vlm"

    @property
    def mineru_base_url(self) -> str:
        self._load_api_config()
        return self._mineru_base_url or "https://mineru.net"

    def reset(self) -> None:
        """Reset cached API configuration so it will be reloaded from disk."""
        self._api_key = None
        self._base_url = None
        self._model = None
        self._provider = None
        self._protocol = None
        self._mineru_token = None
        self._mineru_model_version = None
        self._mineru_base_url = None


settings = Settings()