"""API 配置的存储与管理。

支持多家厂商：每家预设「默认 Base URL + 调用协议 + 常用模型」。
调用协议分两类：

- ``openai``：OpenAI 兼容接口（DeepSeek / OpenAI / Kimi / 通义 / GLM / Gemini 等）
- ``anthropic``：Anthropic(Claude) 原生接口

用户也可选「自定义」——手填 Base URL 并自行选择协议。
具体请求的构造与解析由 ``app.core.llm_client`` 负责。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.core import llm_client
from app.core.config import settings

PROTOCOL_OPENAI = llm_client.PROTOCOL_OPENAI
PROTOCOL_ANTHROPIC = llm_client.PROTOCOL_ANTHROPIC

MINERU_MODEL_VERSIONS = ["pipeline", "vlm", "MinerU-HTML"]

DEFAULT_MINERU_BASE_URL = "https://mineru.net"
DEFAULT_MINERU_MODEL_VERSION = "vlm"


@dataclass
class MinerUConfig:
    """MinerU API configuration."""
    token: str = ""
    model_version: str = DEFAULT_MINERU_MODEL_VERSION
    base_url: str = DEFAULT_MINERU_BASE_URL

    @property
    def is_configured(self) -> bool:
        return bool(self.token and self.token.strip())


@dataclass
class ProviderPreset:
    """一家厂商的预设：默认地址、调用协议、常用模型。"""
    id: str
    name: str
    protocol: str
    default_base_url: str
    models: list[str] = field(default_factory=list)
    note: str = ""


# 厂商预设。models 只是「常用模型」提示——可以在设置页点「获取模型列表」
# 从该厂商的接口拉取真实可用列表。
PROVIDER_PRESETS: tuple[ProviderPreset, ...] = (
    ProviderPreset(
        "deepseek", "DeepSeek", PROTOCOL_OPENAI,
        "https://api.deepseek.com",
        ["deepseek-flash", "deepseek-v4-pro"],
        "默认",
    ),
    ProviderPreset(
        "openai", "OpenAI", PROTOCOL_OPENAI,
        "https://api.openai.com/v1",
        ["gpt-4o", "gpt-4o-mini"],
    ),
    ProviderPreset(
        "anthropic", "Anthropic (Claude)", PROTOCOL_ANTHROPIC,
        "https://api.anthropic.com",
        ["claude-sonnet-4-5", "claude-opus-4-1"],
    ),
    ProviderPreset(
        "moonshot", "Moonshot (Kimi)", PROTOCOL_OPENAI,
        "https://api.moonshot.cn/v1",
        ["moonshot-v1-8k", "moonshot-v1-32k"],
    ),
    ProviderPreset(
        "qwen", "阿里通义千问", PROTOCOL_OPENAI,
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ["qwen-plus", "qwen-max", "qwen-turbo"],
    ),
    ProviderPreset(
        "zhipu", "智谱 GLM", PROTOCOL_OPENAI,
        "https://open.bigmodel.cn/api/paas/v4",
        ["glm-4-plus", "glm-4-air"],
    ),
    ProviderPreset(
        "gemini", "Google Gemini", PROTOCOL_OPENAI,
        "https://generativelanguage.googleapis.com/v1beta/openai",
        ["gemini-2.0-flash", "gemini-1.5-pro"],
    ),
    ProviderPreset(
        "custom", "自定义", PROTOCOL_OPENAI,
        "", [], "手填 Base URL 并选择协议",
    ),
)

_PRESET_BY_ID: dict[str, ProviderPreset] = {p.id: p for p in PROVIDER_PRESETS}

DEFAULT_PROVIDER_ID = "deepseek"
DEFAULT_PROTOCOL = PROTOCOL_OPENAI
DEFAULT_BASE_URL = _PRESET_BY_ID[DEFAULT_PROVIDER_ID].default_base_url
DEFAULT_MODEL = "deepseek-flash"

# 向后兼容的旧常量名（历史代码/文档可能引用）
DEEPSEEK_MODELS = list(_PRESET_BY_ID["deepseek"].models)
DEFAULT_DEEPSEEK_BASE_URL = DEFAULT_BASE_URL
DEFAULT_DEEPSEEK_MODEL = DEFAULT_MODEL


@dataclass
class APIConfig:
    """Combined API configuration for all providers."""
    provider: str = DEFAULT_PROVIDER_ID
    protocol: str = DEFAULT_PROTOCOL
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    mineru: MinerUConfig = field(default_factory=MinerUConfig)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @property
    def mineru_is_configured(self) -> bool:
        return self.mineru.is_configured


CONFIG_FILE = settings.workspace_dir / "api_config.json"


def _ensure_config_dir() -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _preset(provider: str) -> ProviderPreset | None:
    return _PRESET_BY_ID.get((provider or "").strip())


def default_protocol_for(provider: str) -> str:
    """该厂商的默认协议（未知厂商按 OpenAI 兼容处理）。"""
    preset = _preset(provider)
    return preset.protocol if preset else PROTOCOL_OPENAI


def load_config() -> APIConfig:
    """Load API configuration from the JSON file."""
    if not CONFIG_FILE.exists():
        return APIConfig()
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))

        mineru_data = data.get("mineru", {})
        mineru_config = MinerUConfig(
            token=mineru_data.get("token", ""),
            model_version=mineru_data.get("model_version", DEFAULT_MINERU_MODEL_VERSION),
            base_url=mineru_data.get("base_url", DEFAULT_MINERU_BASE_URL),
        )

        provider = data.get("provider", DEFAULT_PROVIDER_ID)
        # 老配置文件没有 protocol 字段：按其厂商预设补默认值，避免误判成 OpenAI 兼容
        protocol = data.get("protocol") or default_protocol_for(provider)

        return APIConfig(
            provider=provider,
            protocol=llm_client.normalize_protocol(protocol),
            api_key=data.get("api_key", ""),
            base_url=data.get("base_url", DEFAULT_BASE_URL),
            model=data.get("model", DEFAULT_MODEL),
            mineru=mineru_config,
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        return APIConfig()


def save_config(config: APIConfig) -> APIConfig:
    """Save API configuration to the JSON file."""
    _ensure_config_dir()
    data = asdict(config)
    data["protocol"] = llm_client.normalize_protocol(config.protocol)
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return config


def update_mineru_config(
    token: str,
    model_version: str = DEFAULT_MINERU_MODEL_VERSION,
    base_url: str = DEFAULT_MINERU_BASE_URL,
) -> MinerUConfig:
    """Update MinerU configuration."""
    config = load_config()
    config.mineru = MinerUConfig(
        token=token,
        model_version=model_version,
        base_url=base_url or DEFAULT_MINERU_BASE_URL,
    )
    save_config(config)
    return config.mineru


def get_available_models(provider: str) -> list[str]:
    """该厂商的「常用模型」提示列表（真实列表请调 list_models）。"""
    preset = _preset(provider)
    return list(preset.models) if preset else []


def get_mineru_model_versions() -> list[str]:
    """Return available MinerU model versions."""
    return MINERU_MODEL_VERSIONS.copy()


def get_provider_info() -> dict:
    """厂商 / 协议 / 默认值，供前端渲染下拉与自动填充。"""
    return {
        "providers": [
            {
                "id": p.id,
                "name": p.name,
                "protocol": p.protocol,
                "default_base_url": p.default_base_url,
                "models": list(p.models),
                "note": p.note,
            }
            for p in PROVIDER_PRESETS
        ],
        "protocols": [
            {"id": PROTOCOL_OPENAI, "label": "OpenAI 兼容"},
            {"id": PROTOCOL_ANTHROPIC, "label": "Anthropic (Claude)"},
        ],
        "defaults": {
            "provider": DEFAULT_PROVIDER_ID,
            "protocol": DEFAULT_PROTOCOL,
            "base_url": DEFAULT_BASE_URL,
            "model": DEFAULT_MODEL,
        },
        # 兼容旧前端：provider -> 模型列表
        "models": {p.id: list(p.models) for p in PROVIDER_PRESETS},
    }


def _friendly_error(message: str) -> str:
    """把底层 HTTP 错误翻译成用户能看懂的提示。"""
    if "HTTP 401" in message or "HTTP 403" in message:
        return "认证失败：API 密钥无效"
    if "HTTP 404" in message:
        return "API 地址不存在：请检查 Base URL"
    if "HTTP 429" in message:
        return "请求过于频繁：API 限流"
    if "API key not configured" in message:
        return "API 密钥未配置"
    return f"测试失败: {message}"


def test_connection(config: APIConfig) -> dict:
    """用给定配置试调一次，验证连通性（支持两种协议）。"""
    if not config.api_key:
        return {"success": False, "message": "API 密钥未配置"}
    if not config.base_url:
        return {"success": False, "message": "API 地址未配置"}
    try:
        llm_client.chat_completion(
            [{"role": "user", "content": "ping"}],
            temperature=0.0,
            max_tokens=16,
            model=config.model,
            timeout=20,
            api_key=config.api_key,
            base_url=config.base_url,
            protocol=config.protocol,
        )
    except llm_client.LLMError as exc:
        return {"success": False, "message": _friendly_error(str(exc))}
    except Exception as exc:  # noqa: BLE001 - 兜底，避免设置页报 500
        return {"success": False, "message": f"测试失败: {exc}"}
    return {"success": True, "message": "连接成功！"}


def fetch_models(
    api_key: str,
    base_url: str,
    protocol: str,
) -> dict:
    """从厂商接口拉取可用模型列表。"""
    if not api_key:
        return {"success": False, "message": "API 密钥未配置", "models": []}
    if not base_url:
        return {"success": False, "message": "API 地址未配置", "models": []}
    try:
        models = llm_client.list_models(
            api_key=api_key, base_url=base_url, protocol=protocol
        )
    except llm_client.LLMError as exc:
        return {"success": False, "message": _friendly_error(str(exc)), "models": []}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"获取失败: {exc}", "models": []}
    if not models:
        return {"success": False, "message": "该接口没有返回任何模型", "models": []}
    return {"success": True, "message": f"获取到 {len(models)} 个模型", "models": models}


def test_mineru_connection(
    token: str,
    model_version: str = DEFAULT_MINERU_MODEL_VERSION,
    base_url: str = DEFAULT_MINERU_BASE_URL,
) -> dict:
    """Test MinerU API connection."""
    import urllib.error
    import urllib.request

    if not token:
        return {"success": False, "message": "MinerU API Token 未配置"}

    api_base = (base_url or DEFAULT_MINERU_BASE_URL).rstrip("/")
    body = {"files": [{"name": "test.pdf"}], "model_version": model_version}
    request = urllib.request.Request(
        f"{api_base}/api/v4/file-urls/batch",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data.get("code") == 0:
                return {"success": True, "message": "MinerU API 连接成功！"}
            return {"success": False, "message": f"MinerU API 返回错误: {data.get('msg', '未知错误')}"}
    except urllib.error.HTTPError as exc:
        error_body = ""
        try:
            error_body = exc.read().decode("utf-8")
        except Exception:
            pass
        if exc.code == 401:
            return {"success": False, "message": "认证失败：MinerU Token 无效"}
        if exc.code == 429:
            return {"success": False, "message": "请求过于频繁：MinerU API 限流"}
        return {"success": False, "message": f"HTTP {exc.code}: {error_body or exc.reason}"}
    except urllib.error.URLError as exc:
        return {"success": False, "message": f"网络错误: {exc.reason}"}
    except Exception as exc:
        return {"success": False, "message": f"测试失败: {str(exc)}"}
