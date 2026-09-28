"""统一的大模型客户端：同时支持 OpenAI 兼容协议与 Anthropic(Claude) 协议。

两种协议的差异全部在这里消化，业务代码只调用 ``chat_completion`` /
``stream_chat`` / ``list_models``，不必关心底层是哪家厂商、哪种格式。

- **openai**：``POST {base}/chat/completions``，``Authorization: Bearer``；
  取 ``choices[0].message.content``（流式为 ``choices[0].delta.content``）。
- **anthropic**：``POST {base}/v1/messages``，``x-api-key`` + ``anthropic-version``；
  ``system`` 必须提到顶层参数、``max_tokens`` 必填；
  取 ``content[0].text``（流式为 ``content_block_delta.delta.text``）。

默认读取 ``settings`` 里的已保存配置；也可通过 ``api_key`` / ``base_url`` /
``protocol`` 显式覆盖（「测试连接」「拉取模型列表」会用到未保存的配置）。
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterable

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

PROTOCOL_OPENAI = "openai"
PROTOCOL_ANTHROPIC = "anthropic"
SUPPORTED_PROTOCOLS = (PROTOCOL_OPENAI, PROTOCOL_ANTHROPIC)

# Anthropic 要求显式指定 anthropic-version 头
ANTHROPIC_VERSION = "2023-06-01"
# Anthropic 的 max_tokens 是必填项；分析类请求返回内容较长，给足空间
DEFAULT_ANTHROPIC_MAX_TOKENS = 8192

SYNC_TIMEOUT = 120
STREAM_TIMEOUT = 300
MODELS_TIMEOUT = 30


class LLMError(RuntimeError):
    """统一的大模型调用错误（协议差异已在此层抹平）。"""


def normalize_protocol(value: str | None) -> str:
    protocol = (value or "").strip().lower()
    return protocol if protocol in SUPPORTED_PROTOCOLS else PROTOCOL_OPENAI


class _Target:
    """一次调用所使用的一套配置。"""

    __slots__ = ("api_key", "base_url", "protocol")

    def __init__(self, api_key: str, base_url: str, protocol: str) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or "").strip().rstrip("/")
        self.protocol = normalize_protocol(protocol)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    @property
    def chat_url(self) -> str:
        if self.protocol == PROTOCOL_ANTHROPIC:
            return f"{self.base_url}/v1/messages"
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        if self.protocol == PROTOCOL_ANTHROPIC:
            return f"{self.base_url}/v1/models"
        return f"{self.base_url}/models"

    @property
    def headers(self) -> dict[str, str]:
        if self.protocol == PROTOCOL_ANTHROPIC:
            return {
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def _target(
    api_key: str | None = None,
    base_url: str | None = None,
    protocol: str | None = None,
) -> _Target:
    return _Target(
        api_key if api_key is not None else settings.llm_api_key,
        base_url if base_url is not None else settings.llm_base_url,
        protocol if protocol is not None else settings.llm_protocol,
    )


def _split_system(messages: Iterable[dict]) -> tuple[str, list[dict]]:
    """Anthropic 要求 system 提示词作为顶层参数，不能出现在 messages 里。"""
    system_parts: list[str] = []
    rest: list[dict] = []
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
        else:
            rest.append({
                "role": message.get("role", "user"),
                "content": message.get("content", ""),
            })
    return "\n\n".join(system_parts), rest


def _build_body(
    target: _Target,
    messages: list[dict],
    temperature: float | None,
    max_tokens: int | None,
    model: str | None,
    stream: bool,
    json_object: bool = False,
) -> dict[str, Any]:
    effective_model = model or settings.llm_model
    if target.protocol == PROTOCOL_ANTHROPIC:
        system, rest = _split_system(messages)
        body: dict[str, Any] = {
            "model": effective_model,
            "messages": rest,
            "max_tokens": max_tokens or DEFAULT_ANTHROPIC_MAX_TOKENS,
        }
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        # Anthropic 没有 response_format，JSON 输出靠提示词约束
    else:
        body = {"model": effective_model, "messages": list(messages)}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens
        if json_object:
            body["response_format"] = {"type": "json_object"}
    if stream:
        body["stream"] = True
    return body


def _extract_text(target: _Target, payload: dict) -> str:
    """从非流式响应里取出正文。"""
    if target.protocol == PROTOCOL_ANTHROPIC:
        blocks = payload.get("content") or []
        return "".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type", "text") == "text"
        )
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "")


def _extract_delta(target: _Target, payload: dict) -> str:
    """从流式分片里取出增量文本。"""
    if target.protocol == PROTOCOL_ANTHROPIC:
        # Anthropic 文本增量事件：
        # {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}
        if payload.get("type") != "content_block_delta":
            return ""
        delta = payload.get("delta") or {}
        if delta.get("type") not in (None, "text_delta"):
            return ""
        return str(delta.get("text") or "")
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("delta") or {}).get("content") or "")


def _http_error(exc: httpx.HTTPStatusError) -> LLMError:
    response = exc.response
    detail = response.text[:300] if response is not None else ""
    status = response.status_code if response is not None else "?"
    return LLMError(f"HTTP {status}: {detail}")


def chat_completion(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    model: str | None = None,
    timeout: int = SYNC_TIMEOUT,
    json_object: bool = False,
    api_key: str | None = None,
    base_url: str | None = None,
    protocol: str | None = None,
) -> str:
    """同步调用一次对话补全，返回文本内容（失败抛 ``LLMError``）。

    ``json_object=True`` 时，OpenAI 兼容协议会带上 ``response_format``；
    Anthropic 无此参数，自动忽略（JSON 输出由提示词约束）。
    """
    target = _target(api_key, base_url, protocol)
    if not target.is_configured:
        raise LLMError("API key not configured")
    body = _build_body(target, messages, temperature, max_tokens, model, stream=False, json_object=json_object)
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(target.chat_url, json=body, headers=target.headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise _http_error(exc) from exc
    except httpx.RequestError as exc:
        raise LLMError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise LLMError("响应不是合法的 JSON") from exc
    return _extract_text(target, payload).strip()


async def stream_chat(
    messages: list[dict],
    *,
    temperature: float | None = 0.7,
    max_tokens: int | None = None,
    model: str | None = None,
) -> AsyncIterator[str]:
    """异步流式调用，逐段产出文本（失败抛 ``LLMError``）。"""
    target = _target()
    if not target.is_configured:
        raise LLMError("API key not configured")
    body = _build_body(target, messages, temperature, max_tokens, model, stream=True)
    async with httpx.AsyncClient(timeout=STREAM_TIMEOUT) as client:
        try:
            async with client.stream("POST", target.chat_url, json=body, headers=target.headers) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    text = _extract_delta(target, payload)
                    if text:
                        yield text
        except httpx.HTTPStatusError as exc:
            raise _http_error(exc) from exc
        except httpx.RequestError as exc:
            raise LLMError(str(exc)) from exc


def list_models(
    *,
    timeout: int = MODELS_TIMEOUT,
    api_key: str | None = None,
    base_url: str | None = None,
    protocol: str | None = None,
) -> list[str]:
    """拉取当前厂商可用的模型列表（失败抛 ``LLMError``）。"""
    target = _target(api_key, base_url, protocol)
    if not target.is_configured:
        raise LLMError("API key not configured")
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(target.models_url, headers=target.headers)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise _http_error(exc) from exc
    except httpx.RequestError as exc:
        raise LLMError(str(exc)) from exc
    except json.JSONDecodeError as exc:
        raise LLMError("响应不是合法的 JSON") from exc

    items = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise LLMError("响应中没有模型列表（data 字段）")
    models: list[str] = []
    for item in items:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
            if model_id:
                models.append(str(model_id))
        elif isinstance(item, str):
            models.append(item)
    return sorted(set(models))
