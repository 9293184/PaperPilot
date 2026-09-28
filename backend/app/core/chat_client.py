"""问答用的流式客户端（薄封装）。

HTTP 细节（OpenAI 兼容 / Anthropic 两种协议）已统一到 ``app.core.llm_client``，
这里只保留问答特有的部分：上下文截断、重试、标题生成与错误类型。
"""

from __future__ import annotations

import logging
from typing import AsyncIterator

from app.core import llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)

# Maximum characters of paper context injected into the system prompt.
# For chat, we keep context smaller than analysis to leave room for the
# conversation history and the model's response.
MAX_CONTEXT_CHARS = 60000

# HTTP timeout for streaming connections (seconds).
STREAM_TIMEOUT = llm_client.STREAM_TIMEOUT

# 生成标题用的模型；None 表示沿用当前配置的模型（跨厂商时更安全）
TITLE_MODEL: str | None = None

# Non-streaming timeout for short completions (title generation).
SHORT_TIMEOUT = 30


class ChatClientError(RuntimeError):
    pass


def _truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate paper context to fit within model context window.

    For chat, we prioritize the head (intro + abstract) and tail
    (conclusion + references) to give the model the most useful context.
    """
    if not text or len(text) <= max_chars:
        return text or ""

    head_size = int(max_chars * 0.5)
    tail_size = max_chars - head_size
    head = text[:head_size]
    tail = text[-tail_size:]
    return (
        head
        + "\n\n[... 中间部分内容已省略以适配上下文长度 ...]\n\n"
        + tail
    )


async def stream_chat(
    messages: list[dict],
    temperature: float = 0.7,
    max_retries: int = 2,
    model: str | None = None,
) -> AsyncIterator[str]:
    """Stream a chat completion response.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        temperature: Sampling temperature (0.0-2.0).
        max_retries: Maximum number of retry attempts for transient errors.
        model: Override model name; None 表示沿用当前配置。

    Yields:
        String tokens/chunks from the model response.

    Raises:
        ChatClientError: If the API call fails after all retries.
    """
    if not settings.llm_api_key:
        raise ChatClientError("API key not configured")

    # 一旦已经向调用方 yield 过内容，就不能再重试：重试会从头再推一遍，
    # 客户端会看到重复的文本前缀。
    yielded_any = False
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            async for chunk in llm_client.stream_chat(
                messages, temperature=temperature, model=model
            ):
                yielded_any = True
                yield chunk
            return
        except llm_client.LLMError as exc:
            last_error = exc
            logger.warning(
                "stream_chat failed attempt=%d/%d error=%s",
                attempt + 1, max_retries, exc,
            )
            if attempt < max_retries and not yielded_any:
                await _async_sleep(1 + attempt)
                continue
            break

    raise ChatClientError(
        f"stream_chat failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


async def _async_sleep(seconds: float) -> None:
    """Simple async sleep helper."""
    import asyncio
    await asyncio.sleep(seconds)


def _fallback_title(message: str) -> str:
    """Generate a title by smart truncation of the first message.

    Used as a fallback when LLM-based title generation is unavailable.
    """
    text = message.strip().replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    # Strip common prefix patterns so the title captures the real intent.
    for prefix in (
        "请帮我分析这段内容：", "请帮我分析这段内容:", "请帮我分析：",
        "请帮我分析:", "请分析", "请总结", "请解释", "请介绍",
        "请列出", "请生成", "请详细", "帮我", "请",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip("：:、，, ").strip()
            break
    if not text:
        text = message.strip().replace("\n", " ")
    if len(text) > 20:
        text = text[:20] + "…"
    return text or "新对话"


async def generate_title(first_message: str) -> str:
    """Generate a concise title for a chat session from the first user message.

    Uses a non-streaming LLM call. Falls back to smart truncation on any error
    so the session always gets a title.

    Args:
        first_message: The first user message in the session.

    Returns:
        A short title string (typically ≤15 chars).
    """
    if not first_message or not first_message.strip():
        return "新对话"

    if not settings.llm_api_key:
        return _fallback_title(first_message)

    try:
        title = await _run_blocking(
            llm_client.chat_completion,
            [
                {
                    "role": "system",
                    "content": (
                        "你是对话标题生成器。根据用户的第一条消息，提炼出对话的核心主题，"
                        "生成一个简短的标题（不超过15个字，不要加引号、书名号或标点结尾）。"
                        "只输出标题文本本身。"
                    ),
                },
                {"role": "user", "content": first_message},
            ],
            temperature=0.3,
            max_tokens=64,
            model=TITLE_MODEL,
            timeout=SHORT_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - 标题失败不能影响问答
        logger.warning("generate_title failed, fallback to truncation: %s", exc)
        return _fallback_title(first_message)

    title = (title or "").strip().strip('"').strip("'").strip("《》").strip()
    return title or _fallback_title(first_message)


async def _run_blocking(func, *args, **kwargs):
    """在线程池里执行同步调用，避免阻塞事件循环。"""
    from fastapi.concurrency import run_in_threadpool

    return await run_in_threadpool(func, *args, **kwargs)
