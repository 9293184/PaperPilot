"""八维分析所用的大模型调用。

协议差异（OpenAI 兼容 / Anthropic）统一由 ``app.core.llm_client`` 处理，
这里只保留提示词构造、重试与结果归一化。
"""

from __future__ import annotations

import json
import logging
import re
import time

from app.core import llm_client
from app.core.config import settings

logger = logging.getLogger(__name__)


class DeepseekError(RuntimeError):
    pass


# Maximum characters of paper text sent to the model in a single request.
# DeepSeek context window is ~64K tokens; 80K chars is a safe upper bound
# for mixed Chinese/English content (roughly 40K-50K tokens).
MAX_FULL_TEXT_CHARS = 80000

# Number of retry attempts when JSON parsing or transient network errors occur.
MAX_RETRIES = 2

# HTTP timeout for DeepSeek API calls (seconds). Large papers/surveys may take
# longer to generate the eight-dimension analysis.
API_TIMEOUT = 300


def _sanitize_json_content(content: str) -> str:
    """Remove/escape control characters that break json.loads.

    DeepSeek occasionally emits raw control characters (e.g. vertical tab
    ``\\x0b``, form feed ``\\x0c``, NUL ``\\x00``) inside JSON string values
    when generating long analyses for large papers.  Even with
    ``json.loads(..., strict=False)`` some of these characters cause
    ``Invalid control character`` errors.

    This function:
    1. Strips a BOM if present.
    2. Removes other illegal control characters (0x00-0x1F) except
       ``\\t``, ``\\n``, ``\\r`` which are valid inside JSON strings when
       ``strict=False``.
    3. Strips Markdown code fences if the model wraps output in them.
    """
    if not content:
        return content

    # Strip BOM
    if content.startswith("\ufeff"):
        content = content[1:]

    # Strip Markdown code fences (```json ... ```)
    content = content.strip()
    if content.startswith("```"):
        # Remove opening fence (```json or ```)
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        # Remove closing fence
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]

    # Remove control characters except \t (0x09), \n (0x0A), \r (0x0D)
    content = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", content)

    return content


def _parse_json_safely(content: str) -> dict:
    """Parse JSON from model output with progressive fallback strategies.

    Args:
        content: Raw model output string.

    Returns:
        Parsed dict.

    Raises:
        json.JSONDecodeError: If all parsing strategies fail.
    """
    # Strategy 1: direct parse with strict=False (allows \t\n\r in strings)
    try:
        return json.loads(content, strict=False)
    except json.JSONDecodeError:
        pass

    # Strategy 2: sanitize control characters then parse
    sanitized = _sanitize_json_content(content)
    try:
        return json.loads(sanitized, strict=False)
    except json.JSONDecodeError:
        pass

    # Strategy 3: try to extract the first {...} block (in case model added
    # extra text before/after the JSON)
    match = re.search(r"\{.*\}", sanitized, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            # Strategy 4: last resort - sanitize the extracted block too
            cleaned = _sanitize_json_content(match.group(0))
            return json.loads(cleaned, strict=False)

    # If all strategies fail, raise the original error by re-parsing
    return json.loads(content, strict=False)


def _truncate_for_context(text: str, max_chars: int = MAX_FULL_TEXT_CHARS) -> str:
    """Truncate text to fit within model context window.

    For large papers (graduation theses, surveys), the full Markdown may
    exceed the model's context window.  We keep the head (title, abstract,
    introduction) and tail (conclusion, references) which are most
    informative for eight-dimension analysis, dropping the middle.
    """
    if not text or len(text) <= max_chars:
        return text or ""

    # Reserve 20% for the head (abstract + intro) and 20% for the tail
    # (conclusion + discussion), dropping 60% from the middle.
    head_size = int(max_chars * 0.4)
    tail_size = max_chars - head_size
    head = text[:head_size]
    tail = text[-tail_size:]
    return (
        head
        + "\n\n[... 中间部分内容已省略以适配模型上下文长度 ...]\n\n"
        + tail
    )


def _build_markdown_analysis_prompts(payload: dict[str, str]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for MinerU Markdown input.

    Markdown input is already clean structured text, so we:
    - Skip the OCR-noise warning
    - Guide the model to leverage Markdown headings for section location
    - Emphasize extracting details from specific Markdown sections
    - Require Markdown-formatted output (bold, lists, paragraphs) for rich display
    """
    system_prompt = (
        '你是一名严谨的论文阅读与综述助手。你的任务不是简单复述摘要，而是基于给定论文内容做深入理解、结构化提炼与批判性分析。'
        '输入是经 MinerU 解析的 Markdown 文本，结构清晰：标题通常以 # 开头，各章节以 ## / ### 标识，'
        '你可以利用这些 Markdown 结构快速定位摘要、引言、方法、实验、结论等部分。'
        '请先在内部完成：1) 判断论文研究对象、任务类型与问题边界；2) 理清核心方法、关键模块、训练/推理流程或理论推导；'
        '3) 识别实验设置、对比基线、指标、消融与结论是否充分；4) 总结贡献、适用场景、局限性与可能风险。'
        '最终只输出严格合法的 JSON，不要输出任何额外解释、代码块或前后缀文字。'
        'JSON 必须且只能包含以下键：tldr, motivation, methodology, experiments, resources, ablation, conclusion, strengths, weaknesses。'
        '\n\n**输出格式要求（Markdown 结构 + LaTeX 数学公式）：**\n'
        '除 tldr 外，所有分析字段（motivation, methodology, experiments, resources, ablation, conclusion, strengths, weaknesses）必须使用 Markdown 格式输出，具体规则：\n'
        '1. **加粗关键信息**：使用 **文字** 加粗标注核心概念、关键指标、重要结论等。\n'
        '2. **有序列表分点**：当内容包含多个要点时，使用 1. 2. 3. 有序列表逐条说明。\n'
        '3. **无序列表列举**：当内容为并列要点时，使用 - 无序列表列举。\n'
        '4. **段落分隔**：不同主题使用空行分隔，每段聚焦一个子主题。\n'
        '5. **每个字段至少包含 2-3 个段落或要点**，避免单段落大段文字。\n'
        '6. **数学公式格式**：所有数学符号、变量、公式必须用 LaTeX 语法包裹：\n'
        '   - 行内公式用 $...$ 包裹，如 $\\alpha=0.6$、$x^2$、$\\beta_{ij}$、$\\frac{a}{b}$\n'
        '   - 块级公式用 $$...$$ 包裹，如 $$\\sum_{i=1}^{n} x_i$$\n'
        '   - 希腊字母用 LaTeX 命令：$\\alpha$ $\\beta$ $\\gamma$ $\\delta$ $\\epsilon$ $\\theta$ $\\lambda$ $\\sigma$ $\\omega$ 等\n'
        '   - 数学运算符用 LaTeX 命令：$\\leq$ $\\geq$ $\\neq$ $\\approx$ $\\times$ $\\div$ $\\pm$ $\\sum$ $\\int$ $\\infty$ 等\n'
        '   - 上下标用 ^ 和 _：$x^2$、$a_{ij}$、$d_{model}$、$h_t$、$W^{(l)}$\n'
        '   - 分数用 \\frac：$\\frac{分子}{分母}$\n'
        '   - 复杂度记号用 $...$ 包裹：$O(n^2)$、$O(n \\log n)$、$O(n^2 \\cdot d)$\n'
        '   - 模型维度、层数等符号用 $...$ 包裹：$d_{model}=512$、$n_{layers}=12$\n'
        '   - 损失函数、激活函数等用 $...$ 包裹：$L=\\frac{1}{N}\\sum_{i=1}^{N} \\ell(x_i,y_i)$、$\\sigma(x)=\\frac{1}{1+e^{-x}}$\n'
        '\n示例格式（JSON 字符串中的 \\n 表示换行）：\n'
        'motivation 示例："**研究背景**\\n当前深度学习在XX领域面临XX挑战...\\n\\n**研究动机**\\n1. 现有方法存在XX问题\\n2. 实际应用需求XX"\n'
        'methodology 示例："**核心框架**\\n本文提出XX架构，包含以下关键模块：\\n1. **模块A**：负责XX功能\\n2. **模块B**：实现XX机制\\n\\n**技术细节**\\n- 采用XX方法优化XX，设置 $\\alpha=0.6$ 作为超参数\\n- 损失函数为 $L = \\frac{1}{N}\\sum_{i=1}^{N} \\ell(x_i, y_i)$"\n'
        'tldr 字段为论文的「一句话精炼摘要」，要求控制在 200 个字符以内，必须让读者一眼看清论文的问题背景、核心方法与主要结论；'
        '不要罗列细节，不要使用「本文」「作者」等空洞主语，直接以「针对……问题，提出……方法，实验表明……」的紧凑句式输出，连贯成一段。'
        '如果某部分信息在输入中缺失，请根据已有内容做合理推断，并明确标注「文中未明确说明」或「无法从当前文本确认」。'
        '重要：JSON 字符串值中不得包含未转义的控制字符（如垂直制表符、换页符），换行请使用 \\n 转义序列。'
    )
    full_text = _truncate_for_context(payload.get("full_text", ""))
    user_prompt = (
        '请基于以下 Markdown 论文文本进行分析。文本已经过 MinerU 结构化解析，章节清晰。\n'
        '**特别指令：请先从 Markdown 中完整提取 Abstract（摘要）的原始英文内容，放在 abstract_original 字段中。**\n'
        '你可以通过查找 ## Abstract 或 ## 摘要 标题快速定位摘要部分。\n'
        '你的任务分两步：\n'
        '1) 先识别并整理基础信息：title、title_cn、title_en、authors、source（期刊名或会议名）。\n'
        '   - 标题通常位于开头的 # 一级标题中。\n'
        '   - 作者通常紧跟标题下方，可能带有单位信息，请只保留作者姓名。\n'
        '2) 再完成 TLDR 与八维分析：tldr、motivation、methodology、experiments、resources、ablation、conclusion、strengths、weaknesses。\n'
        '   - tldr：先用一句话概括全文（≤200 字），让读者瞬间理解论文的问题背景、主要方法与结论。'
        '     禁止使用「本文提出」「作者认为」等空泛句式；建议采用「针对……问题，提出……方法，实验表明……」的紧凑结构。\n'
        '   - motivation：从 ## Introduction 或 ## 1. Introduction 等章节提取研究动机，使用 Markdown 格式分点阐述。\n'
        '   - methodology：从 ## Method / ## Methodology / ## 方法 等章节提取核心方法，使用 Markdown 格式分模块说明。\n'
        '   - experiments：从 ## Experiments / ## 实验 等章节提取实验设置，使用 Markdown 格式分要点列出。\n'
        '   - ablation：从 ## Ablation / ## 消融实验 等章节提取消融研究，使用 Markdown 格式分点展示。\n'
        '   - conclusion：从 ## Conclusion / ## 结论 等章节提取结论，使用 Markdown 格式分段总结。\n'
        '\n**Markdown 格式要求（除 tldr 外的所有分析字段）：**\n'
        '- 使用 **加粗** 标注核心概念、关键指标和重要结论\n'
        '- 使用有序列表（1. 2. 3. ）组织多个要点\n'
        '- 使用无序列表（- ）列举并列信息\n'
        '- 不同主题使用空行分隔段落\n'
        '- 每个字段至少包含 2-3 个结构化段落或要点\n'
        '- **数学公式必须用 LaTeX 包裹**：行内公式用 $...$，块级公式用 $$...$$，如 $\\alpha=0.6$、$\\beta$、$\\frac{a}{b}$、$d_{model}$、$O(n^2)$\n'
        '\n要求：\n'
        '- 如果原文中存在中英文双标题，请分别识别；如果只存在一个标题，也请如实输出。\n'
        '- 如果作者、单位、期刊、会议、卷期页码、年份等信息在原文中能找到，请尽量提取；无法确认则明确写「文中未明确说明」。\n'
        '- 所有输出必须严格为 JSON，不要输出任何额外解释、代码块或前后缀文字。\n'
        '- JSON 必须且只能包含以下键：title, title_cn, title_en, authors, source, tldr, motivation, methodology, experiments, resources, ablation, conclusion, strengths, weaknesses。\n'
        '- tldr 字段必须输出，长度严格控制在 200 字符以内。\n'
        '- 其余字段都用中文输出，使用 Markdown 格式，要求具体、信息密度高、尽量结合论文细节；不要空话套话。\n'
        '- 如果某部分信息在输入中缺失，请根据已有内容做合理推断，并明确标注「文中未明确说明」或「无法从当前文本确认」。\n\n'
        f'论文 Markdown 文本如下：\n{full_text}\n'
    )
    return system_prompt, user_prompt


def _build_ocr_analysis_prompts(payload: dict[str, str]) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for OCR input.

    Preserves the original prompt that warns about OCR noise.
    """
    system_prompt = (
        '你是一名严谨的论文阅读与综述助手。你的任务不是简单复述摘要，而是基于给定论文内容做深入理解、结构化提炼与批判性分析。'
        '请先在内部完成：1) 判断论文研究对象、任务类型与问题边界；2) 理清核心方法、关键模块、训练/推理流程或理论推导；'
        '3) 识别实验设置、对比基线、指标、消融与结论是否充分；4) 总结贡献、适用场景、局限性与可能风险。'
        '最终只输出严格合法的 JSON，不要输出任何额外解释、代码块或前后缀文字。'
        'JSON 必须且只能包含以下键：tldr, motivation, methodology, experiments, resources, ablation, conclusion, strengths, weaknesses。'
        '\n\n**输出格式要求（Markdown 结构 + LaTeX 数学公式）：**\n'
        '除 tldr 外，所有分析字段（motivation, methodology, experiments, resources, ablation, conclusion, strengths, weaknesses）必须使用 Markdown 格式输出，具体规则：\n'
        '1. **加粗关键信息**：使用 **文字** 加粗标注核心概念、关键指标、重要结论等。\n'
        '2. **有序列表分点**：当内容包含多个要点时，使用 1. 2. 3. 有序列表逐条说明。\n'
        '3. **无序列表列举**：当内容为并列要点时，使用 - 无序列表列举。\n'
        '4. **段落分隔**：不同主题使用空行分隔，每段聚焦一个子主题。\n'
        '5. **每个字段至少包含 2-3 个段落或要点**，避免单段落大段文字。\n'
        '6. **数学公式格式**：所有数学符号、变量、公式必须用 LaTeX 语法包裹：\n'
        '   - 行内公式用 $...$ 包裹，如 $\\alpha=0.6$、$x^2$、$\\beta_{ij}$、$\\frac{a}{b}$\n'
        '   - 块级公式用 $$...$$ 包裹，如 $$\\sum_{i=1}^{n} x_i$$\n'
        '   - 希腊字母用 LaTeX 命令：$\\alpha$ $\\beta$ $\\gamma$ $\\delta$ $\\epsilon$ $\\theta$ $\\lambda$ $\\sigma$ $\\omega$ 等\n'
        '   - 数学运算符用 LaTeX 命令：$\\leq$ $\\geq$ $\\neq$ $\\approx$ $\\times$ $\\div$ $\\pm$ $\\sum$ $\\int$ $\\infty$ 等\n'
        '   - 上下标用 ^ 和 _：$x^2$、$a_{ij}$、$d_{model}$、$h_t$、$W^{(l)}$\n'
        '   - 分数用 \\frac：$\\frac{分子}{分母}$\n'
        '   - 复杂度记号用 $...$ 包裹：$O(n^2)$、$O(n \\log n)$、$O(n^2 \\cdot d)$\n'
        '   - 模型维度、层数等符号用 $...$ 包裹：$d_{model}=512$、$n_{layers}=12$\n'
        '   - 损失函数、激活函数等用 $...$ 包裹：$L=\\frac{1}{N}\\sum_{i=1}^{N} \\ell(x_i,y_i)$、$\\sigma(x)=\\frac{1}{1+e^{-x}}$\n'
        '\n示例格式（JSON 字符串中的 \\n 表示换行）：\n'
        'motivation 示例："**研究背景**\\n当前深度学习在XX领域面临XX挑战...\\n\\n**研究动机**\\n1. 现有方法存在XX问题\\n2. 实际应用需求XX"\n'
        'methodology 示例："**核心框架**\\n本文提出XX架构，包含以下关键模块：\\n1. **模块A**：负责XX功能\\n2. **模块B**：实现XX机制\\n\\n**技术细节**\\n- 采用XX方法优化XX，设置 $\\alpha=0.6$ 作为超参数\\n- 损失函数为 $L = \\frac{1}{N}\\sum_{i=1}^{N} \\ell(x_i, y_i)$"\n'
        'tldr 字段为论文的「一句话精炼摘要」，要求控制在 200 个字符以内，必须让读者一眼看清论文的问题背景、核心方法与主要结论；'
        '不要罗列细节，不要使用「本文」「作者」等空洞主语，直接以「针对……问题，提出……方法，实验表明……」的紧凑句式输出，连贯成一段。'
        '如果某部分信息在输入中缺失，请根据已有内容做合理推断，并明确标注「文中未明确说明」或「无法从当前文本确认」。'
        '重要：JSON 字符串值中不得包含未转义的控制字符（如垂直制表符、换页符），换行请使用 \\n 转义序列。'
    )
    full_text = _truncate_for_context(payload.get("full_text", ""))
    user_prompt = (
        '请基于以下论文原文进行分析。原文已经经过 OCR/文本提取预处理，但可能仍存在噪声。\n'
        '**特别指令：请先从原文中完整提取 Abstract（摘要）的原始英文内容，放在 abstract_original 字段中。**\n'
        '如果摘要明显被截断（如以不完整的句子结尾），请标注「摘要可能不完整」。\n'
        '你的任务分两步：\n'
        '1) 先识别并整理基础信息：title、title_cn、title_en、authors、source（期刊名或会议名）。\n'
        '2) 再完成 TLDR 与八维分析：tldr、motivation、methodology、experiments、resources、ablation、conclusion、strengths、weaknesses。\n'
        '   - tldr：先用一句话概括全文（≤200 字），让读者瞬间理解论文的问题背景、主要方法与结论。'
        '     禁止使用「本文提出」「作者认为」等空泛句式；建议采用「针对……问题，提出……方法，实验表明……」的紧凑结构。\n'
        '   - motivation：提取研究动机，使用 Markdown 格式分点阐述。\n'
        '   - methodology：提取核心方法，使用 Markdown 格式分模块说明。\n'
        '   - experiments：提取实验设置，使用 Markdown 格式分要点列出。\n'
        '   - ablation：提取消融研究，使用 Markdown 格式分点展示。\n'
        '   - conclusion：提取结论，使用 Markdown 格式分段总结。\n'
        '\n**Markdown 格式要求（除 tldr 外的所有分析字段）：**\n'
        '- 使用 **加粗** 标注核心概念、关键指标和重要结论\n'
        '- 使用有序列表（1. 2. 3. ）组织多个要点\n'
        '- 使用无序列表（- ）列举并列信息\n'
        '- 不同主题使用空行分隔段落\n'
        '- 每个字段至少包含 2-3 个结构化段落或要点\n'
        '- **数学公式必须用 LaTeX 包裹**：行内公式用 $...$，块级公式用 $$...$$，如 $\\alpha=0.6$、$\\beta$、$\\frac{a}{b}$、$d_{model}$、$O(n^2)$\n'
        '\n要求：\n'
        '- 如果原文中存在中英文双标题，请分别识别；如果只存在一个标题，也请如实输出。\n'
        '- 如果作者、单位、期刊、会议、卷期页码、年份等信息在原文中能找到，请尽量提取；无法确认则明确写「文中未明确说明」。\n'
        '- 所有输出必须严格为 JSON，不要输出任何额外解释、代码块或前后缀文字。\n'
        '- JSON 必须且只能包含以下键：title, title_cn, title_en, authors, source, tldr, motivation, methodology, experiments, resources, ablation, conclusion, strengths, weaknesses。\n'
        '- tldr 字段必须输出，长度严格控制在 200 字符以内。\n'
        '- 其余字段都用中文输出，使用 Markdown 格式，要求具体、信息密度高、尽量结合论文细节；不要空话套话。\n'
        '- 如果某部分信息在输入中缺失，请根据已有内容做合理推断，并明确标注「文中未明确说明」或「无法从当前文本确认」。\n\n'
        f'论文原文如下：\n{full_text}\n'
    )
    return system_prompt, user_prompt


def analyze_text(payload: dict[str, str]) -> dict[str, str]:
    if not settings.llm_api_key:
        raise DeepseekError('API key is not configured')

    extraction_method = payload.get('extraction_method', '')
    is_markdown_input = extraction_method == 'mineru'

    if is_markdown_input:
        system_prompt, user_prompt = _build_markdown_analysis_prompts(payload)
    else:
        system_prompt, user_prompt = _build_ocr_analysis_prompts(payload)

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            # 统一客户端负责协议差异（OpenAI 兼容 / Anthropic）
            content = llm_client.chat_completion(
                messages,
                temperature=0.2,
                json_object=True,
                timeout=API_TIMEOUT,
            )
            parsed = _parse_json_safely(content)

            def _normalize_value(value):
                if value is None:
                    return ''
                if isinstance(value, list):
                    return '；'.join(str(item).strip() for item in value if item and str(item).strip())
                if isinstance(value, str):
                    return value.strip()
                return str(value).strip()

            result = {key: _normalize_value(parsed.get(key)) for key in [
                'title', 'title_cn', 'title_en', 'authors', 'source',
                'tldr', 'motivation', 'methodology', 'experiments', 'resources',
                'ablation', 'conclusion', 'strengths', 'weaknesses',
            ]}
            # Normalize TLDR: collapse newlines/excessive whitespace into
            # single spaces (preserves readability of mixed CN-EN text like
            # "Transformer 架构"), then enforce the 200-character constraint.
            # Truncating at the last 完整 sentence boundary within the limit
            # keeps the summary readable while still respecting the hard cap.
            tldr_value = result.get('tldr', '')
            if tldr_value:
                tldr_value = re.sub(r'\s+', ' ', tldr_value).strip()
                if len(tldr_value) > 200:
                    truncated = tldr_value[:200]
                    last_break = max(
                        truncated.rfind('。'),
                        truncated.rfind('！'),
                        truncated.rfind('？'),
                        truncated.rfind('；'),
                    )
                    if last_break > 60:
                        result['tldr'] = truncated[:last_break + 1]
                    else:
                        result['tldr'] = truncated
                else:
                    result['tldr'] = tldr_value
            return result
        except llm_client.LLMError as exc:
            last_error = exc
            logger.warning(
                "analyze_text request_failed attempt=%d/%d error=%s",
                attempt + 1, MAX_RETRIES + 1, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(2 + attempt * 2)  # longer backoff for network errors
                continue
            raise DeepseekError(str(exc)) from exc
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "analyze_text json_parse_failed attempt=%d/%d error=%s",
                attempt + 1, MAX_RETRIES + 1, exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(1 + attempt)  # brief backoff
                continue

    # All retries exhausted
    raise DeepseekError(
        f"analyze_text failed after {MAX_RETRIES + 1} attempts: {last_error}"
    ) from last_error
