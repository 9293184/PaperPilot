"""Metadata extraction client for paper front matter."""

from __future__ import annotations

import ast
import json
import re

from app.core import llm_client
from app.core.config import settings
from app.core.debug_log import append_debug_record
from app.core.deepseek_client import _parse_json_safely, _sanitize_json_content


class MetadataError(RuntimeError):
    pass


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    temperature: float = 0.0,
    timeout: int = 60,
    json_object: bool = False,
) -> str:
    """调用大模型（协议差异由 llm_client 处理：OpenAI 兼容 / Anthropic）。"""
    return llm_client.chat_completion(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=timeout,
        json_object=json_object,
    )


def extract_abstract_only(abstract_region: str) -> str:
    if not settings.llm_api_key:
        raise MetadataError("API key is not configured")

    if not abstract_region:
        return ""

    system_prompt = "你是一个论文摘要提取助手。你的任务是从给定文本中提取摘要原文。不要总结，不要改写，不要猜测。如果文本中包含摘要，直接返回摘要内容；否则返回空字符串。"

    user_prompt = f"请从以下文本中提取论文摘要原文：\n\n{abstract_region}"

    try:
        content = _call_llm(system_prompt, user_prompt, temperature=0.0, timeout=60)
    except llm_client.LLMError as exc:
        raise MetadataError(str(exc)) from exc

    content = content.strip()
    
    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()
    
    if content.lower().startswith("摘要：") or content.lower().startswith("abstract:"):
        content = content.split("：", 1)[1].strip() if "：" in content else content.split(":", 1)[1].strip()
    
    bad_patterns = [r"^The abstract is:\s*", r"^Abstract:\s*", r"^摘要：\s*", r"^摘要:\s*"]
    for pattern in bad_patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)

    content = re.sub(r"\s+", " ", content).strip()

    # Strip citation markers and stray HTML tags (see _clean_abstract_citations)
    content = _clean_abstract_citations(content)

    return content


def _cjk_ratio(text: str) -> float:
    """Return ratio of CJK characters to total alphanumeric characters."""
    if not text:
        return 0.0
    cjk_count = 0
    total_count = 0
    for ch in text:
        if ch.isspace() or not ch.isalnum():
            continue
        total_count += 1
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
        ):
            cjk_count += 1
    if total_count == 0:
        return 0.0
    return cjk_count / total_count


def _detect_language(text: str) -> str:
    """Detect whether text is predominantly Chinese ('zh') or English ('en').

    Returns empty string if text is empty.
    """
    if not text or not text.strip():
        return ""
    ratio = _cjk_ratio(text)
    return "zh" if ratio >= 0.3 else "en"


def detect_paper_language(parsed: dict[str, str]) -> str:
    """Detect the primary language of a paper from its full raw text.

    This is the authoritative function for determining whether a paper is
    Chinese or English. It must be used instead of ``_detect_language`` on
    ``abstract_region``, because Chinese academic papers commonly contain
    both Chinese and English abstracts. When ``_locate_abstract_region``
    captures the English abstract (which happens frequently because English
    markers are checked first), ``_detect_language(abstract_region)`` returns
    ``"en"``, causing a Chinese paper to be misclassified as English and its
    English title to be displayed as the primary title.

    To avoid this, we detect the paper's language from the full ``raw_text``
    (or a large sample of it). The body text of a Chinese paper is
    predominantly Chinese, so even with an English abstract and English figure
    captions, the CJK ratio stays well above the threshold. Conversely, the
    body text of an English paper is predominantly English.

    Args:
        parsed: The parsed text dict from ``extract_pdf_text`` /
            ``extract_text_from_markdown``. Must contain ``raw_text`` or
            ``full_text``. Falls back to ``abstract_region`` if neither is
            available.

    Returns:
        ``'zh'`` if the paper is predominantly Chinese, ``'en'`` if English,
        ``''`` if the language cannot be determined.
    """
    # Prefer raw_text (full document body), then full_text, then abstract_region
    raw_text = parsed.get("raw_text", "") or parsed.get("full_text", "") or ""
    if not raw_text:
        raw_text = parsed.get("abstract_region", "")

    if not raw_text or not raw_text.strip():
        return ""

    # Sample the first 10000 characters — enough to cover the title page,
    # abstract, and a portion of the introduction/body. This gives a
    # representative sample of the paper's actual language distribution.
    sample = raw_text[:10000]
    return _detect_language(sample)


def _translate_to_chinese(text: str, paper_id: str = "unknown") -> str:
    """Translate English text to Chinese via DeepSeek with language validation.

    Returns empty string if translation fails or result is not Chinese.
    """
    if not text or not text.strip():
        return ""
    if not settings.llm_api_key:
        return ""

    system_prompt = (
        "你是一名专业的学术翻译。请将给定的英文摘要翻译为流畅、准确的中文。"
        "要求：保持学术严谨性，专业术语翻译准确，语句通顺自然。"
        "只输出翻译后的中文内容，不要输出任何解释、前缀、后缀或原文。"
        "不要输出「翻译：」「中文翻译：」等任何标签。"
    )
    user_prompt = f"请将以下英文摘要翻译为中文：\n\n{text}"

    append_debug_record(
        paper_id,
        "translate_abstract_request",
        text_len=len(text),
    )
    try:
        content = _call_llm(system_prompt, user_prompt, temperature=0.0, timeout=60).strip()
    except llm_client.LLMError as exc:
        append_debug_record(paper_id, "translate_abstract_failed", error=str(exc))
        return ""

    if content.startswith('"') and content.endswith('"'):
        content = content[1:-1].strip()

    bad_patterns = [r"^翻译：\s*", r"^中文翻译：\s*", r"^摘要：\s*"]
    for pattern in bad_patterns:
        content = re.sub(pattern, "", content, flags=re.IGNORECASE)

    content = re.sub(r"\s+", " ", content).strip()

    # Validate: translation must actually be Chinese
    if _detect_language(content) != "zh":
        append_debug_record(
            paper_id,
            "translate_abstract_invalid",
            original_preview=text[:200],
            translation_preview=content[:200],
        )
        return ""

    append_debug_record(
        paper_id,
        "translate_abstract_done",
        translation_len=len(content),
    )
    return content


def _normalize_list_text(value: object) -> str:
    if isinstance(value, list | tuple | set):
        items = [str(item).strip() for item in value if str(item).strip()]
        return "；".join(items)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                items = [str(item).strip() for item in parsed if str(item).strip()]
                return "；".join(items)
        return text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    return str(value).strip()


# Patterns that should be stripped from abstract text:
#   1. <sup>NUM</sup> superscript citation markers (with content) — including
#      malformed variants like <sup>1<sup> (unclosed) produced by MinerU.
#      Only matches when the inner content is purely numeric / citation-like
#      (digits, commas, hyphens, spaces) so legitimate superscripts such as
#      <sup>th</sup> or <sup>n</sup> are preserved.
#   2. Bare numeric citation brackets: [1], [1,2], [1, 2, 3], [1-5].
#   3. Stray HTML inline tags (<sup>, <sub>, <i>, <b> ...) — only the tags
#      themselves are removed; inner content is kept. This runs AFTER the
#      citation-specific patterns so non-citation superscripts survive.
_SUP_CITATION_RE = re.compile(
    r"<sup\b[^>]*>\s*\d+(?:\s*[-,]\s*\d+)*\s*(?:</sup>|<sup\b[^>]*>|(?=\s|$))",
    re.IGNORECASE,
)
_CITATION_BRACKET_RE = re.compile(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]")
_HTML_TAG_RE = re.compile(r"</?(?:sup|sub|i|b|em|strong|span|a)\b[^>]*>", re.IGNORECASE)


def _clean_abstract_citations(text: str) -> str:
    """Remove citation markers and stray HTML tags from an abstract string.

    Targets:
      - <sup>1</sup>, <sup>1<sup> (malformed) and similar numeric superscripts
      - Bare numeric citation brackets: [1], [1,2], [1, 2, 3], [1-5]
      - Leftover HTML inline tags (tags only, content preserved)
      - Collapsed whitespace after removals
    """
    if not text:
        return text
    # 1. Remove numeric superscript citations entirely (tag + content)
    cleaned = _SUP_CITATION_RE.sub("", text)
    # 2. Remove bare numeric citation brackets
    cleaned = _CITATION_BRACKET_RE.sub("", cleaned)
    # 3. Strip any remaining HTML inline tags, keeping their inner text
    cleaned = _HTML_TAG_RE.sub("", cleaned)
    # Collapse whitespace introduced by removals (keep single spaces)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# =============================================================================
# Local regex-based metadata extraction (post-processing + fallback)
# =============================================================================

# Matches standard DOIs: 10.<registrant_code>/<suffix>
# Suffix allows alphanumerics, hyphens, underscores, dots, slashes, parens.
_DOI_RE = re.compile(
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    re.IGNORECASE,
)

# Matches 4-digit years in plausible ranges (1950-2099) for publication dates.
_YEAR_RE = re.compile(
    r"(?<!\d)(?:19[5-9]\d|20[0-9]\d)(?!\d)"
)

# Major CS conference / journal name aliases and full names (case-insensitive).
# Mapping: canonical_name -> list of regex patterns (matched in source text).
_VENUE_PATTERNS: list[tuple[str, list[str]]] = [
    ("CVPR", [
        r"\bCVPR\b",
        r"Computer\s+Vision\s+and\s+Pattern\s+Recognition",
        r"IEEE/CVF\s+Conference\s+on\s+Computer\s+Vision\s+and\s+Pattern\s+Recognition",
    ]),
    ("ICCV", [
        r"\bICCV\b",
        r"International\s+Conference\s+on\s+Computer\s+Vision",
    ]),
    ("ECCV", [
        r"\bECCV\b",
        r"European\s+Conference\s+on\s+Computer\s+Vision",
    ]),
    ("NeurIPS", [
        r"\bNeurIPS\b",
        r"\bNIPS\b",
        r"Neural\s+Information\s+Processing\s+Systems",
        r"Advances\s+in\s+Neural\s+Information\s+Processing\s+Systems",
    ]),
    ("ICML", [
        r"\bICML\b",
        r"International\s+Conference\s+on\s+Machine\s+Learning",
    ]),
    ("ICLR", [
        r"\bICLR\b",
        r"International\s+Conference\s+on\s+Learning\s+Representations",
    ]),
    ("ACL", [
        r"\bACL\b",
        r"Annual\s+Meeting\s+of\s+the\s+Association\s+for\s+Computational\s+Linguistics",
        r"Association\s+for\s+Computational\s+Linguistics",
    ]),
    ("EMNLP", [
        r"\bEMNLP\b",
        r"Conference\s+on\s+Empirical\s+Methods\s+in\s+Natural\s+Language\s+Processing",
    ]),
    ("NAACL", [
        r"\bNAACL\b",
        r"North\s+American\s+Chapter\s+of\s+the\s+Association\s+for\s+Computational\s+Linguistics",
    ]),
    ("AAAI", [
        r"\bAAAI\b",
        r"AAAI\s+Conference\s+on\s+Artificial\s+Intelligence",
        r"Association\s+for\s+the\s+Advancement\s+of\s+Artificial\s+Intelligence",
    ]),
    ("IJCAI", [
        r"\bIJCAI\b",
        r"International\s+Joint\s+Conference\s+on\s+Artificial\s+Intelligence",
    ]),
    ("KDD", [
        r"\bKDD\b",
        r"Knowledge\s+Discovery\s+and\s+Data\s+Mining",
        r"ACM\s+SIGKDD\s+Conference\s+on\s+Knowledge\s+Discovery\s+and\s+Data\s+Mining",
    ]),
    ("SIGGRAPH", [
        r"\bSIGGRAPH\b",
        r"Special\s+Interest\s+Group\s+on\s+Computer\s+Graphics\s+and\s+Interactive\s+Techniques",
    ]),
    ("TPAMI", [
        r"\bTPAMI\b",
        r"IEEE\s+Transactions\s+on\s+Pattern\s+Analysis\s+and\s+Machine\s+Intelligence",
    ]),
    ("IJCV", [
        r"\bIJCV\b",
        r"International\s+Journal\s+of\s+Computer\s+Vision",
    ]),
    ("JMLR", [
        r"\bJMLR\b",
        r"Journal\s+of\s+Machine\s+Learning\s+Research",
    ]),
    ("Nature", [
        r"\bNature\b",
    ]),
    ("Science", [
        r"\bScience\b",
    ]),
    ("ArXiv", [
        r"\barXiv\b",
        r"\bCoRR\b",
        r"arXiv:\d{4}\.\d{4,5}",
    ]),
]

# Chinese thesis / dissertation keywords
# Matches patterns like "XX大学", "XX学院", "硕士学位论文", "博士学位论文", "答辩日期", "指导教师" etc.
_UNIV_RE = re.compile(
    r"[\u4e00-\u9fa5]{2,20}大学|[\u4e00-\u9fa5]{2,20}学院|[\u4e00-\u9fa5]{2,20}研究院|中国科学院[\u4e00-\u9fa5]*研究所",
)
_DEGREE_TYPE_RE = re.compile(
    r"博士学位论文|硕士学位论文|学士学位论文|本科毕业论文|硕士论文|博士论文",
)
_ADVISOR_RE = re.compile(
    r"(?:指导教师|导师|指导老师|Supervisor|Advisor)\s*[:：]?\s*([\u4e00-\u9fa5A-Za-z·•・\s]{2,30})",
)
# Chinese date patterns: 2024年5月 or 2024年5月20日
_CN_DATE_RE = re.compile(
    r"(?:(?:19|20)\d{2})\s*年\s*(?:(?:0?[1-9]|1[0-2])\s*月)?(?:(?:0?[1-9]|[12]\d|3[01])\s*日)?",
)
# Defense date / submission date patterns
_DEFENSE_DATE_RE = re.compile(
    r"(?:答辩日期|提交日期|完成日期|定稿日期)\s*[:：]?\s*"
    r"((?:(?:19|20)\d{2}\s*年\s*(?:(?:0?[1-9]|1[0-2])\s*月)?(?:(?:0?[1-9]|[12]\d|3[01])\s*日)?))",
)
# Chinese department / major patterns
_DEPT_RE = re.compile(
    r"(?:学院|系|专业|院)\s*[:：]?\s*([\u4e00-\u9fa5A-Za-z·•・\s]{2,40}?)(?:\n|指导|导师|答辩|$)",
)


def _extract_doi(text: str) -> str:
    """Extract the first plausible DOI from raw text.

    Returns empty string if no DOI is found.
    """
    if not text:
        return ""
    m = _DOI_RE.search(text)
    if not m:
        return ""
    doi = m.group(0).rstrip(".,;:)]}")
    return doi


# Canonical venues that are *real publication venues* vs preprint servers.
# ArXiv is a preprint server and should never override a real conference/journal.
_REAL_PUBLICATION_VENUES = {v for v, _ in _VENUE_PATTERNS if v != "ArXiv"}


def _venue_is_preprint(canonical: str) -> bool:
    return canonical == "ArXiv"


def _find_closest_year_after(match_start: int, match_end: int, text: str,
                             lookahead: int = 400) -> str:
    """Find the first 4-digit year AFTER (or within 50 chars BEFORE) a match.

    Conference publication years almost always follow the venue mention on the
    same line or the next line. Preprint arXiv years typically PRECEDE the
    venue mention because arXiv metadata is in the header while the conference
    line is in the footer / copyright. Searching AFTER the match avoids
    accidentally pulling in arXiv preprint submission years that appear
    *earlier* in the document.
    """
    search_start = max(0, match_start - 50)
    search_end = min(len(text), match_end + lookahead)
    region = text[search_start:search_end]
    # FIRST pass: find first year that ISN'T an inline citation year
    first_raw: str | None = None
    for ym in _YEAR_RE.finditer(region):
        abs_start = search_start + ym.start()
        abs_end = search_start + ym.end()
        if _year_looks_like_inline_citation(abs_start, abs_end, text):
            continue
        return ym.group(0)
    return ""


def _year_looks_like_inline_citation(year_match_start: int, year_match_end: int,
                                     text: str) -> bool:
    """Return True if a 4-digit year sits in a pattern like '(Author et al. YYYY)'
    or '(Author1, Author2, YYYY)' — i.e. a citation to someone else's paper.

    These years MUST NOT be counted as "the paper's own publication year"
    because every paper's Introduction section is full of "(Foo et al. 2023)"
    inline citations.
    """
    # Look back 160 chars from the year for citation context.
    lookback_start = max(0, year_match_start - 160)
    prefix = text[lookback_start:year_match_start]
    # Right side context (up to 80 chars) also used for benchmark detection.
    tail_len_max = min(120, len(text) - year_match_end)
    tail_long = text[year_match_end:year_match_end + tail_len_max] if tail_len_max > 0 else ""
    tail = tail_long[:20]

    # --- Pattern A: explicit inline citation signal words in prefix ---
    # "(Zhao et al. 2023)"  /  "Ji et al. (2023)"  /  "(Lewis et al., 2020)"
    # "Rawte, Sheth, and Das 2023"  /  "Tishby, Pereira, and Bialek 2000"
    if re.search(
        r"(?:et\s*al\.?"                # et al / et al.
        r"|\b(?:and|&)\s+[A-Z]"          # ... and X (two+ authors)
        r"|[A-Z][a-z]+\s*,\s*[A-Z]"     # Surname, X (two-author list)
        r"|[A-Z][a-z]+\s*,\s*\d"        # Surname, YYYY (immediately)
        r"|\([A-Z][A-Za-z\-']+"         # "(Xxx surname open paren citation
        r"|[A-Z][a-z]+’[s]\s+"          # Turing's / Russell's (possessive author)
        r"|[A-Z][a-z]+'s\s+"            # ASCII apostrophe variant
        r")\s*[,.;:)]?\s*\(?\s*$",
        prefix,
    ):
        return True

    # --- Pattern B: citation by close-paren after year ---
    # "(Author, 2023)" → tail has ")" ; "Author (2023)" → tail has ")"
    # If we also have a plausible author-surname prefix, this is a citation.
    if (")" in tail or "," in tail or ";" in tail or "." in tail) and re.search(
        r"(?:et\s*al\.?|[A-Z][a-z]{2,}\s*\(?|[A-Z][a-z]+\s*,\s*[A-Z]|[A-Z][a-z]+’[s]|[A-Z][a-z]+'s)",
        prefix[-100:],
    ):
        return True

    # --- Pattern C: year preceded by a list of capitalized author surnames ---
    # Example: "Xu, Jain, and Kankanhalli (2024)"
    if re.search(r"[A-Z][a-z]+\s*(?:,|&|and)\s*[A-Z][a-z]+\s*\(?\s*$", prefix[-120:]):
        return True

    # --- Pattern D: benchmark / dataset / workshop / challenge / conference
    # acronym-then-year in body context (e.g. "WMT 2014 English-to-German
    # translation task", "ImageNet 2012 dataset", "CoNLL 2003 NER task",
    # "SST-2 (Socher et al. 2013)" wait — no, that last one is a citation we
    # already catch; but "GLUE benchmark (2019)" or "on COCO 2017 val split"
    # are definitely NOT the paper's publication year.
    #
    # Rules: if a year appears as "<ACRONYM/DATASET_NAME> <YYYY>" and the
    # following 80 chars contain task/dataset/benchmark/corpus/workshop/
    # split/collection/test/val/train/challenge keywords → skip it.
    _benchmark_suffix_pat = re.compile(
        r"(?:task|dataset|benchmark|corpus|workshop|split|train|val|test|"
        r"challenge|collection|track|leaderboard|shared\s*task|translation\s*task|"
        r"language\s*modeling|machine\s+translation|captioning|segmentation|"
        r"detection|recognition|classifier|regression)",
        re.IGNORECASE,
    )
    # Preceding acronym / dataset name in last 12-30 chars of prefix
    _prefix_acronym_or_dataset = re.search(
        r"(?:^|\s|[\"'(])\s*"
        r"(?:"
        r"[A-Z][A-Z0-9\-]{1,10}"          # e.g. WMT, COCO, SST-2, GLUE, SQuAD
        r"|ImageNet|WordNet|Freebase|WikiText|PennTreeBank|CIFAR|MNIST"
        r"|Kinetics|Charades|MSCOCO|Cityscapes|ADE20K|LSMDC|ActivityNet"
        r"|SNLI|MultiNLI|QQP|SST|MRPC|STS-B|RTE|WNLI|XSum|CNN/DailyMail"
        r"|CoNLL|SemEval|CLEF|TREC|RoboCup|ILSVRC"
        r"|English-to-German|English-to-French|German-to-English|French-to-English"
        r")\s*[-–]?\s*$",
        prefix[-40:],
    )
    if _prefix_acronym_or_dataset and _benchmark_suffix_pat.search(tail_long):
        return True
    # Looser fall-back: "<ACRONYM> <YEAR>" near "translation task / dataset / benchmark"
    if re.search(
        r"(?:WMT|IWSLT|BLEU|ROUGE|METEOR|BERTScore|SacreBLEU)\s*[-–]?\s*$",
        prefix[-30:],
    ):
        return True

    return False


def _strip_references_section(text: str) -> str:
    """Return text with everything after the References / Bibliography heading
    stripped. Citation years in References are NEVER this paper's own year.

    Also strips inline-author-citation heavy text that comes after an
    explicit heading marker.
    """
    m = re.search(
        r"\n\s*(?:##?\s*)?(?:References|Bibliography|References\s*and\s*Notes|引文?献?|参考资料)\s*\n",
        text,
        re.IGNORECASE,
    )
    if m:
        return text[:m.start()]
    return text


def _extract_year(text: str, prefer_defense: bool = False) -> str:
    """Extract a plausible publication year from raw text.

    Strategy:
      1. If prefer_defense and a Chinese defense/submission date is found, use
         its year (for Chinese theses where 答辩/完成 date == publication date).
      2. Find years NEAR / AFTER real conference/journal markers (NOT preprint
         markers like arXiv). The first year AFTER the venue mention (within
         400 chars) wins. arXiv submission years are ignored when a real
         venue is present.
      3. Check years near DOI (usually close to real publication metadata).
      4. Fall back to any year in the front matter.
    """
    if not text:
        return ""

    # Priority 1: defense date year (Chinese thesis)
    if prefer_defense:
        dm = _DEFENSE_DATE_RE.search(text)
        if dm:
            ym = _YEAR_RE.search(dm.group(1))
            if ym:
                return ym.group(0)

    # Strip References section BEFORE doing any year search — the references
    # section contains 100% citation-years for other papers.
    text_no_refs = _strip_references_section(text)
    front = text_no_refs[:5000]

    # =====================================================================
    # Priority 2: years NEAR / AFTER REAL (non-preprint) venue markers.
    # This avoids the classic bug: arXiv preprint year (2023) appearing
    # BEFORE an AAAI-26 acceptance line and getting picked up instead of
    # the actual 2026 publication year.
    # =====================================================================
    # Pre-compute DOI spans so we can skip venue matches that fall INSIDE a
    # DOI string (e.g. the TPAMI in "10.1109/TPAMI.2016.xxxxx" is the journal
    # code inside a DOI, not the real venue mention; "2016" there is part of
    # the DOI suffix, not the publication year).
    doi_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _DOI_RE.finditer(front)
    ]
    def _inside_doi(pos_start: int, pos_end: int) -> bool:
        return any(ds <= pos_start and de >= pos_end for ds, de in doi_spans)

    best_real_year: tuple[int, str] | None = None  # (priority_rank, year)
    _ambig_venue_lower = {v.lower() for v in _AMBIGUOUS_SOURCE_WORDS}
    for rank, (canonical, patterns) in enumerate(_VENUE_PATTERNS):
        if _venue_is_preprint(canonical):
            continue  # skip arXiv in this pass — handled separately later
        venue_is_ambig = canonical.lower() in _ambig_venue_lower
        venue_best_year: str | None = None
        for pat in patterns:
            for vm in re.finditer(pat, front, re.IGNORECASE):
                if _inside_doi(vm.start(), vm.end()):
                    continue  # this match is inside a DOI string — skip it!
                # Ambiguous-word guard: "Nature"/"Science"/"Cell" common-word
                # body matches do NOT count as venue matches for year lookup,
                # unless the match has DIRECT journal-style context on the
                # same/same+1 lines (same strict check as Pass 3 of the LLM
                # hallucination guard).
                if venue_is_ambig and not _ambiguous_word_has_own_publication_context(
                    front, vm.start(), vm.end(),
                ):
                    continue
                yr = _find_closest_year_after(vm.start(), vm.end(), front)
                if yr:
                    if venue_best_year is None:
                        venue_best_year = yr
        if venue_best_year:
            # Lower rank = better venue (CVPR/NeurIPS etc come first)
            if best_real_year is None or rank < best_real_year[0]:
                best_real_year = (rank, venue_best_year)
                if rank < 15:
                    return venue_best_year
    if best_real_year:
        return best_real_year[1]

    # Priority 3: year near DOI (DOIs are close to publication metadata,
    # not arXiv submission metadata). Use the after-match strategy here too
    # to avoid the DOI year being swallowed by a preceding arXiv stamp year.
    doi_m = _DOI_RE.search(front)
    if doi_m:
        yr = _find_closest_year_after(
            doi_m.start(), doi_m.end(), front, lookahead=500
        )
        if yr:
            return yr
        # Fallback: DOI vicinity search (both directions) — DOI is usually on
        # the copyright line where the year is nearby in either direction.
        start = max(0, doi_m.start() - 400)
        end = min(len(front), doi_m.end() + 400)
        near_doi = front[start:end]
        ym = _YEAR_RE.search(near_doi)
        if ym:
            return ym.group(0)

    # Priority 4: "Proceedings of / To appear in" patterns followed by a year
    proc_m = re.search(
        r"(?:Proceedings\s+of|To\s+appear\s+in|Published\s+in|Accepted\s+at)[^.\n]{0,180}?",
        front,
        re.IGNORECASE,
    )
    if proc_m:
        yr = _find_closest_year_after(proc_m.start(), proc_m.end(), front, lookahead=300)
        if yr:
            return yr

    # Priority 5: any year in first 3000 chars (front matter bias)
    # BUT with two strong guardrails (宁空勿错 — "better empty than wrong"):
    #   (a) skip inline-citation years AND benchmark/dataset-name years;
    #   (b) require nearby PUBLICATION context (©/Copyright/Vol/Issue/pp./
    #       ISBN/ISSN/Proceedings/Accepted/Published or Chinese thesis markers
    #       or defense/submission dates).
    # Years that don't satisfy both — e.g. "WMT 2014 translation task",
    # "ImageNet 2012 dataset", a benchmark name, or plain prose mentions
    # from the Introduction — are SILENTLY IGNORED rather than hallucinated
    # as the paper's own publication year.
    first_3k = front[:3000]
    _p5_pub_ctx = re.compile(
        r"(?:©|Copyright|Vol(?:ume)?\.?\s*\d|Issue\s+\d|pp\.\s*\d|"
        r"Published\s+(?:in|by|on)|Accepted\s+(?:at|for)|ISBN\s|ISSN\s|"
        r"博士学位论文|硕士学位论文|学士学位论文|"
        r"答辩日期|提交日期|完成日期|定稿日期|"
        r"Proceedings\s+of|Curran\s+Associates|IEEE\s+Computer\s+Society|"
        r"ACM\s+Press|Springer|Elsevier|USENIX|ACL\s+Anthology|"
        r"arXiv:\s*\d{4}\.\d{4,5})",
        re.IGNORECASE,
    )
    for ym in _YEAR_RE.finditer(first_3k):
        y_s, y_e = ym.start(), ym.end()
        if _year_looks_like_inline_citation(y_s, y_e, first_3k):
            continue
        w_s = max(0, y_s - 600)
        w_e = min(len(first_3k), y_e + 600)
        if _p5_pub_ctx.search(first_3k[w_s:w_e]):
            return ym.group(0)
        # Also allow: Chinese-form year "2024年" followed by month/day near
        # the paper's first page header (Chinese thesis date formats). This
        # is a narrower contextual guard than the generic body year case.
        y_txt = ym.group(0)
        rest_3k = first_3k[y_e:]
        if re.match(
            rf"\s*年\s*(?:(?:0?[1-9]|1[0-2])\s*月)?",
            rest_3k,
        ):
            return y_txt

    # "Last resort": we only reach this when there's no DOI, no real venue,
    # no Proceedings-of header, and the front matter contains ONLY inline-
    # citation years (Introduction section cites "(Foo 2023)" etc.). The
    # paper is a bare draft/preprint/working paper with no metadata.
    #
    # Historically we scanned the whole document here, but body text is full
    # of citation-years like "(Author et al. YYYY)" and historical references
    # ("Turing's 1936 proof") — neither is this paper's own publication year.
    # We therefore require "publication context" (copyright line / volume /
    # issue / page numbers) to accept any year in this final pass. Without
    # that context, return "" rather than hallucinate a wrong year.
    _pub_ctx_pat = re.compile(
        r"(?:©|Copyright|Vol(?:ume)?\.?\s*\d|Issue\s+\d|pp\.\s*\d|"
        r"Published\s+(?:in|by|on)|Accepted\s+(?:at|for)|ISBN\s|ISSN\s|"
        r"博士学位论文|硕士学位论文|学士学位论文)",
        re.IGNORECASE,
    )
    for ym in _YEAR_RE.finditer(text_no_refs):
        y_s, y_e = ym.start(), ym.end()
        if _year_looks_like_inline_citation(y_s, y_e, text_no_refs):
            continue
        window_s = max(0, y_s - 500)
        window_e = min(len(text_no_refs), y_e + 500)
        if _pub_ctx_pat.search(text_no_refs[window_s:window_e]):
            return ym.group(0)
    return ""


def _extract_venue(text: str) -> str:
    """Extract a canonical conference / journal name.

    Priority: real conferences/journals ALWAYS win over preprint servers
    (ArXiv). If both are present, a paper is a CVPR/NeurIPS/AAAI paper that
    *also* has an arXiv preprint — its source should be the real venue.

    Returns the canonical alias (e.g. "CVPR") if one of the known venues is
    found; otherwise returns the raw match of the first journal/conference
    line pattern.
    """
    if not text:
        return ""
    # Strip References/Bibliography section BEFORE searching for venues —
    # References cite 100+ OTHER papers' venues (e.g. "NIPS 2017", "CVPR 2024")
    # and those must NEVER be misinterpreted as THIS paper's venue.
    # For short papers (<8000 chars), the entire references section would
    # otherwise be included in the "front" scan.
    body_no_refs = _strip_references_section(text)
    front = body_no_refs[:8000]
    lower_front = front[:5000]

    # Pre-compute DOI spans: matches that fall entirely inside a DOI string
    # (e.g. "TPAMI" inside "10.1109/TPAMI.2016.xxx") do NOT count as real
    # venue mentions — they are just journal codes in the DOI identifier.
    doi_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in _DOI_RE.finditer(lower_front)
    ]
    def _match_inside_doi(match_start: int, match_end: int) -> bool:
        return any(ds <= match_start and de >= match_end for ds, de in doi_spans)

    # Pass 1: real (non-preprint) venues only. This guarantees that a paper
    # accepted to AAAI is marked AAAI even when "arXiv:2305.xxx" appears
    # earlier in the text.
    _ambig_venue_lower = {v.lower() for v in _AMBIGUOUS_SOURCE_WORDS}
    for canonical, patterns in _VENUE_PATTERNS:
        if _venue_is_preprint(canonical):
            continue
        is_ambiguous = canonical.lower() in _ambig_venue_lower
        for pat in patterns:
            m = re.search(pat, lower_front, re.IGNORECASE)
            if m and not _match_inside_doi(m.start(), m.end()):
                # Ambiguous-word disambiguation guard: "nature" / "science" /
                # "cell" are common English words that appear in body text.
                # Only accept these as genuine venue matches if the match
                # has DIRECT journal-style publication context on the same
                # or adjacent lines (NOT random publication markers elsewhere
                # on the page, which would produce false positives).
                if is_ambiguous:
                    if not _ambiguous_word_has_own_publication_context(
                        lower_front, m.start(), m.end(),
                    ):
                        continue  # common-word occurrence → SKIP this match
                return canonical

    # Pass 2: generic "Proceedings of ..." / journal line patterns
    proc_match = re.search(
        r"Proceedings\s+of\s+(?:the\s+)?[A-Z][A-Za-z0-9\-/.,:;() ]{5,120}",
        front,
    )
    if proc_match:
        raw = proc_match.group(0).strip().rstrip(".,")
        # Clean up common trailing page/volume fragments
        raw = re.sub(r"\s+(?:pp\.?|pages?|vol\.?|volume)\s*\d.*$", "", raw, flags=re.IGNORECASE)
        if len(raw) >= 8:
            return raw

    # Pass 3: IEEE / ACM / Springer journal patterns
    journal_match = re.search(
        r"(?:IEEE|ACM|Springer|Elsevier)\s+(?:Transactions|Journal|Magazine|Letters|Proceedings)\s+on\s+[A-Z][A-Za-z0-9\-/.,:;() ]{5,100}",
        front,
    )
    if journal_match:
        return journal_match.group(0).strip().rstrip(".,")

    # Pass 4 (last resort): preprint server (ArXiv). This only runs when NO
    # real publication venue could be identified.
    for canonical, patterns in _VENUE_PATTERNS:
        if not _venue_is_preprint(canonical):
            continue
        for pat in patterns:
            if re.search(pat, lower_front, re.IGNORECASE):
                return canonical

    return ""


def _extract_chinese_thesis_info(text: str) -> dict[str, str]:
    """Extract metadata specific to Chinese theses/dissertations.

    Keys returned: source (university + degree), advisors, defense_date.
    All values default to "" when not found.
    """
    if not text:
        return {"source": "", "advisors": "", "defense_date": ""}

    result: dict[str, str] = {"source": "", "advisors": "", "defense_date": ""}
    front = text[:8000]

    # Degree type
    degree = ""
    dm = _DEGREE_TYPE_RE.search(front)
    if dm:
        degree = dm.group(0)

    # University / institute
    univ = ""
    um = _UNIV_RE.search(front)
    if um:
        univ = um.group(0)

    # Build source: 大学 + 学位论文
    source_parts = [p for p in (univ, degree) if p]
    if source_parts:
        result["source"] = "，".join(source_parts)

    # Advisor / supervisor
    am = _ADVISOR_RE.search(front)
    if am:
        advisor = am.group(1).strip()
        # Clean up trailing department / title fragments
        advisor = re.split(r"\s*(?:教授|副教授|讲师|研究员|副研究员|博士|院士)\s*", advisor)[0].strip()
        advisor = re.sub(r"[\u4e00-\u9fa5]{0,4}(?:学院|系|所|中心|实验室|研究院).*$", "", advisor).strip()
        if 2 <= len(advisor) <= 20:
            result["advisors"] = advisor

    # Defense date
    ddm = _DEFENSE_DATE_RE.search(front)
    if ddm:
        result["defense_date"] = ddm.group(1).strip()
    else:
        cdm = _CN_DATE_RE.search(front[:3000])
        if cdm:
            result["defense_date"] = cdm.group(0).strip()

    return result


# =============================================================================
# Anti-hallucination post-validation helpers
# =============================================================================

# Explicit official "Keywords" section markers. Papers that have official keywords
# ALWAYS use one of these standard headings. If NONE are present, the paper has
# NO official keywords — any "keywords" produced by the LLM must be discarded,
# because the LLM is just summarizing body topics (hallucinating official metadata).
_KEYWORDS_SECTION_RE = re.compile(
    r"(?:"
    # Markdown heading:  ## Keywords, # Keywords, ## 关键词.
    # The heading marker is unambiguous because body paragraphs never start
    # with a heading `#` followed by the keywords word.
    r"^#+\s*(?:Keywords?|关键词|关键字)\s*$"
    # Explicit list-style line that STARTS with the heading AND is followed
    # by a colon (e.g. "Keywords: X, Y, Z") — this never occurs in prose.
    r"|^\s*(?:Keywords?|Index\s+Terms?|Key\s+phrases?|关键词|关键字)\s*[:：]\s*"
    r")",
    re.MULTILINE | re.IGNORECASE,
)

# Ambiguous journal names that are ALSO common English words.
# These words will match in any body paragraph that happens to discuss
# "... nature of ...", "... cell biology ...", "... science shows ...".
# We MUST require ADDITIONAL publication context (Vol/Issue/pp/copyright/DOI
# with venue-specific prefix) before accepting these as genuine source matches.
_AMBIGUOUS_SOURCE_WORDS = {
    "Nature",
    "Science",
    "Cell",
}

# Explicit PUBLICATION CONTEXT patterns — strong evidence that a nearby venue
# name is NOT just a common-word occurrence in body text.
_PUBLICATION_CONTEXT_RE = re.compile(
    r"(?:"
    r"Proceedings\s+of\s+(?:the\s+)?"
    r"|Published\s+in\s+"
    r"|To\s+appear\s+(?:in|at)\s+"
    r"|Accepted\s+(?:at|for\s+publication\s+in)\s+"
    r"|©\s*\d{4}\s*[\u4e00-\u9fa5A-Za-z&\s]{0,60}?"
    r"|Copyright\s*©?\s*\d{4}"
    r"|Vol(?:ume)?\.?\s*\d+"
    r"|Issue\s+\d+"
    r"|pp\.\s*\d+[-–—]\s*\d+"
    r"|DOI:\s*10\.\d+/[A-Za-z]+"  # DOI publisher prefix e.g. 10.1038/nature1234 = Nature
    r"|ISSN\s*\d"
    r"|ISBN\s*\d"
    r"|博士学位论文|硕士学位论文|学士学位论文"   # Chinese thesis markers
    r")",
    re.IGNORECASE,
)


def has_explicit_keywords_section(text: str) -> bool:
    """Return True only if the paper contains a STANDARD keywords section.

    Official keywords sections use standardized headings (## Keywords,
    Index Terms, 关键词, etc.). If none of these headings appear in the paper,
    it has NO official keywords and we must leave the field blank. The LLM
    must NOT be allowed to invent "keywords" by summarizing body topics.
    """
    if not text:
        return False
    return bool(_KEYWORDS_SECTION_RE.search(text[:20000]))


def _has_publication_context_near(text: str, venue_span_start: int,
                                  venue_span_end: int,
                                  context_window: int = 500) -> bool:
    """Check if a venue-name occurrence sits near an explicit publication-
    context marker (Vol / Issue / pp / Proceedings of / Copyright © ...).

    This is the disambiguation guard for ambiguous words like 'Nature':
    - "the non-deterministic nature of LLMs"   → NO context → word, not journal
    - "Nature Vol. 623, pp. 456–462. © 2024"   → HAS context → real journal
    """
    start = max(0, venue_span_start - context_window)
    end = min(len(text), venue_span_end + context_window)
    region = text[start:end]
    return bool(_PUBLICATION_CONTEXT_RE.search(region))


def _ambiguous_word_has_own_publication_context(
    text: str, match_start: int, match_end: int,
) -> bool:
    """NARROW check for ambiguous words (Nature/Science/Cell) to distinguish a
    real journal venue mention from a common-English-word occurrence.

    Returns True ONLY if the ambiguous word match is directly embedded within
    a journal-citation style phrase on the SAME line / within the same
    sentence fragment. This avoids the false positive where an unrelated
    "© 2026 AAAI" footer 200 chars away gets credited to the phrase
    "the non-deterministic nature of modern LLMs" in the Introduction.
    """
    # Same-line context: only consider the line that contains the match, plus
    # the preceding and following lines (total ~3 lines max).
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end_fwd = text.find("\n", match_end)
    if line_end_fwd == -1:
        line_end_fwd = len(text)
    # Include up to 1 line before and after
    region_start = max(0, text.rfind("\n", 0, line_start - 1) + 1)
    region_end_next = text.find("\n", line_end_fwd + 1)
    region_end = line_end_fwd if region_end_next == -1 else region_end_next

    region = text[region_start:region_end]
    rel_start = match_start - region_start
    rel_end = match_end - region_start

    # Check A: the word itself is immediately followed by journal metadata
    # within the SAME line. Example patterns:
    #   "Nature Vol. 623"  "Nature, 623(7593)"  "Nature 623, pp."
    #   "Science 388 (6747): 1234–1240"  "Cell 186, 1–15"
    suffix = region[rel_end:rel_end + 160]
    if re.match(
        r"\s*[,，]?\s*(?:Vol(?:ume)?\.?\s*\d|[0-9]+\s*\(|[0-9]+,?\s*pp|"
        r"DOI\s*[:：]|ISSN\s*[:：]?|ISBN\s*[:：]?|©|Copyright|"
        r"\([0-9]{4}\)|[0-9]{4}\s*[,，]|"
        r"Published\s+by|Springer|Elsevier|Wiley|Macmillan|American\s+Association)",
        suffix,
        re.IGNORECASE,
    ):
        return True

    # Check B: preceded within 60 chars by "Published in/To appear in/Accepted at"
    prefix = region[max(0, rel_start - 120):rel_start]
    if re.search(
        r"(?:Published\s+in|To\s+appear\s+in|Accepted\s+(?:at|for)|"
        r"In\s+(?:press|proceedings)|Appeared\s+in)\s+[:：]?\s*[A-Z]?\s*$",
        prefix,
        re.IGNORECASE,
    ):
        return True

    # Check C: "Nature" journal style citation line with ISSN/DOI on same
    # paragraph — any DOI or ISSN in the 3-line region counts as a strong
    # signal ONLY IF the region ALSO has Vol/Issue/pp metadata.
    if re.search(r"(?:Vol\.?\s*\d|Issue\s*\d|pp\.\s*\d)", region, re.IGNORECASE) \
       and re.search(r"(?:DOI\s*[:：]|ISSN\s*[:：]?|10\.\d{4,9}/)", region, re.IGNORECASE):
        return True

    return False


def source_is_plausible_publication(source_candidate: str, full_text: str) -> tuple[bool, str]:
    """Validate an LLM-proposed source field for hallucination.

    Returns (is_valid, reason) where:
    - is_valid = True  → the source passes anti-hallucination checks
    - is_valid = False → the source should be CLEARED (set to "")
    - reason           → short debug string explaining the verdict
    """
    if not source_candidate or not source_candidate.strip():
        return False, "empty"
    src = source_candidate.strip()
    src_lower = src.lower()

    # Body-only text (with References/Bibliography section stripped) for
    # ALL occurrence checks. Cited venues in References NEVER count as
    # evidence that THIS paper was published there.
    body_no_refs = _strip_references_section(full_text or "")

    # Pass 1: known canonical venues from the VENUE_PATTERNS table.
    # If the LLM gave a name that matches ANY of our high-confidence alias
    # patterns, we trust it (the regex library is already curated).
    # EXCEPTION: ambiguous common words ("Nature", "Science", "Cell") —
    # these are also listed in _VENUE_PATTERNS (they ARE real venues) but
    # they appear as normal vocabulary in body text. For these, we MUST
    # fall through to Pass 3 below, which checks publication-context.
    _ambig_lower = {a.lower() for a in _AMBIGUOUS_SOURCE_WORDS}
    _is_ambig = src_lower in _ambig_lower or any(
        amb in src_lower for amb in _ambig_lower
    )
    if not _is_ambig:
        for canonical, patterns in _VENUE_PATTERNS:
            if src_lower == canonical.lower():
                # Double-check: known venue must actually appear in the paper
                # BODY (not inside References section!). This prevents LLM
                # from saying "NeurIPS" when the paper only cites other
                # NeurIPS papers in its bibliography.
                in_paper_body = bool(re.search(re.escape(canonical), body_no_refs, re.IGNORECASE))
                if in_paper_body:
                    return True, f"known_venue:{canonical}"
                # Even if exact canonical not in paper body, check alias
                # patterns still in body (e.g. LLM says AAAI, body has AAAI-26).
                any_pattern_found = False
                for pat in patterns:
                    if re.search(pat, body_no_refs, re.IGNORECASE):
                        any_pattern_found = True
                        break
                if any_pattern_found:
                    return True, f"known_venue_alias_in_text:{canonical}"
                # Not present in body text at all — this LLM source is a pure guess.
                return False, f"known_venue_absent_from_body:{canonical}"
            for pat in patterns:
                if re.fullmatch(pat, src, re.IGNORECASE):
                    # Still must verify at least one alias actually appears in body
                    any_pattern_found = any(
                        bool(re.search(p, body_no_refs, re.IGNORECASE))
                        for p in patterns
                    )
                    if any_pattern_found:
                        return True, f"known_venue_pattern:{canonical}"
                    # Reject — alias never appears in body either.
                    return False, f"known_venue_pattern_absent_from_body:{canonical}"

    # Pass 2: Chinese thesis format. Accept "XX大学，硕士/博士学位论文" etc.
    if _UNIV_RE.search(src) and _DEGREE_TYPE_RE.search(src):
        return True, "chinese_thesis_format"

    # Pass 3: Ambiguous-common-word disambiguation. For "Nature"/"Science"/etc.
    # we require an EXPLICIT publication context that is DIRECTLY ASSOCIATED
    # with the word itself (same line / same phrase, not merely somewhere on
    # the page). The generic _has_publication_context_near wrongly credits
    # unrelated "© 2026 AAAI" footers to body occurrences like "the nature
    # of modern LLMs".
    for amb in _AMBIGUOUS_SOURCE_WORDS:
        if amb.lower() in src_lower and len(src) <= len(amb) + 5:
            found_with_context = False
            for vm in re.finditer(re.escape(amb), body_no_refs, re.IGNORECASE):
                if _ambiguous_word_has_own_publication_context(
                    body_no_refs, vm.start(), vm.end(),
                ):
                    found_with_context = True
                    break
            if not found_with_context:
                return False, f"ambiguous_word_no_direct_journal_context:{amb}"
            return True, f"ambiguous_word_with_pub_context:{amb}"

    # Pass 4: generic venue — must appear in the paper BODY NEAR publication
    # context markers (Vol/Issue/pp/Proceedings of/Copyright etc.) OR be a
    # "Proceedings of XXX" / "IEEE Transactions on XXX" style string.
    generic_venue_pat = re.compile(
        r"(?:Proceedings\s+of|IEEE/ACM\s+|Transactions\s+on|Journal\s+of|Conference\s+on)",
        re.IGNORECASE,
    )
    if generic_venue_pat.search(src):
        return True, "proceedings_or_txn_format"

    # Final check: does the source string actually appear in the BODY AND
    # if so, is it near publication context?
    safe_pat = re.escape(src)
    matches = list(re.finditer(safe_pat, body_no_refs, re.IGNORECASE))[:20]
    if not matches:
        return False, "not_found_in_paper_body"
    any_in_context = any(
        _has_publication_context_near(body_no_refs, m.start(), m.end())
        for m in matches
    )
    if any_in_context:
        return True, "found_in_body_pub_context"
    # Last-resort guardrail: if we couldn't verify it but it's a short string,
    # reject to avoid LLM guesses.
    if len(src) <= 12:
        return False, "unverified_short_string"
    return True, "long_unverified_pass"


def _build_markdown_metadata_prompt(payload: dict[str, str]) -> tuple[str, str, str]:
    """Build (system_prompt, user_prompt, combined_text) for Markdown input.

    Markdown input (from MinerU) is already clean structured text, so we:
    - Skip the PDF-structure warnings (no %PDF-/obj/endobj in Markdown)
    - Use a single [ABSTRACT_REGION] + [MARKDOWN_CONTENT] layout instead of
      the multi-region OCR layout
    - Guide the model to leverage Markdown headings (# title, ## Abstract)
    """
    abstract_region = payload.get("abstract_region", "")
    # Truncate candidate text for large papers - metadata extraction only
    # needs the front matter (title, authors, abstract), not the full body.
    candidate_text = payload.get("candidate_text", "")[:40000]

    combined_text_parts = []
    if abstract_region:
        combined_text_parts.append("[ABSTRACT_REGION]\n" + abstract_region)
    if candidate_text:
        combined_text_parts.append("[MARKDOWN_CONTENT]\n" + candidate_text)
    combined_text = "\n\n".join(part for part in combined_text_parts if part.strip())

    system_prompt = (
        "你是一名精通中英文的论文摘要与元信息抽取助手，必须确保翻译质量。"
        "输入是经 MinerU 解析的 Markdown 文本，结构清晰，标题通常以 # 开头，"
        "摘要通常位于 '## Abstract' 或 '## 摘要' 标题下方。"
        "输入中 [ABSTRACT_REGION] 是本地算法定位的摘要候选区域，应优先查看；"
        "[MARKDOWN_CONTENT] 是完整论文 Markdown，用于补充提取其他元信息。"
        "最终只输出严格合法的 JSON，不要输出任何额外解释、Markdown、代码块或前后缀文字。"
        "JSON 必须且只能包含以下键：title_cn, title_en, authors, source, abstract_cn, abstract_en, keywords, year, doi。"
        "关键指令 - 英文论文：title_en 输出原文标题，title_cn 必须输出中文翻译；"
        "abstract_en 输出原文摘要，abstract_cn 必须输出中文翻译。"
        "关键指令 - 中文论文：title_cn 输出原文标题，title_en 留空；"
        "abstract_cn 输出原文摘要，abstract_en 留空。"
        "如果论文同时包含中英文标题或摘要，则分别提取对应语言的原文。"
        "翻译要求：对于英文论文，必须将标题和摘要翻译为流畅、准确的中文，"
        "不得返回英文原文或部分英文；专业术语翻译要准确；保持学术严谨性。"
        "如果没有明确摘要，就返回空字符串，不要编造。"
        "作者请尽量去掉单位、脚注编号和通讯作者标记，多个作者用分号分隔。"
        "【绝对禁止编造！找不到就必须留空字符串】："
        "对于 keywords/source/year/doi 四个字段，任何在原文中找不到明确、直接、官方声明的内容，都必须输出空字符串 ''，绝对不允许猜测、推断、总结或编造。"
        "  ❌ 反例1（source）：正文里出现 'the non-deterministic nature of modern LLMs'，"
        "    这里的 'nature' 是普通单词，意思是'性质/特性'，绝不能把 source 填成 'Nature'！"
        "    只有出现 'Nature Vol.XXX pp.YYY'、'Published in Nature'、'© 2024 Springer Nature' "
        "    等明确出版上下文时，才能认定来源是 Nature 期刊。"
        "  ❌ 反例2（keywords）：如果原文没有『## Keywords』『Keywords:』『关键词』"
        "    『Index Terms』等官方关键词标题区，就必须输出空字符串 ''，"
        "    绝对不允许自己总结文章主题词作为关键词！哪怕这些词在正文里出现过也不行。"
        "  ❌ 反例3（year）：参考文献/引用里的 'arXiv:2310.11511'、"
        "    '(Zhao et al., 2023)'、'2021 Conference on ...' 是别人论文的年份，"
        "    不能作为本文年份。本文年份必须来自首页标题附近、DOI 附近、版权行（© 20XX）、"
        "    或明确的 'Proceedings of XX 20XX' / 'Published 20XX'。"
        "  ❌ 反例4（source 参考文献）：References 里的 'Proceedings of the 2021 "
        "    Conference on EMNLP' 是被引用论文的出处，不能当成本文 source。"
        "    source 只能从首页页脚、版权声明、标题下方、header/footer 里找。"
        "【source 提取重点】：source 请填写期刊名或会议名（优先使用缩写或通用名，如 CVPR、NeurIPS、TPAMI、ACL、EMNLP 等，而不是冗长全称）；"
        "若找到论文在 XX会议/期刊 上发表的明确信息，请务必提取。"
        "常见会议/期刊名称参考：CVPR, ICCV, ECCV (计算机视觉), NeurIPS, ICML, ICLR (机器学习), ACL, EMNLP, NAACL (NLP), AAAI, IJCAI (AI), KDD (数据挖掘), SIGGRAPH (图形学), "
        "TPAMI, IJCV, JMLR (顶刊), ArXiv 等。"
        "【DOI 提取重点】：DOI 格式为 10.XXXX/XXXXXX，通常位于首页脚注、页眉或 PDF 元数据中，前缀为 'doi:', 'DOI:', 'https://doi.org/'；"
        "如果找到明确 DOI，请完整提取（注意包含前后两段）。"
        "【year 提取重点】：year 为四位数字年份，优先提取论文发表年份，通常出现在 DOI 附近、会议/期刊名附近或首页底部脚注；"
        "对于中文毕业论文，优先使用答辩日期/提交日期/完成日期中的年份。"
        "【中文毕业论文特判】：如果文本中出现「XX大学」「XX学院」「硕士学位论文」「博士学位论文」「指导教师」「答辩日期」等关键词，"
        "请将 source 提取为学校名称加学位类型（如「清华大学，硕士学位论文」）；year 优先使用答辩/提交/完成日期的年份；"
        "作者字段只需学生姓名，不需导师（导师信息已在 source 中体现）。"
        "keywords 用分号分隔，仅在存在官方 Keywords section 时填写。"
    )
    user_prompt = (
        "请从以下 Markdown 文本中抽取论文摘要与元信息。\n"
        "重点：优先从 [ABSTRACT_REGION] 提取摘要，若该区域不含摘要，"
        "再从 [MARKDOWN_CONTENT] 中寻找 '## Abstract' 或 '## 摘要' 标题下的内容。\n"
        "标题通常位于 Markdown 开头的 # 一级标题中。\n"
        "作者通常紧跟标题下方，可能带有单位信息，请只保留作者姓名。\n\n"
        f"{combined_text}"
    )
    return system_prompt, user_prompt, combined_text


def _build_ocr_metadata_prompt(payload: dict[str, str]) -> tuple[str, str, str]:
    """Build (system_prompt, user_prompt, combined_text) for OCR input.

    Preserves the original multi-region prompt that warns about PDF
    structural noise (%PDF-, obj, endobj, stream, etc.).
    """
    metadata_pages = payload.get("metadata_pages_text", "")
    first_pages = payload.get("first_pages_marked_text", payload.get("first_pages_text", ""))
    metadata_text = payload.get("metadata_text", "")
    # Truncate candidate text for large papers - metadata extraction only
    # needs the front matter (title, authors, abstract), not the full body.
    candidate_text = payload.get("candidate_text", "")[:40000]
    abstract_region = payload.get("abstract_region", "")

    combined_text_parts = []
    if abstract_region:
        combined_text_parts.append("[ABSTRACT_REGION]\n" + abstract_region)
    combined_text_parts.extend([
        "[METADATA_PAGES]\n" + metadata_pages,
        "[FIRST_PAGES]\n" + first_pages,
        "[METADATA_TEXT]\n" + metadata_text,
        "[CANDIDATE_TEXT]\n" + candidate_text,
    ])
    combined_text = "\n\n".join(part for part in combined_text_parts if part.strip())

    system_prompt = (
        "你是一名精通中英文的论文摘要与元信息抽取助手，必须确保翻译质量。"
        "输入中包含多个文本区域，其中 [ABSTRACT_REGION] 是我们通过本地算法定位的摘要候选区域，这是最可能包含摘要的地方。"
        "你的首要任务是先查看 [ABSTRACT_REGION] 区域，如果其中包含摘要内容，则从中提取；否则从其他区域寻找。"
        "你必须忽略 PDF 文件头、对象流、压缩流、XMP/XML、LaTeX 结构串，以及任何类似 %PDF-、obj、endobj、stream、TJ、Tm 的内容。"
        "最终只输出严格合法的 JSON，不要输出任何额外解释、Markdown、代码块或前后缀文字。"
        "JSON 必须且只能包含以下键：title_cn, title_en, authors, source, abstract_cn, abstract_en, keywords, year, doi。"
        "关键指令 - 英文论文：title_en 输出原文标题，title_cn 必须输出中文翻译；abstract_en 输出原文摘要，abstract_cn 必须输出中文翻译。"
        "关键指令 - 中文论文：title_cn 输出原文标题，title_en 留空；abstract_cn 输出原文摘要，abstract_en 留空。"
        "如果论文同时包含中英文标题或摘要，则分别提取对应语言的原文。"
        "翻译要求：对于英文论文，必须将标题和摘要翻译为流畅、准确的中文，不得返回英文原文或部分英文；专业术语翻译要准确；保持学术严谨性。"
        "如果文中存在 PDF 元数据中的 title/subject/creator/keywords/doi，请结合正文候选进行判断，但不要直接把元数据字段原样当摘要。"
        "如果摘要跨页，请把正文候选中连贯的摘要内容合并后再输出。"
        "如果没有明确摘要，就返回空字符串，不要编造。"
        "作者请尽量去掉单位、脚注编号和通讯作者标记。"
        "【绝对禁止编造！找不到就必须留空字符串】："
        "对于 keywords/source/year/doi 四个字段，任何在原文中找不到明确、直接、官方声明的内容，都必须输出空字符串 ''，绝对不允许猜测、推断、总结或编造。"
        "  ❌ 反例1（source）：正文里出现 'the non-deterministic nature of modern LLMs'，"
        "    这里的 'nature' 是普通单词，意思是'性质/特性'，绝不能把 source 填成 'Nature'！"
        "    只有出现 'Nature Vol.XXX pp.YYY'、'Published in Nature'、'© 2024 Springer Nature' "
        "    等明确出版上下文时，才能认定来源是 Nature 期刊。"
        "  ❌ 反例2（keywords）：如果原文没有『## Keywords』『Keywords:』『关键词』"
        "    『Index Terms』等官方关键词标题区，就必须输出空字符串 ''，"
        "    绝对不允许自己总结文章主题词作为关键词！哪怕这些词在正文里出现过也不行。"
        "  ❌ 反例3（year）：参考文献/引用里的 'arXiv:2310.11511'、"
        "    '(Zhao et al., 2023)'、'2021 Conference on ...' 是别人论文的年份，"
        "    不能作为本文年份。本文年份必须来自首页标题附近、DOI 附近、版权行（© 20XX）、"
        "    或明确的 'Proceedings of XX 20XX' / 'Published 20XX'。"
        "  ❌ 反例4（source 参考文献）：References 里的 'Proceedings of the 2021 "
        "    Conference on EMNLP' 是被引用论文的出处，不能当成本文 source。"
        "    source 只能从首页页脚、版权声明、标题下方、header/footer 里找。"
        "【source 提取重点】：source 请填写期刊名或会议名（优先使用缩写或通用名，如 CVPR、NeurIPS、TPAMI、ACL、EMNLP 等，而不是冗长全称）；"
        "若找到论文在 XX会议/期刊 上发表的明确信息，请务必提取。"
        "常见会议/期刊名称参考：CVPR, ICCV, ECCV (计算机视觉), NeurIPS, ICML, ICLR (机器学习), ACL, EMNLP, NAACL (NLP), AAAI, IJCAI (AI), KDD (数据挖掘), SIGGRAPH (图形学), "
        "TPAMI, IJCV, JMLR (顶刊), ArXiv 等。"
        "【DOI 提取重点】：DOI 格式为 10.XXXX/XXXXXX，通常位于首页脚注、页眉或 PDF 元数据中，前缀为 'doi:', 'DOI:', 'https://doi.org/'；"
        "如果找到明确 DOI，请完整提取（注意包含前后两段）。"
        "【year 提取重点】：year 为四位数字年份，优先提取论文发表年份，通常出现在 DOI 附近、会议/期刊名附近或首页底部脚注；"
        "对于中文毕业论文，优先使用答辩日期/提交日期/完成日期中的年份。"
        "【中文毕业论文特判】：如果文本中出现「XX大学」「XX学院」「硕士学位论文」「博士学位论文」「指导教师」「答辩日期」等关键词，"
        "请将 source 提取为学校名称加学位类型（如「清华大学，硕士学位论文」）；year 优先使用答辩/提交/完成日期的年份；"
        "作者字段只需学生姓名，不需导师（导师信息已在 source 中体现）。"
    )
    user_prompt = (
        "请从以下输入中抽取论文摘要与元信息。\n"
        "重点：请优先从 [ABSTRACT_REGION] 区域提取摘要，该区域是我们定位的摘要候选区域。\n"
        "摘要必须来自可读正文，不是 PDF 结构、对象流或元数据字符串。\n"
        "如果封面页或版权页先出现，请继续从后续页面中寻找真正摘要。\n\n"
        f"{combined_text}"
    )
    return system_prompt, user_prompt, combined_text


def extract_metadata(payload: dict[str, str]) -> dict[str, str]:
    if not settings.llm_api_key:
        raise MetadataError("API key is not configured")

    extraction_method = payload.get("extraction_method", "")
    is_markdown_input = extraction_method == "mineru"

    if is_markdown_input:
        system_prompt, user_prompt, combined_text = _build_markdown_metadata_prompt(payload)
    else:
        system_prompt, user_prompt, combined_text = _build_ocr_metadata_prompt(payload)

    append_debug_record(
        payload.get("paper_id", "unknown"),
        "metadata_request",
        extraction_method=extraction_method,
        is_markdown_input=is_markdown_input,
        combined_text_len=len(combined_text),
    )
    try:
        content = _call_llm(
            system_prompt, user_prompt, temperature=0.0, timeout=300, json_object=True
        )
    except llm_client.LLMError as exc:
        raise MetadataError(str(exc)) from exc

    parsed = _parse_json_safely(content)
    append_debug_record(
        payload.get("paper_id", "unknown"),
        "metadata_response",
        model_content=content,
        parsed=parsed,
    )

    def _clean(value: object) -> str:
        return _normalize_list_text(value).strip()

    def _is_valid_abstract(text: str, title_cn: str = "", title_en: str = "") -> bool:
        if not text:
            return False
        stripped = text.strip()
        if len(stripped) < 80:
            return False
        lower = stripped.lower()
        padded = f" {lower} "
        # Check for unambiguous PDF structural noise markers.
        bad_tokens = (
            "%pdf-",
            " endobj ",
            " obj <<",
            " endstream ",
            " xref ",
            " trailer ",
        )
        if any(token in padded for token in bad_tokens):
            return False
        # Reject if the text starts with title-like markers and has too many short lines
        # (typical when the entire first page text is dumped as abstract)
        title_cn_stripped = (title_cn or "").strip()
        title_en_stripped = (title_en or "").strip()
        if title_cn_stripped and stripped.startswith(title_cn_stripped):
            remainder = stripped[len(title_cn_stripped):].strip()
            if len(remainder) < 80:
                return False
        if title_en_stripped and stripped.lower().startswith(title_en_stripped.lower()):
            remainder = stripped[len(title_en_stripped):].strip()
            if len(remainder) < 80:
                return False
        # Reject obvious non-abstract markers: starts with Table of Contents,
        # paper metadata markers, or lists of authors/affiliations.
        leading_indicators = (
            "contents",
            "table of contents",
            "copyright",
            "all rights reserved",
            "ieee",
            "acm ",
            "received ",
            "revised ",
            "accepted ",
        )
        first_200 = lower[:200]
        if any(ind in first_200 for ind in leading_indicators):
            return False
        return True

    abstract_fallback = _clean(parsed.get("abstract", ""))
    abstract_cn = _clean(parsed.get("abstract_cn", abstract_fallback))
    abstract_en = _clean(parsed.get("abstract_en", abstract_fallback))
    if not _is_valid_abstract(abstract_cn):
        abstract_cn = ""
    if not _is_valid_abstract(abstract_en):
        abstract_en = ""

    # --- Language-aware abstract normalization ---
    # Determine the paper's original language from the FULL raw text, not
    # just the abstract_region. Chinese academic papers commonly contain both
    # Chinese and English abstracts; _locate_abstract_region() may capture
    # the English abstract, causing _detect_language(abstract_region) to
    # return "en" and misclassify a Chinese paper as English. Using the full
    # raw_text gives a representative sample of the paper's actual language
    # distribution (body text dominates over abstracts).
    paper_id_meta = payload.get("paper_id", "unknown")
    paper_lang = detect_paper_language(payload)

    # If full-text detection fails (no raw_text), fall back to LLM output
    if not paper_lang:
        abstract_region = payload.get("abstract_region", "")
        paper_lang = _detect_language(abstract_region)

    if not paper_lang:
        if abstract_cn and _detect_language(abstract_cn) == "zh" and not abstract_en:
            paper_lang = "zh"
        elif abstract_en and _detect_language(abstract_en) == "en" and not abstract_cn:
            paper_lang = "en"
        elif abstract_cn and _detect_language(abstract_cn) == "zh":
            paper_lang = "zh"
        elif abstract_en and _detect_language(abstract_en) == "en":
            paper_lang = "en"

    if paper_lang == "zh":
        # Chinese paper: abstract_cn = original Chinese, abstract_en = ""
        if not abstract_cn:
            if abstract_en and _detect_language(abstract_en) == "zh":
                abstract_cn = abstract_en
            elif abstract_fallback and _detect_language(abstract_fallback) == "zh":
                abstract_cn = abstract_fallback
        abstract_en = ""
    elif paper_lang == "en":
        # English paper: abstract_en = original English, abstract_cn = Chinese translation
        if not abstract_en:
            if abstract_cn and _detect_language(abstract_cn) == "en":
                abstract_en = abstract_cn
            elif abstract_fallback and _detect_language(abstract_fallback) == "en":
                abstract_en = abstract_fallback
        # Validate abstract_cn is actually Chinese; if not, re-translate
        if not abstract_cn or _detect_language(abstract_cn) != "zh":
            append_debug_record(
                paper_id_meta,
                "abstract_cn_retranslate",
                reason="Chinese abstract missing or not in Chinese",
                old_abstract_cn_preview=abstract_cn[:200] if abstract_cn else "",
            )
            abstract_cn = _translate_to_chinese(abstract_en, paper_id_meta)
    else:
        # Unknown language: validate each field's language, clear mismatches
        if abstract_cn and _detect_language(abstract_cn) != "zh":
            abstract_cn = ""
        if abstract_en and _detect_language(abstract_en) != "en":
            abstract_en = ""

    keywords = _clean(parsed.get("keywords", ""))
    authors = _clean(parsed.get("authors", ""))
    title_cn = _clean(parsed.get("title_cn", ""))
    title_en = _clean(parsed.get("title_en", ""))
    source = _clean(parsed.get("source", ""))
    year = _clean(parsed.get("year", ""))
    doi = _clean(parsed.get("doi", ""))
    # Validate title languages to prevent cross-language contamination.
    if title_cn and _detect_language(title_cn) != "zh":
        title_cn = ""
    if title_en and _detect_language(title_en) != "en":
        title_en = ""
    # Re-run abstract validity check WITH title context to catch cases where
    # the LLM dumped title+authors+affiliations as the abstract.
    if abstract_cn and not _is_valid_abstract(abstract_cn, title_cn, title_en):
        abstract_cn = ""
    if abstract_en and not _is_valid_abstract(abstract_en, title_cn, title_en):
        abstract_en = ""
    if abstract_fallback and not _is_valid_abstract(abstract_fallback, title_cn, title_en):
        abstract_fallback = ""
    # Strip citation markers and stray HTML tags from abstract fields.
    abstract_cn = _clean_abstract_citations(abstract_cn)
    abstract_en = _clean_abstract_citations(abstract_en)
    abstract_fallback = _clean_abstract_citations(abstract_fallback)
    abstract = abstract_cn or abstract_en or abstract_fallback

    # =========================================================================
    # Local regex enhancement: fill / augment source, year, DOI.
    # Strategy: regex extraction acts as (a) fallback when LLM returns empty,
    # (b) canonical override when LLM gave a verbose/non-standard name and the
    # regex found a known short alias (e.g. LLM says "Proceedings of the IEEE/CVF
    # Conference on CV and Pattern Recognition" -> regex says "CVPR").
    # =========================================================================
    _full_text_for_regex = "\n".join(filter(None, [
        payload.get("raw_text", "")[:15000],
        payload.get("full_text", "")[:15000],
        payload.get("candidate_text", "")[:15000],
        payload.get("metadata_pages_text", "")[:10000],
        payload.get("first_pages_text", "")[:10000],
        payload.get("metadata_text", "")[:10000],
        payload.get("abstract_region", "")[:5000],
    ]))

    is_chinese_thesis_env = (
        paper_lang == "zh"
        and bool(_DEGREE_TYPE_RE.search(_full_text_for_regex[:8000]) or _UNIV_RE.search(_full_text_for_regex[:8000]))
    )

    regex_boost_debug: dict[str, str] = {}

    # --- DOI: regex overrides LLM if LLM gave nothing OR regex gave a valid
    # canonical DOI that is clearly better (e.g. LLM returned "10.xxxx/...
    # [something]" with extra junk). We trust regex DOI when its prefix 10.xxx
    # matches what LLM gave but regex is cleaner.
    regex_doi = _extract_doi(_full_text_for_regex)
    if regex_doi:
        if not doi:
            regex_boost_debug["doi"] = f"regex_fill:{regex_doi}"
            doi = regex_doi
        elif doi != regex_doi and regex_doi in doi:
            # LLM included extra characters around the DOI (e.g. parentheses,
            # page numbers). Clean it up.
            regex_boost_debug["doi"] = f"regex_clean:{doi}->{regex_doi}"
            doi = regex_doi

    # Determine regex venue ONCE (used by both source & year logic below)
    regex_venue = "" if is_chinese_thesis_env else _extract_venue(_full_text_for_regex)

    # Is the LLM source pointing at a preprint server while regex found a real venue?
    llm_source_looks_preprint = (
        bool(source) and "arxiv" in source.lower() and not source.lower().startswith("proceedings")
    )
    regex_has_real_venue = bool(regex_venue) and not _venue_is_preprint(regex_venue)

    # --- Source / venue:
    # 1) Chinese thesis path (highest priority for zh papers with markers)
    if is_chinese_thesis_env:
        thesis_info = _extract_chinese_thesis_info(_full_text_for_regex)
        if thesis_info["source"]:
            if not source or len(source) < 4:
                regex_boost_debug["source"] = f"thesis_fill:{thesis_info['source']}"
                source = thesis_info["source"]
            elif thesis_info["source"] not in source and len(source) < 15:
                # LLM gave something short/unclear, prefer explicit thesis source
                regex_boost_debug["source"] = f"thesis_override:{source}->{thesis_info['source']}"
                source = thesis_info["source"]
        if thesis_info["advisors"] and authors:
            regex_boost_debug["advisors"] = thesis_info["advisors"]
    # 2) Known venue regex canonical override
    elif regex_venue:
        if not source:
            regex_boost_debug["source"] = f"venue_fill:{regex_venue}"
            source = regex_venue
        else:
            lower_source = source.lower()
            # UNCONDITIONAL upgrade path: regex found a real conference/journal
            # but LLM hallucinated/misclassified it as arXiv. A paper that is
            # accepted to AAAI is an AAAI paper regardless of its arXiv copy.
            if regex_has_real_venue and llm_source_looks_preprint:
                regex_boost_debug["source"] = f"venue_upgrade_from_preprint:{source}->{regex_venue}"
                source = regex_venue
            # Normal upgrade: regex found a known short alias that LLM missed /
            # expressed verbosely (e.g. "Proc. IEEE/CVF CVPR" → "CVPR").
            elif regex_venue.lower() not in lower_source and len(regex_venue) <= len(source):
                regex_boost_debug["source"] = f"venue_upgrade:{source}->{regex_venue}"
                source = regex_venue

    # --- Year:
    prefer_defense = is_chinese_thesis_env
    regex_year = _extract_year(_full_text_for_regex, prefer_defense=prefer_defense)
    if regex_year:
        if not year:
            regex_boost_debug["year"] = f"regex_fill:{regex_year}"
            year = regex_year
        elif year != regex_year:
            # FORCE override when: LLM source is a preprint (ArXiv) but regex
            # found a real venue + year. The LLM almost certainly picked up
            # the arXiv submission year instead of the publication year.
            force_override = regex_has_real_venue and llm_source_looks_preprint
            if prefer_defense or force_override:
                reason = "thesis" if prefer_defense else "real_venue_over_preprint"
                regex_boost_debug["year"] = f"{reason}_year_override:{year}->{regex_year}"
                year = regex_year
            else:
                # Otherwise only override if LLM year looks suspicious:
                # not a 4-digit year, or way outside plausible range.
                try:
                    year_i = int(year)
                    if year_i < 1950 or year_i > 2099:
                        regex_boost_debug["year"] = f"regex_override_bad_llm:{year}->{regex_year}"
                        year = regex_year
                except (TypeError, ValueError):
                    regex_boost_debug["year"] = f"regex_override_non_numeric:{year}->{regex_year}"
                    year = regex_year

    if regex_boost_debug:
        append_debug_record(
            payload.get("paper_id", "unknown"),
            "metadata_regex_boost",
            boost=regex_boost_debug,
            is_chinese_thesis=is_chinese_thesis_env,
        )

    # =========================================================================
    # Anti-hallucination POST-VALIDATION (FINAL GUARDRAIL)
    #
    # Even with strong prompts, LLMs occasionally hallucinate — especially for
    # ambiguous common words ("Nature" ≠ journal Nature) and body-topic
    # summarization presented as "keywords". This section clears any field
    # that cannot be rigorously verified against the original paper text.
    # =========================================================================
    hallucination_clears: dict[str, str] = {}

    # ---- 1. Keywords:  must have an EXPLICIT official Keywords section.
    # LLM must NOT be allowed to "summarize body topics" as keywords.
    if keywords:
        explicit_kw_section = has_explicit_keywords_section(
            payload.get("raw_text", "")
            or payload.get("full_text", "")
            or payload.get("candidate_text", "")
            or payload.get("metadata_pages_text", "")
            or payload.get("first_pages_text", "")
            or payload.get("metadata_text", "")
            or combined_text
        )
        if not explicit_kw_section:
            hallucination_clears["keywords"] = (
                f"cleared:'{keywords[:80]}...' → reason: no explicit Keywords/关键词 "
                f"section found in paper (LLM summarized body topics instead)"
            )
            keywords = ""

    # ---- 2. Source:  must pass anti-hallucination plausibility checks.
    if source:
        _text_for_source_check = (
            payload.get("raw_text", "")
            or payload.get("full_text", "")
            or payload.get("candidate_text", "")
            or payload.get("metadata_pages_text", "")
            or payload.get("first_pages_text", "")
            or payload.get("metadata_text", "")
            or combined_text
        )
        src_valid, src_reason = source_is_plausible_publication(source, _text_for_source_check)
        if not src_valid:
            hallucination_clears["source"] = (
                f"cleared:'{source}' → reason: {src_reason}"
            )
            source = ""

    # ---- 3. Year:  if we have a year, sanity-check it.
    # (a) Must be 4-digit 1950–2099.
    # (b) If the year did NOT come from our regex extraction (i.e. it was
    #     LLM-proposed or came from a non-pub-context source), verify the
    #     year actually appears in the NON-REFERENCES body text AND sits
    #     NEAR publication-context markers (or Chinese thesis date markers).
    #     Benchmark years (WMT 2014, ImageNet 2012, etc.) are explicitly
    #     rejected —宁空勿错: no context → empty the field.
    if year:
        # Basic range check (already done above during regex boost, but redo
        # here so the logic is centralized even if regex_year is empty).
        try:
            year_i = int(year)
            if year_i < 1950 or year_i > 2099:
                hallucination_clears["year"] = f"cleared:'{year}' → out of 1950-2099 range"
                year = ""
        except (TypeError, ValueError):
            hallucination_clears["year"] = f"cleared:'{year}' → not a numeric year"
            year = ""
        # If year still looks numeric, and regex extracted a DIFFERENT year
        # (or none at all), apply the publication-context existence check to
        # guard against LLM pulling "WMT 2014"-style benchmark years.
        if year:
            _text_for_year_check = (
                payload.get("raw_text", "")
                or payload.get("full_text", "")
                or payload.get("candidate_text", "")
                or payload.get("metadata_pages_text", "")
                or payload.get("first_pages_text", "")
                or payload.get("metadata_text", "")
                or combined_text
            )
            year_from_regex = regex_year if regex_year else ""
            llm_provided_unverified = (year != year_from_regex)
            if llm_provided_unverified:
                # Verify year actually appears in body (non-references) and
                # has nearby publication context or Chinese thesis markers.
                body_no_refs_y = _strip_references_section(_text_for_year_check)
                _year_ctx_pub_pat = re.compile(
                    r"(?:©|Copyright|Vol(?:ume)?\.?\s*\d|Issue\s+\d|pp\.\s*\d|"
                    r"Published\s+(?:in|by|on)|Accepted\s+(?:at|for)|ISBN\s|ISSN\s|"
                    r"博士学位论文|硕士学位论文|学士学位论文|"
                    r"答辩日期|提交日期|完成日期|定稿日期|"
                    r"Proceedings\s+of|Curran\s+Associates|IEEE\s+Computer\s+Society|"
                    r"ACM\s+Press|Springer|Elsevier|USENIX|ACL\s+Anthology|"
                    r"arXiv:\s*\d{4}\.\d{4,5})",
                    re.IGNORECASE,
                )
                verify_ok = False
                for ym in re.finditer(re.escape(year), body_no_refs_y):
                    y_s, y_e = ym.start(), ym.end()
                    # (guard) skip this match if it's inside a longer digit
                    # string like arXiv ID 1706.03762 where 1706 ≠ 2017
                    if y_s > 0 and body_no_refs_y[y_s - 1].isdigit():
                        continue
                    if y_e < len(body_no_refs_y) and body_no_refs_y[y_e].isdigit():
                        continue
                    # (guard) WMT / ImageNet style benchmark → skip outright
                    if _year_looks_like_inline_citation(y_s, y_e, body_no_refs_y):
                        continue
                    # Check for Chinese date suffix: "2024年"
                    suff = body_no_refs_y[y_e:y_e + 12]
                    if re.match(r"\s*年\s*(?:(?:0?[1-9]|1[0-2])\s*月)?", suff):
                        verify_ok = True
                        break
                    # Check for publication-context marker near the year
                    w_s = max(0, y_s - 600)
                    w_e = min(len(body_no_refs_y), y_e + 600)
                    if _year_ctx_pub_pat.search(body_no_refs_y[w_s:w_e]):
                        verify_ok = True
                        break
                if not verify_ok:
                    hallucination_clears["year"] = (
                        f"cleared:'{year}' → unverified: no occurrence in "
                        f"body-with-publication-context (regex year was "
                        f"'{year_from_regex}')"
                    )
                    year = ""

    # ---- 4. DOI:  format sanity check (10.XXXX/XXXXXX).
    if doi:
        valid_doi = _extract_doi(f"DOI: {doi}") or _extract_doi(doi)
        if not valid_doi:
            hallucination_clears["doi"] = f"cleared:'{doi}' → invalid DOI format"
            doi = ""
        elif valid_doi != doi:
            hallucination_clears["doi"] = f"cleaned:'{doi}'→'{valid_doi}'"
            doi = valid_doi

    if hallucination_clears:
        append_debug_record(
            payload.get("paper_id", "unknown"),
            "metadata_hallucination_guard",
            cleared_fields=hallucination_clears,
        )

    result = {
        "title_cn": title_cn,
        "title_en": title_en,
        "authors": authors,
        "source": source,
        "abstract": abstract,
        "abstract_cn": abstract_cn,
        "abstract_en": abstract_en,
        "keywords": keywords,
        "year": year,
        "doi": doi,
    }
    append_debug_record(payload.get("paper_id", "unknown"), "metadata_result", result=result)
    return result
