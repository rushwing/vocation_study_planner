"""Web research service: search for Chinese curriculum materials via Brave Search."""

from __future__ import annotations

import json
import logging
from urllib.parse import urlencode, urlparse

import httpx

from app.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

_BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

_GRADE_CN = {
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
    "10": "十",
    "11": "十一",
    "12": "十二",
}

_EXTRACT_SYSTEM = (
    "You are an educational content extractor. Given web search results about a study topic, "
    "extract key educational points from each source. "
    "Respond ONLY with a JSON array matching this schema:\n"
    '[{"title": "...", "source": "...", "url": "...", "key_points": ["point1", "point2"]}]\n'
    "Include 3-5 key_points per item. Use the language of the content (Chinese if content is Chinese). "
    "Output ONLY the JSON array, no markdown fences."
)


def _grade_to_cn(grade: str) -> str:
    return _GRADE_CN.get(str(grade).strip(), grade)


def _extract_source(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _build_query(subject: str, grade: str, description: str) -> str:
    grade_cn = _grade_to_cn(grade)
    short_desc = description[:30] if description else ""
    parts = [f"小红书 {grade_cn}年级 {subject} 学习方法 知识点"]
    if short_desc:
        parts.append(short_desc)
    return " ".join(parts)


async def _brave_search(query: str, n_results: int = 5) -> list[dict]:
    """Search via Brave Search API. Returns list of result dicts."""
    settings = get_settings()
    if not settings.BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not set; skipping web search")
        return []
    params = urlencode({"q": query, "count": n_results, "search_lang": "zh", "country": "CN"})
    url = f"{_BRAVE_SEARCH_ENDPOINT}?{params}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url,
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": settings.BRAVE_API_KEY,
                },
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("web", {}).get("results", [])[:n_results]
            # Normalise to {title, url, description} — same shape as before
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "description": r.get("description", ""),
                }
                for r in results
            ]
    except Exception as exc:
        logger.warning("Brave search failed for query=%r: %s", query, exc)
        return []


async def _extract_materials(search_results: list[dict]) -> list[dict]:
    """Use LLM to extract structured materials from Brave search results."""
    content_blocks = []
    for i, result in enumerate(search_results):
        title = result.get("title", "")
        url = result.get("url", "")
        description = result.get("description", "")
        content_blocks.append(f"### Source {i + 1}: {title}\nURL: {url}\n{description}")

    combined = "\n\n".join(content_blocks)
    if not combined.strip():
        return []

    try:
        result_text, _, _ = await llm_service.chat_complete(
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user", "content": combined},
            ],
            max_tokens=1024,
            temperature=0.3,
        )
        extracted = json.loads(result_text)
        if isinstance(extracted, list):
            return extracted
        return []
    except Exception as exc:
        logger.warning("Material extraction failed: %s", exc)
        return []


async def search_study_materials(
    subject: str,
    grade: str,
    description: str,
    n_results: int = 5,
) -> list[dict]:
    """Search for Chinese educational materials via Brave Search.

    Returns list[{title, source, url, key_points}]. Returns [] on any error.
    """
    try:
        query = _build_query(subject, grade, description)
        search_results = await _brave_search(query, n_results=n_results)
        if not search_results:
            return []
        return await _extract_materials(search_results)
    except Exception as exc:
        logger.warning("search_study_materials failed for subject=%r: %s", subject, exc)
        return []
