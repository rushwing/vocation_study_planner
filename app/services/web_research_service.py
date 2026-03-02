"""Web research service: search for Chinese curriculum materials via Jina Search + Reader."""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import quote, urlparse

import httpx

from app.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

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


async def _jina_search(query: str, n_results: int = 5) -> list[dict]:
    """Search via Jina Search API. Returns list of result dicts."""
    settings = get_settings()
    encoded = quote(query)
    url = f"{settings.JINA_SEARCH_URL}/{encoded}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Accept": "application/json"}
            if settings.JINA_API_KEY:
                headers["Authorization"] = f"Bearer {settings.JINA_API_KEY}"
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])[:n_results]
    except Exception as exc:
        logger.warning("Jina search failed for query=%r: %s", query, exc)
        return []


async def _jina_fetch(url: str) -> str:
    """Fetch full page content via Jina Reader. Returns text (capped at 4000 chars)."""
    settings = get_settings()
    reader_url = f"{settings.JINA_READER_URL}/{url}"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {}
            if settings.JINA_API_KEY:
                headers["Authorization"] = f"Bearer {settings.JINA_API_KEY}"
            response = await client.get(reader_url, headers=headers)
            response.raise_for_status()
            return response.text[:4000]
    except Exception as exc:
        logger.warning("Jina reader failed for url=%r: %s", url, exc)
        return ""


async def _extract_materials(search_results: list[dict], fetched_contents: list[str]) -> list[dict]:
    """Use LLM to extract structured materials from combined search + fetch results."""
    content_blocks = []
    for i, result in enumerate(search_results):
        title = result.get("title", "")
        url = result.get("url", "")
        if i < len(fetched_contents) and fetched_contents[i]:
            content = fetched_contents[i][:2000]
        else:
            content = result.get("description", "") or result.get("content", "")
        content_blocks.append(f"### Source {i + 1}: {title}\nURL: {url}\n{content}")

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
    """Search for Chinese educational materials via Jina Search + Jina Reader.

    Returns list[{title, source, url, key_points}]. Returns [] on any error.
    """
    try:
        query = _build_query(subject, grade, description)
        search_results = await _jina_search(query, n_results=n_results)
        if not search_results:
            return []

        # Fetch full content for top 2 results in parallel
        top_urls = [r.get("url", "") for r in search_results[:2]]
        fetch_tasks = [_jina_fetch(url) for url in top_urls if url]
        fetched = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        fetched_contents: list[str] = []
        for item in fetched:
            if isinstance(item, Exception):
                fetched_contents.append("")
            else:
                fetched_contents.append(item)

        return await _extract_materials(search_results, fetched_contents)
    except Exception as exc:
        logger.warning("search_study_materials failed for subject=%r: %s", subject, exc)
        return []
