"""Web research service: search for Chinese curriculum materials.

Provider strategy:
  1. Tavily (primary) — richer content field, better RAG quality
  2. Brave Search (fallback) — used when Tavily quota is exhausted (HTTP 429)
     or when TAVILY_API_KEY is not set

Both providers return a normalised list[{title, url, content}] consumed by
_extract_materials(). Callers see no difference.
"""

from __future__ import annotations

import json
import logging
from urllib.parse import urlencode, urlparse

import httpx

from app.config import get_settings
from app.services import llm_service

logger = logging.getLogger(__name__)

_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

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


async def _tavily_search(query: str, n_results: int = 5) -> list[dict]:
    """Search via Tavily API.

    Returns normalised list[{title, url, content}].
    Raises httpx.HTTPStatusError on HTTP errors (caller checks status_code for 429).
    Raises other exceptions on network / parse failures.
    """
    settings = get_settings()
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.post(
            _TAVILY_ENDPOINT,
            json={
                "api_key": settings.TAVILY_API_KEY,
                "query": query,
                "max_results": n_results,
                "include_answer": False,
            },
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        results = response.json().get("results", [])[:n_results]
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results
        ]


async def _brave_search(query: str, n_results: int = 5) -> list[dict]:
    """Search via Brave Search API.

    Returns normalised list[{title, url, content}]. Returns [] on any error.
    """
    settings = get_settings()
    if not settings.BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not set; cannot fall back to Brave")
        return []
    params = urlencode({"q": query, "count": n_results, "search_lang": "zh", "country": "CN"})
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{_BRAVE_ENDPOINT}?{params}",
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": settings.BRAVE_API_KEY,
                },
            )
            response.raise_for_status()
            results = response.json().get("web", {}).get("results", [])[:n_results]
            return [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("description", ""),  # Brave returns snippets, not full content
                }
                for r in results
            ]
    except Exception as exc:
        logger.warning("Brave search failed for query=%r: %s", query, exc)
        return []


async def _search_with_fallback(query: str, n_results: int = 5) -> list[dict]:
    """Try Tavily; fall back to Brave on quota (429) or any failure."""
    settings = get_settings()

    if settings.TAVILY_API_KEY:
        try:
            results = await _tavily_search(query, n_results)
            logger.debug("Tavily search returned %d results", len(results))
            return results
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                logger.warning(
                    "Tavily quota exhausted (HTTP 429); falling back to Brave Search"
                )
            else:
                logger.warning(
                    "Tavily search HTTP error %d for query=%r; falling back to Brave Search",
                    exc.response.status_code,
                    query,
                )
        except Exception as exc:
            logger.warning(
                "Tavily search failed for query=%r: %s; falling back to Brave Search",
                query,
                exc,
            )
    else:
        logger.debug("TAVILY_API_KEY not set; using Brave Search directly")

    return await _brave_search(query, n_results)


async def _extract_materials(search_results: list[dict]) -> list[dict]:
    """Use LLM to extract structured educational materials from search results."""
    content_blocks = []
    for i, result in enumerate(search_results):
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
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
    """Search for Chinese educational materials (Tavily → Brave fallback).

    Returns list[{title, source, url, key_points}]. Returns [] on any error.
    """
    try:
        query = _build_query(subject, grade, description)
        search_results = await _search_with_fallback(query, n_results=n_results)
        if not search_results:
            return []
        return await _extract_materials(search_results)
    except Exception as exc:
        logger.warning("search_study_materials failed for subject=%r: %s", subject, exc)
        return []
