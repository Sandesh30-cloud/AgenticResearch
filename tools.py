from datetime import date
from datetime import datetime
from urllib.parse import urlparse
import os

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.tools import StructuredTool
from tavily import TavilyClient

from resilience import rate_limit_search, run_stage

load_dotenv()


def _valid_http_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not any(c.isspace() for c in url)
    except (AttributeError, ValueError):
        return False


def _publication_date(soup: BeautifulSoup) -> str | None:
    candidates = []
    for key in ("article:published_time", "datePublished", "pubdate", "publishdate", "date"):
        for attr in ("property", "name", "itemprop"):
            node = soup.find("meta", attrs={attr: key})
            if node and node.get("content"):
                candidates.append(node["content"].strip())
    for time_node in soup.find_all("time", limit=5):
        value = time_node.get("datetime") or time_node.get_text(" ", strip=True)
        if value:
            candidates.append(value.strip())
    for value in candidates:
        try:
            parsed = date.fromisoformat(value[:10])
            return parsed.isoformat()
        except (ValueError, TypeError):
            continue
    return None


def _validated_date(value) -> str:
    if not value:
        return "Not provided"
    raw = str(value).strip()
    for fmt in (None, "%Y-%m-%d", "%Y/%m/%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00")).date() if fmt is None else datetime.strptime(raw[:10] if fmt == "%Y-%m-%d" else raw, fmt).date()
            if parsed <= date.today():
                return parsed.isoformat()
        except ValueError:
            continue
    return "Unavailable or invalid"


class _TimeoutSession(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", float(os.getenv("SEARCH_TIMEOUT_SECONDS", "15")))
        return super().request(method, url, **kwargs)


def _search(query: str, *, max_results: int, retry_attempts: int, calls_per_minute: int) -> str:
    def search():
        rate_limit_search(calls_per_minute)
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"), session=_TimeoutSession())
        return client.search(query=query, max_results=max_results)

    try:
        results = run_stage("web_search", search, max_attempts=retry_attempts)
    except Exception as exc:
        return f"ERROR [search_unavailable]: Search could not be completed after retries: {exc}. Use other available sources and clearly disclose that search results are unavailable."

    out = []
    for result in results.get("results", []):
        url = str(result.get("url", "")).strip()
        if not _valid_http_url(url):
            continue
        published = _validated_date(result.get("published_date") or result.get("published_at"))
        out.append(f"Title: {result.get('title', 'Untitled')}\nURL: {url}\nPublication date: {published}\nSnippet: {str(result.get('content', ''))[:300]}\n")
    return "\n----\n".join(out) if out else "No sources with valid HTTP(S) URLs were found. Continue using available snippets and disclose the source limitation."


@tool
def web_search(query: str) -> str:
    """Search Tavily and return validated source URLs, snippets, and publication dates when available."""
    return _search(
        query,
        max_results=5,
        retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
        calls_per_minute=int(os.getenv("SEARCH_RATE_LIMIT_PER_MINUTE", "20")),
    )


def customer_search_tool(customer_id: str, *, max_results: int, retry_attempts: int,
                         calls_per_minute: int) -> StructuredTool:
    def search_for_customer(query: str) -> str:
        return _search(query, max_results=max_results, retry_attempts=retry_attempts,
                       calls_per_minute=calls_per_minute)
    return StructuredTool.from_function(
        func=search_for_customer,
        name=f"web_search_{customer_id}",
        description="Search for recent, reliable sources and return validated URLs, dates, and snippets.",
    )


@tool
def scrape_url(url: str) -> str:
    """Fetch a source URL with timeout and retries; return text plus validated publication date."""
    if not _valid_http_url(url):
        return "ERROR [invalid_source_url]: URL must be a valid HTTP(S) URL. Select another source from search results."

    def fetch():
        response = requests.get(url, timeout=float(os.getenv("SOURCE_TIMEOUT_SECONDS", "10")),
                                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})
        response.raise_for_status()
        return response

    try:
        response = run_stage("scrape_url", fetch, max_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")))
        soup = BeautifulSoup(response.text, "html.parser")
        published = _publication_date(soup)
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)[:3000]
        date_line = f"Publication date: {published}" if published else "Publication date: unavailable or unverified"
        if not text:
            return f"ERROR [empty_source]: The page returned no readable text. {date_line} Try another source."
        return f"Source URL: {response.url}\n{date_line}\n\n{text}"
    except Exception as exc:
        return f"ERROR [source_unavailable]: Could not access {url} after retries ({type(exc).__name__}: {exc}). Use the search snippet or try another source; do not treat the failed fetch as verified evidence."
