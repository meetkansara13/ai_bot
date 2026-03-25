from duckduckgo_search import DDGS


def web_search(query: str, max_results: int = 10) -> str:
    """Search the web using DuckDuckGo and return formatted results."""
    text = ""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for r in results:
                text += f"TITLE: {r['title']}\n"
                text += f"URL: {r.get('href', '')}\n"
                text += f"BODY: {r['body'][:700]}\n\n"
        return text if text else None
    except Exception:
        return None