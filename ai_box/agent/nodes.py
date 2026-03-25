from langchain_core.messages import AIMessage, HumanMessage
from groq import Groq
from dotenv import load_dotenv
import os
import re
import sympy
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from datetime import datetime, timedelta

from agent.tools import web_search
from rag.vector_store import load_vector_store

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
vectorstore = load_vector_store()

# ─── Models ──────────────────────────────────────────────────────────────────
FAST_MODEL      = "llama-3.3-70b-versatile"   # General Q&A, search, RAG
REASONING_MODEL = "llama3-8b-8192"  # Math, coding, hard reasoning


# ─── Keyword triggers for routing ────────────────────────────────────────────
SEARCH_KEYWORDS = [
    "latest", "current", "today", "now", "live", "recent", "breaking",
    "right now", "this week", "this month", "this year",
    "score", "scores", "match", "ipl", "cricket", "football", "fifa",
    "nba", "nfl", "tennis", "f1", "formula 1", "tournament", "league",
    "championship", "winner", "won", "result", "standings", "points table",
    "2024", "2025", "2026",
    "news", "update", "announced", "launched", "released", "price",
    "weather", "stock", "market", "election", "president", "prime minister",
    "pm of", "ceo of", "founded", "net worth",
    "iphone", "samsung", "android", "windows", "gpt", "gemini",
    "openai", "google", "meta", "apple", "tesla", "nvidia", "spacex",
]

CODE_KEYWORDS = [
    "code", "program", "script", "function", "class", "algorithm",
    "python", "javascript", "java", "c++", "html", "css", "sql",
    "debug", "error", "fix", "bug", "implement", "write a",
    "how to build", "how to create", "develop", "api", "flask",
    "django", "react", "node", "fastapi",
]

MATH_KEYWORDS = [
    "calculate", "solve", "compute", "integral", "derivative",
    "equation", "algebra", "geometry", "trigonometry", "matrix",
    "probability", "statistics", "percent", "percentage", "square root",
    "factorial", "logarithm", "sin", "cos", "tan",
]

DATE_TRIGGERS = [
    "today's date", "what is today", "what's today", "what day is",
    "yesterday's date", "tomorrow's date", "what date",
]

RAG_KEYWORDS = [
    "world cup", "icc", "t20 world cup", "odi world cup",
    "women's cricket world cup", "cricket world cup",
]


def needs_search(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SEARCH_KEYWORDS)

def needs_code(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CODE_KEYWORDS)

def needs_math(text: str) -> bool:
    t = text.lower()
    has_keyword = any(kw in t for kw in MATH_KEYWORDS)
    has_expr = bool(re.search(r'\d+\s*[\+\-\*\/\%\^]\s*\d+', t))
    return has_keyword or has_expr

def needs_date(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in DATE_TRIGGERS)

def needs_rag(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in RAG_KEYWORDS)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def get_recent_context(messages, n=8):
    lines = []
    for m in messages[-n:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def summarize_if_long(messages):
    """Keep context window manageable for long conversations."""
    if len(messages) <= 14:
        return messages
    old = messages[:-10]
    recent = messages[-10:]
    history_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in old
    ])
    resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "user", "content":
            f"Summarize this conversation in 3-4 sentences, keeping all key facts:\n\n{history_text}"}],
        temperature=0, max_tokens=200
    )
    summary = resp.choices[0].message.content
    return [AIMessage(content=f"[Conversation summary: {summary}]")] + recent


def clean_deepseek_response(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from DeepSeek output."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


# ─── DATE ────────────────────────────────────────────────────────────────────
def date_node(state):
    user_msg = state["messages"][-1].content.lower()
    today = datetime.now()
    if "yesterday" in user_msg:
        result, label = today - timedelta(days=1), "Yesterday's date"
    elif "tomorrow" in user_msg:
        result, label = today + timedelta(days=1), "Tomorrow's date"
    else:
        result, label = today, "Today's date"
    answer = f"📅 {label} is **{result.strftime('%A, %d %B %Y')}**."
    return {"messages": state["messages"] + [AIMessage(content=answer)], "next": "end"}


# ─── PLANNER ─────────────────────────────────────────────────────────────────
def planner_node(state):
    query = state["messages"][-1].content

    # Fast keyword routing — no LLM call needed
    if needs_date(query):
        return {"messages": state["messages"], "next": "date"}
    if needs_search(query):
        return {"messages": state["messages"], "next": "search"}
    if needs_code(query):
        return {"messages": state["messages"], "next": "code"}
    if needs_math(query):
        return {"messages": state["messages"], "next": "math"}
    if needs_rag(query):
        return {"messages": state["messages"], "next": "rag"}

    # LLM fallback for ambiguous queries
    messages = summarize_if_long(state["messages"])
    context = get_recent_context(messages)

    prompt = f"""You are an AI router. Pick the best tool for the user's query.

Tools:
- search → live/current info, news, sports, prices, recent events, people
- code   → writing code, debugging, programming questions
- math   → calculations, equations, formulas, statistics
- rag    → internal cricket world cup knowledge base
- llm    → general knowledge, explanations, history, science, creative writing
- date   → only if asking for today/yesterday/tomorrow date

Conversation:
{context}

Reply with ONLY one word: search, code, math, rag, llm, or date"""

    resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=5
    )
    decision = resp.choices[0].message.content.strip().lower()
    for tool in ["search", "code", "math", "rag", "llm", "date"]:
        if tool in decision:
            return {"messages": state["messages"], "next": tool}

    return {"messages": state["messages"], "next": "llm"}


# ─── RAG ─────────────────────────────────────────────────────────────────────
def rag_node(state):
    query = state["messages"][-1].content
    docs = vectorstore.similarity_search(query, k=5)
    if not docs:
        answer = "I don't have that in my knowledge base. Try asking me to search the web."
    else:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Answer accurately using ONLY the context below.

Context:
{context}

Question: {query}

If the context doesn't clearly answer the question, say so honestly."""
        resp = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=500
        )
        answer = resp.choices[0].message.content
    return {"messages": state["messages"] + [AIMessage(content=answer)], "next": "end"}


# ─── SEARCH ──────────────────────────────────────────────────────────────────
def search_node(state):
    query = state["messages"][-1].content

    # Step 1: Refine query for better search results
    refine_prompt = f"""Convert this question into a short, precise search query (max 8 words).
Remove filler words. Keep entity + topic + year if relevant.

Examples:
"what's the latest cricket score today?" → "live cricket score today 2025"
"when will ipl 2026 start?" → "IPL 2026 schedule start date"
"who is PM of India right now?" → "India Prime Minister 2025"
"latest openai news" → "OpenAI news March 2025"

Question: {query}
Search query:"""

    refine_resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": refine_prompt}],
        temperature=0, max_tokens=25
    )
    search_query = refine_resp.choices[0].message.content.strip().strip('"').strip("'")

    # Step 2: Search
    results = web_search(search_query)
    if not results:
        results = web_search(query)  # fallback to original
    if not results:
        return llm_node(state)       # fallback to LLM if search fails

    # Step 3: Answer from results
    prompt = f"""You are a helpful, accurate AI assistant with access to live search results.

IMPORTANT RULES:
- Answer based on the search results provided
- Be specific — include names, numbers, dates, scores from the results
- Never say "I don't have real-time access" — you have the results right here
- If results clearly answer the question, give a direct confident answer
- If results are partial, give the best available answer and note what's uncertain
- Keep the answer clear and well-structured

Search Results:
{results}

User Question: {query}

Answer:"""

    resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=800
    )
    answer = resp.choices[0].message.content
    return {"messages": state["messages"] + [AIMessage(content=answer)], "next": "end"}


# ─── MATH ────────────────────────────────────────────────────────────────────
def math_node(state):
    query = state["messages"][-1].content

    # Use DeepSeek R1 for reasoning-heavy math
    prompt = f"""You are an expert mathematician. Solve this problem step by step.

Problem: {query}

Instructions:
- Show your working clearly
- Label each step
- Give the final answer on the last line, clearly marked
- If it's a word problem, identify the key values first"""

    resp = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=1500
    )
    answer = clean_deepseek_response(resp.choices[0].message.content)

    # Also try sympy for pure expressions
    try:
        clean_q = re.sub(r'[^0-9+\-*/().^a-zA-Z\s]', '', query)
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(clean_q, transformations=transformations)
        result = sympy.simplify(expr)
        answer += f"\n\n🧮 **Computed result: `{result}`**"
    except Exception:
        pass

    return {"messages": state["messages"] + [AIMessage(content=answer)], "next": "end"}


# ─── CODE ────────────────────────────────────────────────────────────────────
def code_node(state):
    messages = summarize_if_long(state["messages"])

    # Use DeepSeek R1 for code reasoning
    system = """You are an expert software engineer and coding assistant. 

When answering:
- Write clean, working, well-commented code
- Explain what the code does
- Point out edge cases or important considerations
- If debugging, explain the root cause clearly
- Format code in proper markdown code blocks with language tags"""

    formatted = [{"role": "system", "content": system}]
    for msg in messages[-12:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        formatted.append({"role": role, "content": msg.content})

    resp = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=formatted,
        temperature=0.2, max_tokens=2000
    )
    answer = clean_deepseek_response(resp.choices[0].message.content)
    return {"messages": state["messages"] + [AIMessage(content=answer)], "next": "end"}


# ─── GENERAL LLM ─────────────────────────────────────────────────────────────
def llm_node(state):
    messages = summarize_if_long(state["messages"])

    system = """You are a highly knowledgeable, accurate AI assistant — like ChatGPT.

Your goals:
- Give accurate, detailed, and helpful answers
- Be clear and well-structured
- For factual questions, be confident and specific
- For opinions or subjective topics, present balanced perspectives
- Never make up facts — if unsure, say so
- Match the user's tone (casual or formal)"""

    formatted = [{"role": "system", "content": system}]
    for msg in messages[-14:]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        formatted.append({"role": role, "content": msg.content})

    resp = client.chat.completions.create(
        model=FAST_MODEL,
        messages=formatted,
        temperature=0.4, max_tokens=1000
    )
    answer = resp.choices[0].message.content
    return {"messages": state["messages"] + [AIMessage(content=answer)], "next": "end"}