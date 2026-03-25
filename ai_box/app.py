import streamlit as st
from agent.graph import graph
from langchain_core.messages import HumanMessage, AIMessage
import datetime
import os
import re
from io import BytesIO
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Agent Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .tool-badge {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: bold;
        margin-bottom: 6px;
    }
    .badge-search { background: #1a73e8; color: white; }
    .badge-rag    { background: #34a853; color: white; }
    .badge-math   { background: #fbbc04; color: black; }
    .badge-llm    { background: #ea4335; color: white; }
    .badge-date   { background: #9334e6; color: white; }
    .badge-code   { background: #ff6d00; color: white; }
</style>
""", unsafe_allow_html=True)

# ─── Groq client for Whisper ──────────────────────────────────────────────────
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ─── Session State ────────────────────────────────────────────────────────────
for key, default in [
    ("messages", []),
    ("show_tool", True),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Whisper Transcription ────────────────────────────────────────────────────
def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    try:
        transcription = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=(filename, audio_bytes, "audio/wav"),
            language="en",
            response_format="text"
        )
        return transcription.strip()
    except Exception as e:
        return f"[Transcription error: {e}]"

# ─── Tool labels & badge map ──────────────────────────────────────────────────
TOOL_LABELS = {
    "search": "🌐 Web Search",
    "rag":    "📚 Knowledge Base",
    "math":   "🧮 Math · DeepSeek R1",
    "code":   "💻 Code · DeepSeek R1",
    "llm":    "🧠 LLaMA 3.3 70B",
    "date":   "📅 Date",
}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    st.subheader("🎙️ Voice Input")
    stt_enabled = st.toggle("Whisper large-v3 STT", value=False)
    if stt_enabled:
        st.caption("Record your question — Groq Whisper transcribes it instantly")

    st.divider()
    st.subheader("🧠 Display")
    st.session_state.show_tool = st.toggle("Show which tool answered", value=True)

    st.divider()
    st.subheader("📋 Chat")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.messages:
        export_text = "\n\n".join([
            f"{'You' if m['role'] == 'user' else 'AI'}: {m['content']}"
            for m in st.session_state.messages
        ])
        st.download_button(
            "💾 Export Chat (.txt)",
            data=export_text,
            file_name=f"chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

    st.divider()
    st.subheader("💡 Try asking:")
    examples = [
        "Latest cricket score today",
        "When does IPL 2026 start?",
        "Who won the 2024 US election?",
        "Write a Python Flask REST API",
        "Solve: integral of x^2 from 0 to 3",
        "Explain how transformers work in AI",
        "What is 18% of 4500?",
        "Latest AI news today",
        "What is today's date?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state["prefill"] = ex

    st.divider()
    st.caption("🤖 LLaMA 3.3 70B · DeepSeek R1 · Whisper large-v3")
    st.caption("⚡ All models free via Groq API")

# ─── Main Header ──────────────────────────────────────────────────────────────
st.title("🤖 Universal AI Agent Pro")
st.caption("LangGraph · RAG · Live Search · Whisper STT · LLaMA 3.3 70B · DeepSeek R1")

# ─── Whisper Mic Input ────────────────────────────────────────────────────────
whisper_transcript = ""
if stt_enabled:
    st.markdown("### 🎙️ Voice Input")
    audio_input = st.audio_input("🎙️ Click the mic, speak your question, click stop")
    if audio_input is not None:
        with st.spinner("⏳ Transcribing with Whisper large-v3..."):
            audio_bytes = audio_input.read()
            whisper_transcript = transcribe_audio(
                audio_bytes, audio_input.name or "audio.wav"
            )
        if not whisper_transcript.startswith("[Transcription error"):
            st.success(f"📝 **You said:** {whisper_transcript}")
        else:
            st.error(whisper_transcript)
            whisper_transcript = ""
    st.divider()

# ─── Chat History ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and st.session_state.show_tool and msg.get("tool"):
            tool = msg["tool"]
            label = TOOL_LABELS.get(tool, tool.upper())
            st.markdown(
                f'<span class="tool-badge badge-{tool}">{label}</span>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])

# ─── Input ────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill", "")
typed_prompt = st.chat_input("Ask me anything...")

# Priority: typed input > whisper transcript > sidebar button
final_prompt = typed_prompt or whisper_transcript or prefill

if final_prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": final_prompt})
    with st.chat_message("user"):
        st.markdown(final_prompt)

    # Build LangChain message history
    langchain_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else AIMessage(content=m["content"])
        for m in st.session_state.messages
    ]

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = graph.invoke({
                "messages": langchain_messages,
                "next": "planner"
            })

        answer = result["messages"][-1].content

        # Detect tool used: find last state where "next" was a tool name
        tool_used = "llm"
        for msg in reversed(result["messages"]):
            pass  # tool tracking via graph state
        # Simple detection from answer patterns
        if "```" in answer:
            tool_used = "code"
        elif "🧮" in answer or "Computed" in answer:
            tool_used = "math"
        elif "📅" in answer:
            tool_used = "date"

        if st.session_state.show_tool:
            label = TOOL_LABELS.get(tool_used, "🧠 LLM")
            st.markdown(
                f'<span class="tool-badge badge-{tool_used}">{label}</span>',
                unsafe_allow_html=True
            )

        st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool": tool_used,
    })