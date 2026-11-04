import html
import time

import streamlit as st
from agents import build_reader_agent, build_search_agent, build_writer_chain, build_critic_chain
from customer_config import resolve_customer_profile
from resilience import run_stage
from schemas import ResearchRequest

st.set_page_config(
    page_title="AgenticResearch · Multi-Agent System",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AGENTS = [
    {"id": "search", "num": "01", "name": "Search", "role": "Retrieval", "icon": "⌕", "desc": "Web search via Tavily"},
    {"id": "reader", "num": "02", "name": "Reader", "role": "Extraction", "icon": "⎘", "desc": "Scrape & parse sources"},
    {"id": "writer", "num": "03", "name": "Writer", "role": "Synthesis", "icon": "✦", "desc": "Structured report draft"},
    {"id": "critic", "num": "04", "name": "Critic", "role": "Evaluation", "icon": "◎", "desc": "Quality review & score"},
]

EXAMPLES = ["LLM agents 2026", "CRISPR gene editing", "Fusion energy progress"]


def content_to_str(value) -> str:
    """Normalize LangChain/Gemini message content (str or list of blocks) to plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(getattr(block, "text", ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(value)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; height: 0; }
.block-container { padding: 0 2rem 3rem; max-width: 1280px; }

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; color: #e4e4ef; }

.stApp {
    background: #030308;
    background-image:
        linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px),
        radial-gradient(ellipse 70% 50% at 50% -20%, rgba(99,102,241,0.18), transparent 55%),
        radial-gradient(ellipse 50% 40% at 100% 50%, rgba(34,211,238,0.08), transparent 50%);
    background-size: 48px 48px, 48px 48px, auto, auto;
}

/* ── Top bar ── */
.sys-bar {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.75rem;
    padding: 0.85rem 0; margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(99,102,241,0.15);
}
.sys-brand { display: flex; align-items: center; gap: 0.65rem; }
.sys-logo {
    width: 32px; height: 32px; border-radius: 8px;
    background: linear-gradient(135deg, #6366f1, #22d3ee);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; color: #030308; font-weight: 700;
}
.sys-title { font-weight: 600; font-size: 0.95rem; letter-spacing: -0.02em; }
.sys-pills { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.pill {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.12em;
    text-transform: uppercase; padding: 0.35rem 0.65rem; border-radius: 999px;
    border: 1px solid rgba(99,102,241,0.25); background: rgba(99,102,241,0.08); color: #a5b4fc;
}
.pill.live { border-color: rgba(52,211,153,0.4); color: #6ee7b7; background: rgba(52,211,153,0.08); }
.pill.live::before { content: ''; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: #34d399; margin-right: 6px; animation: pulse-dot 1.5s ease infinite; vertical-align: middle; }
@keyframes pulse-dot { 0%,100%{opacity:1} 50%{opacity:0.35} }

/* ── Hero ── */
.hero-grid { display: grid; grid-template-columns: 1fr auto; gap: 2rem; align-items: center;
    padding: 2rem 0 1.5rem; }
@media (max-width: 900px) { .hero-grid { grid-template-columns: 1fr; } }
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.28em;
    text-transform: uppercase; color: #22d3ee; margin-bottom: 0.75rem;
}
.hero h1 {
    font-size: clamp(2.2rem, 5vw, 3.4rem); font-weight: 700; line-height: 1.05;
    letter-spacing: -0.04em; margin: 0 0 0.75rem; color: #f8f8ff;
}
.hero h1 em { font-style: normal; background: linear-gradient(90deg, #818cf8, #22d3ee);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero-sub { font-size: 1rem; color: #8888a0; max-width: 480px; line-height: 1.6; font-weight: 300; }

/* ── Agent graph (decorative) ── */
.agent-graph {
    width: 220px; height: 160px; position: relative; opacity: 0.9;
}
.graph-node {
    position: absolute; width: 44px; height: 44px; border-radius: 12px;
    border: 1px solid rgba(99,102,241,0.35); background: rgba(8,8,16,0.9);
    display: flex; align-items: center; justify-content: center;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.55rem; color: #a5b4fc;
    box-shadow: 0 0 24px rgba(99,102,241,0.15);
}
.graph-node.n1 { top: 0; left: 50%; transform: translateX(-50%); }
.graph-node.n2 { bottom: 20px; left: 0; }
.graph-node.n3 { bottom: 20px; right: 0; }
.graph-node.n4 { bottom: 0; left: 50%; transform: translateX(-50%); }
.graph-line {
    position: absolute; background: linear-gradient(90deg, transparent, rgba(34,211,238,0.4), transparent);
    height: 1px; transform-origin: left center;
}
.graph-hub {
    position: absolute; top: 42%; left: 50%; transform: translate(-50%,-50%);
    width: 56px; height: 56px; border-radius: 50%;
    border: 1px solid rgba(34,211,238,0.5);
    background: radial-gradient(circle, rgba(34,211,238,0.2), transparent 70%);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; animation: hub-glow 3s ease-in-out infinite;
}
@keyframes hub-glow { 0%,100%{ box-shadow: 0 0 20px rgba(34,211,238,0.2); } 50%{ box-shadow: 0 0 40px rgba(34,211,238,0.45); } }

/* ── Cards ── */
.glass {
    background: rgba(12,12,20,0.75); border: 1px solid rgba(99,102,241,0.12);
    border-radius: 16px; padding: 1.5rem 1.75rem; backdrop-filter: blur(12px);
}
.section-label {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #6366f1; margin-bottom: 1rem;
}

/* ── Agent cards ── */
.agent-card {
    display: flex; gap: 1rem; align-items: flex-start;
    padding: 1rem 1.15rem; margin-bottom: 0.65rem; border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06); background: rgba(255,255,255,0.02);
    transition: border-color 0.25s, box-shadow 0.25s, background 0.25s;
    position: relative; overflow: hidden;
}
.agent-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: rgba(255,255,255,0.06); border-radius: 14px 0 0 14px;
}
.agent-card.running {
    border-color: rgba(34,211,238,0.45); background: rgba(34,211,238,0.04);
    box-shadow: 0 0 32px rgba(34,211,238,0.08);
}
.agent-card.running::before { background: #22d3ee; animation: scan 1.2s ease infinite; }
.agent-card.done { border-color: rgba(52,211,153,0.35); background: rgba(52,211,153,0.03); }
.agent-card.done::before { background: #34d399; }
@keyframes scan { 0%,100%{opacity:1} 50%{opacity:0.4} }

.agent-icon {
    width: 40px; height: 40px; border-radius: 10px; flex-shrink: 0;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.2);
    display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
}
.agent-card.running .agent-icon { background: rgba(34,211,238,0.15); border-color: rgba(34,211,238,0.35); }
.agent-meta { flex: 1; min-width: 0; }
.agent-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.agent-name { font-weight: 600; font-size: 0.92rem; color: #f0f0f8; }
.agent-role {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; letter-spacing: 0.08em;
    padding: 0.15rem 0.45rem; border-radius: 4px; background: rgba(99,102,241,0.15); color: #a5b4fc;
}
.agent-desc { font-size: 0.78rem; color: #6b6b80; margin-top: 0.25rem; }
.agent-status {
    margin-left: auto; font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem; letter-spacing: 0.1em;
}
.st-waiting { color: #4a4a5c; }
.st-running { color: #22d3ee; }
.st-done { color: #34d399; }

/* ── Streamlit widgets ── */
.stTextInput > label {
    font-family: 'IBM Plex Mono', monospace !important; font-size: 0.68rem !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important; color: #818cf8 !important;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(99,102,241,0.25) !important;
    border-radius: 12px !important; color: #f0f0f8 !important; font-family: 'Outfit', sans-serif !important;
    padding: 0.85rem 1rem !important; font-size: 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #22d3ee !important; box-shadow: 0 0 0 3px rgba(34,211,238,0.12) !important;
}
.stButton > button[kind="primary"], .stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #0891b2 100%) !important;
    color: #fff !important; font-family: 'Outfit', sans-serif !important; font-weight: 600 !important;
    border: none !important; border-radius: 12px !important; padding: 0.75rem 1.5rem !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.35) !important; transition: transform 0.15s, box-shadow 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important; box-shadow: 0 8px 32px rgba(99,102,241,0.45) !important;
}
div[data-testid="stDownloadButton"] > button {
    background: transparent !important; border: 1px solid rgba(34,211,238,0.4) !important;
    color: #67e8f9 !important; box-shadow: none !important;
}
.stSpinner > div { color: #22d3ee !important; }

.chip-row { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.75rem; }
.chip-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: #5c5c70; letter-spacing: 0.12em; }

/* ── Results ── */
.result-block {
    background: rgba(8,8,14,0.8); border: 1px solid rgba(99,102,241,0.12);
    border-radius: 14px; padding: 1.5rem 1.75rem; margin-bottom: 1.25rem;
}
.result-block.report { border-color: rgba(34,211,238,0.25); }
.result-block.critic { border-color: rgba(52,211,153,0.25); }
.result-head {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; letter-spacing: 0.18em;
    text-transform: uppercase; padding-bottom: 0.75rem; margin-bottom: 1rem;
    border-bottom: 1px solid rgba(99,102,241,0.12);
}
.result-head.cyan { color: #22d3ee; border-color: rgba(34,211,238,0.15); }
.result-head.green { color: #34d399; border-color: rgba(52,211,153,0.15); }
.raw-pre {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; line-height: 1.65;
    color: #9898b0; white-space: pre-wrap; max-height: 280px; overflow-y: auto;
}

.activity-log {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: #6b6b85;
    padding: 0.75rem 1rem; border-radius: 10px; background: rgba(0,0,0,0.35);
    border: 1px solid rgba(99,102,241,0.1); margin-top: 1rem; line-height: 1.8;
}
.activity-log .line-active { color: #22d3ee; }

.footer-note {
    text-align: center; margin-top: 2.5rem; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.1em; color: #45455a;
}

[data-testid="stExpander"] { background: rgba(12,12,20,0.6); border: 1px solid rgba(99,102,241,0.1); border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def step_status(step_id: str) -> str:
    r = st.session_state.results
    if step_id in r:
        return "done"
    if st.session_state.running:
        order = [a["id"] for a in AGENTS]
        for aid in order:
            if aid not in r:
                return "running" if aid == step_id else "waiting"
    return "waiting"


def render_agent_card(agent: dict, state: str):
    labels = {"waiting": ("IDLE", "st-waiting"), "running": ("ACTIVE", "st-running"), "done": ("DONE", "st-done")}
    label, cls = labels[state]
    st.markdown(
        f"""
        <div class="agent-card {state}">
            <div class="agent-icon">{agent['icon']}</div>
            <div class="agent-meta">
                <div class="agent-row">
                    <span class="agent-name">{agent['num']} · {agent['name']}</span>
                    <span class="agent-role">{agent['role']}</span>
                    <span class="agent-status {cls}">{label}</span>
                </div>
                <div class="agent-desc">{agent['desc']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown("""
    <div class="sys-bar">
        <div class="sys-brand">
            <div class="sys-logo">◈</div>
            <span class="sys-title">AgenticResearch</span>
        </div>
        <div class="sys-pills">
            <span class="pill live">Orchestrator online</span>
            <span class="pill">4 agents</span>
            <span class="pill">LangGraph pipeline</span>
        </div>
    </div>
    <div class="hero-grid">
        <div>
            <div class="hero-eyebrow">Autonomous multi-agent system</div>
            <h1><em>Agentic</em> Research</h1>
            <p class="hero-sub">
                Orchestrated agents search the web, extract sources, synthesize reports,
                and critique output — end to end on any topic.
            </p>
        </div>
        <div class="agent-graph" aria-hidden="true">
            <div class="graph-hub">◈</div>
            <div class="graph-node n1">S</div>
            <div class="graph-node n2">R</div>
            <div class="graph-node n3">W</div>
            <div class="graph-node n4">C</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
for key in ("results", "running", "done", "log", "executions"):
    if key not in st.session_state:
        st.session_state[key] = {} if key in ("results", "executions") else ([] if key == "log" else False)


render_hero()

col_cmd, col_agents = st.columns([1.05, 0.95], gap="large")

with col_cmd:
    st.markdown('<div class="glass"><div class="section-label">Mission control</div>', unsafe_allow_html=True)

    if st.session_state.get("preset_topic"):
        st.session_state.topic_input = st.session_state.preset_topic
        del st.session_state.preset_topic

    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Agentic AI frameworks in 2026",
        key="topic_input",
        label_visibility="visible",
    )
    customer_id = st.text_input(
        "Customer profile",
        value="default",
        help="Enter a saved profile ID or paste a short customer brief (company, industry, objective, audience).",
        key="customer_id_input",
    )
    research_questions_text = st.text_area(
        "Research questions (optional)",
        placeholder="Enter one question per line",
        key="research_questions_input",
        height=100,
    )
    deliverables_text = st.text_input(
        "Report sections (optional)",
        placeholder="Comma-separated, e.g. Executive summary, Market trends, Risks, Sources",
        key="deliverables_input",
    )
    run_btn = st.button("Deploy agent pipeline", type="primary", use_container_width=True)

    st.markdown('<div class="chip-row"><span class="chip-label">QUICK RUN →</span>', unsafe_allow_html=True)
    chip_cols = st.columns(len(EXAMPLES))
    for i, ex in enumerate(EXAMPLES):
        if chip_cols[i].button(ex, key=f"chip_{i}", use_container_width=True):
            st.session_state.preset_topic = ex
            st.session_state.results = {}
            st.session_state.executions = {}
            st.session_state.running = True
            st.session_state.done = False
            st.session_state.log = [f"▸ Queued: {ex}"]
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.log:
        log_html = "".join(
            f'<div class="{"line-active" if i == len(st.session_state.log) - 1 and st.session_state.running else ""}">{html.escape(line)}</div>'
            for i, line in enumerate(st.session_state.log)
        )
        st.markdown(f'<div class="activity-log">{log_html}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_agents:
    st.markdown('<div class="glass"><div class="section-label">Agent fleet</div>', unsafe_allow_html=True)
    for agent in AGENTS:
        render_agent_card(agent, step_status(agent["id"]))
    st.markdown("</div>", unsafe_allow_html=True)


def append_log(msg: str):
    st.session_state.log = [*st.session_state.get("log", []), msg]


# ── Start pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    try:
        customer_questions = [q.strip() for q in research_questions_text.splitlines() if q.strip()]
        requested_sections = [section.strip() for section in deliverables_text.split(",") if section.strip()]
        resolved_config, profile_context = resolve_customer_profile(customer_id)
        ResearchRequest.model_validate({
            "topic": topic.strip(),
            "customer_id": resolved_config.customer_id,
            "customer_context": profile_context,
            "research_questions": customer_questions,
            "deliverables": requested_sections,
        })
        st.session_state.results = {}
        st.session_state.executions = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.log = [f"▸ Pipeline started: {topic.strip()}"]
        st.rerun()
    except Exception as exc:
        st.warning(f"Check the research request: {exc}")


# ── Stepwise execution (live UI updates) ─────────────────────────────────────
if st.session_state.running and not st.session_state.done:
    results = dict(st.session_state.results)
    topic_val = st.session_state.topic_input
    try:
        customer_config, profile_context = resolve_customer_profile(st.session_state.get("customer_id_input", ""))
    except Exception as exc:
        st.error(f"Customer configuration error: {exc}")
        st.session_state.running = False
        st.session_state.done = True
        st.stop()

    if "search" not in results:
        append_log("▸ Search agent: querying Tavily…")
        try:
            with st.spinner("Search agent active…"):
                search_agent = build_search_agent(customer_config)
                question_context = "\n".join(st.session_state.get("research_questions_input", "").splitlines())
                sr = run_stage("search", lambda: search_agent.invoke({
                    "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}\n"
                                                f"Customer context:\n{profile_context}\n"
                                                f"Research questions:\n{question_context}")]
                }), executions=st.session_state.executions, max_attempts=customer_config.retry_attempts)
                results["search"] = content_to_str(sr["messages"][-1].content)
        except Exception as exc:
            results["search"] = f"ERROR [search_unavailable]: {exc}. Continue with available information and disclose the limitation."
            st.session_state.executions.setdefault("search", {"agent": "search", "status": "failed", "error": {"code": "stage_failed", "message": str(exc)}})
        append_log("✓ Search agent: results collected")
        st.session_state.results = results
        st.rerun()

    elif "reader" not in results:
        append_log("▸ Reader agent: scraping sources…")
        try:
            with st.spinner("Reader agent active…"):
                reader_agent = build_reader_agent(customer_config)
                rr = run_stage("reader", lambda: reader_agent.invoke({
                    "messages": [("user",
                        f"Based on the following search results about '{topic_val}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Customer context:\n{profile_context}\n"
                        f"Research questions:\n{st.session_state.get('research_questions_input', '')}\n\n"
                        f"Search Results:\n{content_to_str(results['search'])[:800]}",
                    )]
                }), executions=st.session_state.executions, max_attempts=customer_config.retry_attempts)
                results["reader"] = content_to_str(rr["messages"][-1].content)
        except Exception as exc:
            results["reader"] = f"ERROR [reader_unavailable]: {exc}. Use search snippets; disclose that source pages could not be accessed."
            st.session_state.executions.setdefault("reader", {"agent": "reader", "status": "failed", "error": {"code": "stage_failed", "message": str(exc)}})
        append_log("✓ Reader agent: content extracted")
        st.session_state.results = results
        st.rerun()

    elif "writer" not in results:
        append_log("▸ Writer chain: drafting report…")
        try:
            with st.spinner("Writer chain active…"):
                research_combined = (
                    f"SEARCH RESULTS:\n{content_to_str(results['search'])}\n\n"
                    f"DETAILED SCRAPED CONTENT:\n{content_to_str(results['reader'])}"
                )
                research_questions = [q.strip() for q in st.session_state.get("research_questions_input", "").splitlines() if q.strip()]
                deliverables = [section.strip() for section in st.session_state.get("deliverables_input", "").split(",") if section.strip()]
                writer = build_writer_chain(customer_config, research_questions, deliverables, profile_context)
                results["writer"] = content_to_str(run_stage("writer", lambda: writer.invoke({"topic": topic_val, "research": research_combined}), executions=st.session_state.executions, max_attempts=customer_config.retry_attempts))
        except Exception as exc:
            results["writer"] = f"ERROR [writer_failed]: {exc}"
            st.session_state.executions.setdefault("writer", {"agent": "writer", "status": "failed", "error": {"code": "stage_failed", "message": str(exc)}})
        append_log("✓ Writer chain: report ready")
        st.session_state.results = results
        st.rerun()

    elif "critic" not in results:
        append_log("▸ Critic chain: evaluating quality…")
        try:
            with st.spinner("Critic chain active…"):
                results["critic"] = content_to_str(run_stage("critic", lambda: build_critic_chain(customer_config).invoke({"report": results["writer"]}), executions=st.session_state.executions, max_attempts=customer_config.retry_attempts))
        except Exception as exc:
            results["critic"] = f"ERROR [critic_unavailable]: {exc}"
            st.session_state.executions.setdefault("critic", {"agent": "critic", "status": "failed", "error": {"code": "stage_failed", "message": str(exc)}})
        append_log("✓ Critic chain: review complete")
        st.session_state.results = results
        st.session_state.running = False
        st.session_state.done = True
        st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
r = st.session_state.results

if st.session_state.executions:
    with st.expander("Execution details", expanded=False):
        for name, execution in st.session_state.executions.items():
            status = execution.get("status", "unknown").upper()
            duration = execution.get("duration_seconds")
            attempts = execution.get("attempts", 0)
            summary = f"{name.title()}: {status} · {attempts} attempt(s)"
            if duration is not None:
                summary += f" · {duration}s"
            st.write(summary)
            if execution.get("error"):
                st.json(execution["error"])

if r:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="margin-bottom:1rem;">Output artifacts</div>', unsafe_allow_html=True)

    if "search" in r:
        with st.expander("Search agent · raw output", expanded=False):
            st.markdown(
                f'<div class="result-block"><pre class="raw-pre">{html.escape(content_to_str(r["search"]))}</pre></div>',
                unsafe_allow_html=True,
            )

    if "reader" in r:
        with st.expander("Reader agent · raw output", expanded=False):
            st.markdown(
                f'<div class="result-block"><pre class="raw-pre">{html.escape(content_to_str(r["reader"]))}</pre></div>',
                unsafe_allow_html=True,
            )

    if "writer" in r:
        st.markdown(
            '<div class="result-block report"><div class="result-head cyan">Final research report</div>',
            unsafe_allow_html=True,
        )
        report_text = content_to_str(r["writer"])
        st.markdown(report_text)
        st.markdown("</div>", unsafe_allow_html=True)
        st.download_button(
            label="Download report (.md)",
            data=report_text,
            file_name=f"agentic_research_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in r:
        st.markdown(
            '<div class="result-block critic"><div class="result-head green">Critic evaluation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(content_to_str(r["critic"]))
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-note">AgenticResearch · Multi-agent LangChain pipeline</div>',
    unsafe_allow_html=True,
)
