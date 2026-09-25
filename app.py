# -*- coding: utf-8 -*-
"""PUBG Policy Assistant — frontend/UI shell (Streamlit).

STRICTLY UI ONLY. No retrieval, embeddings, BM25, RRF, LLM calls,
evaluation, or backend RAG logic lives here.

Future backend connection point (single place)::

    from src.ui_backend import ask_question  # -> handle_query
    answer = ask_question(user_query)        # TODO: wire to RAG later

Replacing ``src/ui_backend.ask_question`` with a call to
``src.task10_generation.generate_with_citation`` requires no page redesign.

Run::

    streamlit run app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.ui_backend import filter_sources, handle_query
from src.ui_components import (
    EXAMPLE_PROMPTS,
    assistant_message_html,
    portrait_html,
    retrieval_info_html,
    source_card_html,
    user_message_html,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TANVU_IMG = BASE_DIR / "assets" / "tanvu.jpeg"
HIMASS_IMG = BASE_DIR / "assets" / "himass.jpeg"

st.set_page_config(
    page_title="PUBG Policy Assistant — UI Preview",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "source_filter" not in st.session_state:
    st.session_state.source_filter = "all"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "is_loading" not in st.session_state:
    st.session_state.is_loading = False

# ---------------------------------------------------------------------------
# Global dark PUBG/esports theme
# ---------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
:root{
  --bg:#05070a; --panel:#0d1116; --panel2:#12181f;
  --line:rgba(255,255,255,.09); --line-soft:rgba(255,255,255,.06);
  --text:#edf0f3; --muted:#9aa3ad; --faint:#6b7480;
  --gold:#e2b34c; --gold-soft:rgba(226,179,76,.13);
}
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
  background:var(--bg) !important; color:var(--text) !important;
}
[data-testid="stAppViewContainer"]{overflow-x:hidden;}
[data-testid="stHeader"]{background:rgba(5,7,10,.85) !important;
  backdrop-filter:blur(10px); border-bottom:1px solid var(--line-soft);}
.main .block-container{max-width:1560px !important; padding:0 28px 40px !important;}
*{scrollbar-width:thin; scrollbar-color:#2a323b transparent;}
button{font-family:inherit !important;}
button:focus-visible, textarea:focus-visible, input:focus-visible{
  outline:2px solid var(--gold) !important; outline-offset:2px;}
.stButton>button, .stFormSubmitButton>button{min-height:40px; border-radius:999px !important;
  border:1px solid var(--line) !important; background:rgba(255,255,255,.03) !important;
  color:var(--text) !important; font-size:13.5px !important; line-height:1.5 !important;
  padding:8px 16px !important; transition:border-color .18s, background .18s;}
.stButton>button:hover, .stFormSubmitButton>button:hover{
  border-color:rgba(226,179,76,.55) !important; background:rgba(226,179,76,.08) !important;}
.stButton>button:disabled, .stFormSubmitButton>button:disabled{opacity:.55; cursor:wait;}
.stFormSubmitButton>button[kind="primary"]{background:var(--gold-soft) !important;
  border-color:rgba(226,179,76,.55) !important; color:var(--gold) !important; font-weight:800;}
.stTextArea textarea{background:rgba(11,15,20,.85) !important; color:var(--text) !important;
  border:1px solid var(--line) !important; border-radius:18px !important;
  font-size:16px !important; line-height:1.55 !important; backdrop-filter:blur(8px);}
.stTextArea textarea:focus{border-color:rgba(226,179,76,.6) !important;
  box-shadow:0 0 0 1px rgba(226,179,76,.35) !important;}
.stTextArea textarea::placeholder{color:var(--faint) !important;}

/* Header */
.pubg-header{display:flex; align-items:flex-start; justify-content:space-between;
  gap:16px; padding:14px 4px 12px; border-bottom:1px solid var(--line-soft);
  margin-bottom:14px; flex-wrap:wrap;}
.pubg-kicker{font-size:10.5px; letter-spacing:.28em; color:var(--gold);
  font-weight:700; margin-bottom:4px;}
.pubg-title{font-size:24px; font-weight:800; letter-spacing:.01em; margin:0; color:#fff;}
.pubg-sub{color:var(--muted); font-size:13.5px; margin:4px 0 0; max-width:640px;}
.pubg-badges{display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding-top:6px;}
.badge{font-size:12px; font-weight:700; letter-spacing:.08em; padding:7px 12px;
  border-radius:999px; border:1px solid var(--line); color:var(--muted);
  background:rgba(255,255,255,.03);}
.badge-preview{color:var(--gold); border-color:rgba(226,179,76,.5);
  background:var(--gold-soft);}

/* Layout helpers */
.center-wrap{max-width:780px; margin:0 auto; width:100%;}
.chat-shell{background:transparent; border:none; box-shadow:none; overflow:visible;}
.chat-scroll{padding:8px 6px;}
.composer-zone{position:sticky; bottom:10px; z-index:6; margin-top:10px;
  padding:12px; background:rgba(10,13,17,.82); backdrop-filter:blur(12px);
  border:1px solid var(--line); border-radius:22px;
  box-shadow:0 18px 50px rgba(0,0,0,.5);}

/* Empty state */
.empty-hero{text-align:center; padding:30px 20px 6px;}
.empty-eyebrow{font-size:10.5px; letter-spacing:.3em; color:var(--faint); font-weight:700;}
.empty-hero h2{font-size:24px; margin:8px 0 6px; color:#fff; font-weight:800;}
.empty-hero p{color:var(--muted); font-size:14px; margin:0 auto; max-width:560px;}
.example-grid{display:grid; grid-template-columns:1fr 1fr; gap:10px; padding:18px 22px 6px;}
.example-card{text-align:left; width:100%; border:1px solid var(--line) !important;
  background:rgba(255,255,255,.025) !important; border-radius:14px !important;
  padding:13px 14px !important; font-size:14px !important; line-height:1.5 !important;
  min-height:64px;}
.demo-note{text-align:center; color:var(--faint); font-size:12px; padding:2px 20px 16px;}

/* Messages */
.msg{display:flex; gap:12px; margin:16px 0; align-items:flex-start;}
.msg-avatar{flex:0 0 34px; width:34px; height:34px; border-radius:50%;
  display:flex; align-items:center; justify-content:center;
  font-size:13px; font-weight:800;}
.msg-avatar-user{background:var(--gold-soft); border:1px solid rgba(226,179,76,.5); color:var(--gold);}
.msg-avatar-bot{background:#1a222b; border:1px solid var(--line); color:#cfd6dd;}
.msg-bubble{border-radius:14px; padding:12px 15px; font-size:15.5px; line-height:1.65;
  max-width:100%; word-wrap:break-word; overflow-wrap:anywhere;}
.msg-user{justify-content:flex-end;}
.msg-bubble-user{background:#1a2129; border:1px solid rgba(226,179,76,.28); color:#f2f4f6;}
.msg-assistant-card{background:var(--panel2); border:1px solid var(--line);
  border-radius:16px; padding:16px 18px; margin:6px 0 4px;}
.msg-assistant-card .stMarkdown p, .assistant-md{font-size:15.5px; line-height:1.7; color:#e6eaee;}
.assistant-head{display:flex; align-items:center; gap:10px; margin-bottom:8px;}
.assistant-name{font-size:13px; font-weight:800; letter-spacing:.12em; color:#d6dbe1;}
.mock-tag{font-size:11px; font-weight:700; color:var(--gold);
  border:1px solid rgba(226,179,76,.45); background:var(--gold-soft);
  padding:3px 9px; border-radius:999px; letter-spacing:.06em;}
.assistant-footer{margin-top:12px;}

/* Retrieval + sources */
.retrieval-row{display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin:4px 0 12px;}
.ret-label{font-size:11px; letter-spacing:.2em; color:var(--faint); font-weight:700;}
.ret-pill{font-size:12px; font-weight:700; color:#cfd6dd; background:#1a222b;
  border:1px solid var(--line); border-radius:999px; padding:5px 11px;}
.ret-demo{color:var(--gold); border-color:rgba(226,179,76,.4); background:var(--gold-soft);}
.src-heading{font-size:12px; letter-spacing:.22em; color:var(--muted);
  font-weight:800; margin:6px 0 10px;}
.src-list{display:flex; flex-direction:column; gap:10px;}
.src-card{display:flex; gap:12px; background:rgba(255,255,255,.02);
  border:1px solid var(--line); border-radius:14px; padding:13px 14px;}
.src-index{color:var(--gold); font-weight:800; font-size:14px; flex:0 0 auto;}
.src-title{font-weight:700; font-size:14.5px; color:#fff;}
.src-pub{color:var(--muted); font-size:13px; margin-top:2px;}
.src-meta{display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-top:8px;}
.src-badge{font-size:11px; font-weight:800; letter-spacing:.05em;
  padding:4px 10px; border-radius:999px; border:1px solid var(--line);}
.src-official{color:var(--gold); border-color:rgba(226,179,76,.5); background:var(--gold-soft);}
.src-independent{color:#b9c1c9; background:rgba(255,255,255,.04);}
.src-extra{font-size:11.5px; color:var(--faint);}
.src-snippet{font-size:13px; color:var(--muted); margin-top:8px; line-height:1.6;}
.src-link{margin-top:8px;}
.src-url{font-size:12.5px; color:#8fb8dd; word-break:break-all; overflow-wrap:anywhere;}
.src-no-url{color:var(--faint); font-size:12.5px;}
.src-empty{color:var(--faint); font-size:13.5px; padding:6px 0;}

/* Portraits — players emerge from the dark, no hard rectangle */
.portrait-col{display:flex; flex-direction:column; align-items:center;
  padding-top:6px; min-width:0;}
.portrait-frame{position:relative; width:100%; max-width:340px;}
.portrait-img{display:block; width:100%; height:auto; max-height:74vh;
  object-fit:contain; aspect-ratio:auto; filter:grayscale(1) contrast(1.06) brightness(.97);
  -webkit-mask-image:radial-gradient(ellipse 88% 86% at 50% 40%, #000 60%, transparent 98%);
  mask-image:radial-gradient(ellipse 88% 86% at 50% 40%, #000 60%, transparent 98%);}
.portrait-fade{position:absolute; inset:auto 0 0 0; height:120px; pointer-events:none;
  background:linear-gradient(180deg, transparent, var(--bg) 92%);}
.portrait-fade-inner{position:absolute; inset:0; pointer-events:none;}
.portrait-fade-inner-left{background:linear-gradient(to left, var(--bg) 0%, transparent 26%);}
.portrait-fade-inner-right{background:linear-gradient(to right, var(--bg) 0%, transparent 26%);}
.portrait-label{margin-top:10px; font-size:15px; font-weight:800; letter-spacing:.34em;
  color:#e8ebee; text-indent:.34em;}
.portrait-sub{font-size:12px; letter-spacing:.18em; color:var(--faint); margin-top:4px;}
.portrait-missing{font-size:11.5px; color:var(--faint); margin-top:8px; text-align:center;
  max-width:240px; line-height:1.5;}

/* Filter + footer */
.filter-row{display:flex; align-items:center; gap:8px; padding:2px 6px 0; flex-wrap:wrap;}
.filter-label{font-size:11px; letter-spacing:.16em; color:var(--faint); font-weight:800;}
.pubg-footer{display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;
  color:var(--faint); font-size:12px; padding:16px 6px 0;}
.loading-dots{color:var(--gold); font-weight:700;}

/* Responsive: portraits support, never cover chat */
@media (max-width:1180px){ .portrait-frame{max-width:270px;} .pubg-title{font-size:26px;} }
@media (max-width:950px){
  div[data-testid="column"]:nth-of-type(1),
  div[data-testid="column"]:nth-of-type(3){display:none !important;}
  .main .block-container{padding:0 14px 30px !important;}
  .example-grid{grid-template-columns:1fr;}
  .chat-scroll{padding:16px 14px 4px;}
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Query processing (single backend boundary)
# ---------------------------------------------------------------------------
def _process_query(raw_query: str) -> None:
    query = (raw_query or "").strip()
    if not query or st.session_state.is_loading:
        return
    st.session_state.is_loading = True
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("Đang chuẩn bị câu trả lời (UI preview — chưa retrieval thật)…"):
        time.sleep(0.6)  # demo loading state only
        # === FUTURE HOOK: answer = ask_question(user_query) ===
        result = handle_query(query, top_k=int(st.session_state.top_k))
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieval_method": result.get("retrieval_method", "hybrid (demo)"),
            "is_mock": bool(result.get("is_mock", True)),
        }
    )
    st.session_state.is_loading = False


pending_example = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="pubg-header">
  <div>
    <div class="pubg-kicker">PUBG // ESPORTS KNOWLEDGE</div>
    <h1 class="pubg-title">PUBG Policy Assistant</h1>
    <p class="pubg-sub">Quy tắc · Án phạt · Hỗ trợ · Vụ việc</p>
  </div>
  <div class="pubg-badges">
    <span class="badge badge-preview">UI Preview</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main 3-column composition: TanVuu (LEFT) | Chat (CENTER) | Himass (RIGHT)
# ---------------------------------------------------------------------------
left_col, center_col, right_col = st.columns([1.02, 2.35, 1.02], gap="medium")

with left_col:
    st.markdown(
        portrait_html(TANVU_IMG, name="TANVUU", subtitle="PUBG Player", side="left"),
        unsafe_allow_html=True,
    )

with right_col:
    st.markdown(
        portrait_html(HIMASS_IMG, name="HIMASS", subtitle="PUBG Player", side="right"),
        unsafe_allow_html=True,
    )

with center_col:
    st.markdown("<div class='center-wrap'><div class='chat-shell'>", unsafe_allow_html=True)

    # Source filter (UI-only, must not clutter)
    st.markdown(
        "<div class='filter-row'><span class='filter-label'>NGUỒN</span></div>",
        unsafe_allow_html=True,
    )
    filt = st.session_state.source_filter
    try:
        if hasattr(st, "segmented_control"):
            choice = st.segmented_control(
                "Nguồn",
                options=["all", "official", "independent"],
                format_func=lambda v: {
                    "all": "Tất cả",
                    "official": "Official",
                    "independent": "Independent",
                }[v],
                default=filt,
                label_visibility="collapsed",
            )
            if choice:
                st.session_state.source_filter = choice
        else:
            raise AttributeError
    except Exception:
        choice = st.radio(
            "Nguồn",
            options=["all", "official", "independent"],
            format_func=lambda v: {
                "all": "Tất cả",
                "official": "Official",
                "independent": "Independent",
            }[v],
            index=["all", "official", "independent"].index(filt),
            horizontal=True,
            label_visibility="collapsed",
        )
        st.session_state.source_filter = choice

    st.markdown("<div class='chat-scroll'>", unsafe_allow_html=True)

    if not st.session_state.messages:
        # ---- Empty state ----
        st.markdown(
            """
<div class="empty-hero">
  <div class="empty-eyebrow">PUBG POLICY ASSISTANT</div>
  <h2>Bạn muốn tìm hiểu điều gì về PUBG?</h2>
  <p>Tra cứu quy tắc, án phạt, hỗ trợ người chơi và các nguồn liên quan.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        eg_left, eg_right = st.columns(2, gap="small")
        for i, prompt in enumerate(EXAMPLE_PROMPTS):
            target_col = eg_left if i % 2 == 0 else eg_right
            with target_col:
                if st.button(
                    prompt,
                    key=f"example_{i}",
                    use_container_width=True,
                    disabled=st.session_state.is_loading,
                ):
                    pending_example = prompt
        st.markdown(
            "<div class='demo-note'>Các gợi ý trên chỉ là ví dụ giao diện — "
            "chưa kết nối retrieval.</div>",
            unsafe_allow_html=True,
        )
    else:
        # ---- Conversation history ----
        for msg in st.session_state.messages:
            if msg.get("role") == "user":
                st.markdown(user_message_html(msg.get("content", "")), unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='msg'><div class='msg-avatar msg-avatar-bot'>◈</div>"
                    "<div style='flex:1; min-width:0;'>"
                    "<div class='assistant-head'>"
                    "<span class='assistant-name'>PUBG ASSISTANT</span>"
                    "<span class='mock-tag'>UI PREVIEW · DEMO</span>"
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(msg.get("content", ""))
                visible_sources = filter_sources(
                    msg.get("sources", []), st.session_state.source_filter
                )
                st.markdown(
                    assistant_message_html(
                        msg.get("content", ""),
                        sources=visible_sources,
                        retrieval_method=msg.get("retrieval_method", "hybrid (demo)"),
                        is_mock=msg.get("is_mock", True),
                    ),
                    unsafe_allow_html=True,
                )
                st.markdown("</div></div>", unsafe_allow_html=True)

        if st.session_state.is_loading:
            st.markdown(
                "<div class='loading-dots'>● ● ● đang soạn câu trả lời…</div>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)  # close chat-scroll

    # ---- Composer ----
    st.markdown("<div class='composer-zone'>", unsafe_allow_html=True)
    with st.form(key="composer", clear_on_submit=True):
        user_text = st.text_area(
            "Nhập câu hỏi",
            placeholder="Hỏi về PUBG...",
            height=88,
            max_chars=2000,
            label_visibility="collapsed",
            disabled=st.session_state.is_loading,
        )
        send_col, clear_col, hint_col = st.columns([1, 1, 2.2], gap="small")
        with send_col:
            sent = st.form_submit_button(
                "➤ Gửi",
                type="primary",
                use_container_width=True,
                disabled=st.session_state.is_loading,
            )
        with clear_col:
            cleared = st.form_submit_button("✕ Xóa chat", use_container_width=True)
        with hint_col:
            st.caption("Enter để xuống dòng · Nhấn **Gửi** để hỏi (demo).")
    with st.expander("Tùy chọn nâng cao (dành cho backend sau này)", expanded=False):
        st.session_state.top_k = st.slider(
            "Số chunks (top_k — hiện chưa dùng)",
            3,
            10,
            int(st.session_state.top_k),
            disabled=st.session_state.is_loading,
        )
    st.markdown("</div>", unsafe_allow_html=True)  # close composer-zone
    st.markdown("</div></div>", unsafe_allow_html=True)  # close chat-shell + center-wrap

    # ---- Event handling (after render declarations, same run) ----
    if cleared:
        st.session_state.messages = []
        st.session_state.is_loading = False
        st.rerun()
    if sent and (user_text or "").strip():
        _process_query(user_text)
        st.rerun()
    if pending_example:
        _process_query(pending_example)
        st.rerun()

# ---------------------------------------------------------------------------
# Minimal footer / status
# ---------------------------------------------------------------------------
st.markdown(
    """
<div class="pubg-footer">
  <span>◈ UI Preview — backend RAG chưa hoạt động. Mọi câu trả lời &amp; nguồn hiện tại đều là demo.</span>
  <span>Kết nối sau tại <code>src/task10_generation:generate_with_citation</code> qua <code>src/ui_backend.ask_question</code></span>
</div>
""",
    unsafe_allow_html=True,
)
