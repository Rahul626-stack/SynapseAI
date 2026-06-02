"""
Synapse AI — Ultra-Premium Educational Dashboard
Adaptive quiz generation powered by CrewAI + RAG + Bloom's Taxonomy.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

# Ensure the 'src' directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

# ── Background warm-up: start loading model+DB without blocking UI ────────────
import threading
from synapse_ai.rag import _get_embed_model, _get_client

if "warmup_done" not in st.session_state:
    st.session_state["warmup_done"] = False
    def _warmup():
        _get_embed_model()
        _get_client()
        st.session_state["warmup_done"] = True
    threading.Thread(target=_warmup, daemon=True).start()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synapse AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS for Ultra-Premium UI ────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography & Theme */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Background & Ambient Glow */
    .stApp {
        background-color: #030712 !important;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(59, 130, 246, 0.18), transparent 40%),
            radial-gradient(circle at 90% 90%, rgba(139, 92, 246, 0.18), transparent 40%),
            radial-gradient(circle at 50% 50%, rgba(16, 185, 129, 0.08), transparent 50%) !important;
        background-attachment: fixed;
        color: #F8FAFC !important;
    }

    /* Hide Streamlit Chrome */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    footer {display: none !important;}

    /* Main Layout */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 4rem !important;
        max-width: 1100px !important;
    }
    
    /* Typography Hierarchy */
    h1 {
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        background: linear-gradient(180deg, #FFFFFF 0%, #A1A1AA 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem !important;
    }
    h2, h3 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #E2E8F0 !important;
    }
    p, li {
        color: #94A3B8 !important;
        line-height: 1.6 !important;
        font-weight: 400 !important;
    }

    /* Glassmorphism Cards (Quiz Cards & Containers) */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
        box-shadow: 0 8px 40px rgba(0, 0, 0, 0.2);
    }
    
    /* Bento Grid (Hero Metrics) */
    .bento-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-top: 32px;
        margin-bottom: 32px;
    }
    .bento-item {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: transform 0.2s ease;
    }
    .bento-item:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(255, 255, 255, 0.1);
    }
    .bento-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
    }
    .bento-label {
        font-size: 0.875rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        background: rgba(255, 255, 255, 0.05) !important;
        color: #FFFFFF !important;
        height: auto !important;
        padding: 10px 20px !important;
    }
    .stButton > button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        transform: scale(0.98) !important;
    }
    /* Primary Action Button (Generate Quiz) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%) !important;
        border: none !important;
        box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 25px rgba(139, 92, 246, 0.5) !important;
        transform: translateY(-2px) scale(0.99) !important;
    }
    
    /* FIX: Radio Buttons (Make text visible in dark mode) */
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #E2E8F0 !important;
        font-weight: 500 !important;
        font-size: 15px !important;
    }
    .stRadio > div[role="radiogroup"] > label > div:first-child {
        background-color: rgba(255,255,255,0.05) !important;
        border-color: rgba(255,255,255,0.2) !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(10, 10, 10, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] div {
        color: #CBD5E1 !important;
    }
    
    /* Inputs (Text, Number, Select) */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        color: #FFFFFF !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #8B5CF6 !important;
        box-shadow: 0 0 0 1px #8B5CF6 !important;
    }
    
    /* Badges */
    .bloom-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Score Banner */
    .score-banner {
        position: relative;
        padding: 40px 24px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 32px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .score-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.8;
        z-index: -1;
    }
    .score-banner h2 {
        font-size: 56px;
        font-weight: 800 !important;
        margin: 0;
        color: #FFFFFF !important;
        line-height: 1;
        letter-spacing: -0.05em !important;
    }
    .score-banner p {
        font-size: 16px;
        font-weight: 500 !important;
        margin: 12px 0 0 0;
        color: rgba(255,255,255,0.8) !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Streamlit Expander styling */
    .st-emotion-cache-p5msec {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3B82F6, #8B5CF6) !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Bloom's mapping (Dark Theme Optimized) ──────────────────────────────────
BLOOM_COLORS = {
    "Remember":   {"bg": "rgba(2, 119, 189, 0.1)", "text": "#38BDF8", "hex": "#38BDF8"},
    "Understand": {"bg": "rgba(123, 31, 162, 0.1)", "text": "#C084FC", "hex": "#C084FC"},
    "Apply":      {"bg": "rgba(46, 125, 50, 0.1)",  "text": "#4ADE80", "hex": "#4ADE80"},
    "Analyze":    {"bg": "rgba(230, 81, 0, 0.1)",   "text": "#FB923C", "hex": "#FB923C"},
    "Evaluate":   {"bg": "rgba(198, 40, 40, 0.1)",  "text": "#F87171", "hex": "#F87171"},
    "Create":     {"bg": "rgba(173, 20, 87, 0.1)",  "text": "#F472B6", "hex": "#F472B6"},
}
BLOOM_ORDER = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
# ── Question Type Instructions ────────────────────────────────────────────────
def build_question_type_instructions(num_questions: int, mapped_types: list) -> str:
    if not mapped_types:
        return ""
    
    base_count = num_questions // len(mapped_types)
    remainder = num_questions % len(mapped_types)
    
    distribution = {}
    for i, qt in enumerate(mapped_types):
        count = base_count + (1 if i < remainder else 0)
        distribution[qt] = count
        
    lines = ["QUESTION TYPE DISTRIBUTION REQUIREMENT:"]
    lines.append("You MUST generate exactly the following number of questions per type:\n")
    for qt, count in distribution.items():
        if count > 0:
            lines.append(f"- {qt}: {count} question(s).")
            
    return "\n".join(lines)


# ── Callbacks for CrewAI Streaming ──────────────────────────────────────────
def agent_step_callback(step_output):
    """Callback triggered on each step of the agent to stream updates to UI."""
    if "agent_status" not in st.session_state:
        st.session_state["agent_status"] = []
    
    text = getattr(step_output, 'text', str(step_output))
    clean_text = str(text).split('\n')[0][:80] + "..." if len(str(text)) > 80 else str(text)
    
    st.session_state["agent_status"].append(f"✦ Synapse Core: {clean_text}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 30px;">
            <div style="background: linear-gradient(135deg, #3B82F6, #8B5CF6); border-radius: 8px; width: 32px; height: 32px; display: flex; align-items: center; justify-content: center; margin-right: 12px; font-weight: 800; color: white;">S</div>
            <h2 style="margin: 0; font-weight: 700; letter-spacing: -0.02em;">Synapse AI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B;'>Knowledge Source</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Textbook", type=["pdf"], label_visibility="collapsed")
    
    st.markdown("<br><p style='font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #64748B;'>Engine Parameters</p>", unsafe_allow_html=True)
    topic = st.text_input("Focus Topic", placeholder="Leave blank for entire document")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        num_questions = st.number_input("Questions", min_value=3, max_value=20, value=5)
    with col_s2:
        difficulty = st.selectbox("Level", ["Adaptive", "Beginner", "Advanced"])

    q_types = st.multiselect(
        "Question Architecture",
        ["MCQ", "True/False", "Short Answer"],
        default=["MCQ"],
    )

    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("Synthesize Quiz", use_container_width=True, type="primary")


# ── Detect new PDF upload and clear stale cache ──────────────────────────────
if uploaded_file and uploaded_file.name != st.session_state.get("pdf_name"):
    for key in ["col_name", "pdf_name", "chunk_count", "questions", "answers", "submitted", "quiz_topic"]:
        st.session_state.pop(key, None)

# ── Ingest PDF on upload ──────────────────────────────────────────────────────
if uploaded_file and "col_name" not in st.session_state:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    with st.status("Initializing Knowledge Base...", expanded=True) as status:
        from synapse_ai.rag import ingest_pdf, _get_client
        st.write("Extracting vector context via PyMuPDF...")
        st.write("Applying semantic fracturing...")
        
        try:
            st.session_state["col_name"] = ingest_pdf(tmp_path, original_filename=uploaded_file.name)
        except ValueError as e:
            status.update(label="Ingestion Failed", state="error", expanded=False)
            st.error(f"Could not process the uploaded PDF: {e}")
            st.stop()
        
        st.session_state["pdf_name"] = uploaded_file.name
        
        client = _get_client()
        try:
            col = client.get_collection(st.session_state["col_name"])
            st.session_state["chunk_count"] = col.count()
        except Exception:
            st.session_state["chunk_count"] = 0
            
        status.update(label="Knowledge Base Synchronized", state="complete", expanded=False)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass


# ── Main area — Welcome screen ──────────────────────────────────────────────
if not generate_btn and "questions" not in st.session_state:
    if not uploaded_file:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.title("Accelerate human cognition.")
        st.markdown(
            "<p style='font-size: 20px; max-width: 600px;'>Upload a document and let Synapse orchestrate a crew of autonomous AI agents to construct a personalized, pedagogically-sound assessment.</p>",
            unsafe_allow_html=True
        )
        
        st.markdown(f"""
        <div class="bento-grid">
            <div class="bento-item">
                <p class="bento-value">6</p>
                <p class="bento-label">Cognitive Tiers</p>
            </div>
            <div class="bento-item">
                <p class="bento-value">Groq</p>
                <p class="bento-label">LLM Engine</p>
            </div>
            <div class="bento-item">
                <p class="bento-value">CrewAI</p>
                <p class="bento-label">Orchestration</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("Initialize the engine by uploading a PDF document in the sidebar.")
        st.stop()
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.title(f"Target Acquired: {uploaded_file.name}")
        
        st.markdown(f"""
        <div class="bento-grid">
            <div class="bento-item">
                <p class="bento-value" style="color: #4ADE80;">Ready</p>
                <p class="bento-label">System Status</p>
            </div>
            <div class="bento-item">
                <p class="bento-value">{st.session_state.get('chunk_count', 0)}</p>
                <p class="bento-label">Semantic Vectors</p>
            </div>
            <div class="bento-item">
                <p class="bento-value" style="font-size: 1.5rem; line-height: 1.2;">{topic if topic else "Full Document"}</p>
                <p class="bento-label">Target Scope</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<p style='font-size: 18px;'>Configure your parameters in the sidebar and click <b>Synthesize Quiz</b> to begin.</p>", unsafe_allow_html=True)


# ── Generate quiz ─────────────────────────────────────────────────────────────
if generate_btn:
    if not topic:
        topic = "Entire Document"

    type_map = {"MCQ": "multiple_choice", "True/False": "true_false", "Short Answer": "short_answer"}
    question_types_str = ", ".join(type_map.get(t, t) for t in q_types)

    st.session_state["agent_status"] = []
    
    with st.status("Neural Synthesis in Progress...", expanded=True) as status_box:
        st.write("Retrieving high-dimensional semantic context...")
        from synapse_ai.rag import retrieve_context
        from synapse_ai.crew import SynapseAICrew
        from synapse_ai.blooms import build_bloom_prompt_instructions

        context = retrieve_context(st.session_state["col_name"], topic, top_k=6)
        
        if not context.strip():
            status_box.update(label="Context Retrieval Failed", state="error")
            st.error(f"Could not find any relevant vectors matching '{topic}'.")
            st.stop()
            
        st.write(f"Context locked. Mapping to Bloom's taxonomy...")
        bloom_instructions = build_bloom_prompt_instructions(num_questions)
        
        mapped_types = [type_map.get(t, t) for t in q_types]
        question_type_instructions = build_question_type_instructions(num_questions, mapped_types)

        st.write("Delegating to autonomous agents...")
        
        crew_engine = SynapseAICrew()
        crew_engine.step_callback = agent_step_callback
        
        import random
        random_seed = random.randint(10000, 99999)
        randomness_instruction = f"IMPORTANT VARIABILITY REQUIREMENT (Seed {random_seed}): You MUST deliberately choose different facts, concepts, or perspectives from the context than you normally would. Ensure this quiz is highly unique and does not repeat standard generic questions."
        
        try:
            result = crew_engine.crew().kickoff(
                inputs={
                    "topic": topic,
                    "context": context,
                    "num_questions": num_questions,
                    "question_types": question_types_str,
                    "question_type_instructions": question_type_instructions,
                    "bloom_instructions": bloom_instructions,
                    "randomness_instruction": randomness_instruction,
                }
            )
        except Exception as e:
            status_box.update(label="Engine Failure", state="error")
            err_str = str(e)
            if "503" in err_str or "ServiceUnavailableError" in err_str:
                st.error("The Groq AI engine is currently experiencing high demand. Please wait a few moments and try generating the quiz again.")
            else:
                st.error(f"An unexpected engine failure occurred: {err_str}")
            st.stop()

        status_box.update(label="Synthesis Complete", state="complete", expanded=False)

        # Parse output
        if hasattr(result, "pydantic") and result.pydantic is not None:
            quiz_output = result.pydantic
            questions = [q.model_dump() for q in quiz_output.questions]
        else:
            import json
            raw = result.raw if hasattr(result, "raw") else str(result)
            raw = raw.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                parsed = json.loads(raw)
                questions = parsed.get("questions", parsed) if isinstance(parsed, dict) else parsed
            except Exception as e:
                st.error(f"Engine parsing failure: {e}")
                st.code(str(result))
                st.stop()

        st.session_state["questions"] = questions
        st.session_state["answers"] = {}
        st.session_state["submitted"] = False
        st.session_state["quiz_topic"] = topic


# ── Render quiz ───────────────────────────────────────────────────────────────
if "questions" in st.session_state and st.session_state["questions"]:
    questions = st.session_state["questions"]
    quiz_topic = st.session_state.get("quiz_topic", "Assessment")

    if not st.session_state.get("submitted"):
        st.markdown("<br>", unsafe_allow_html=True)
        st.title(f"Evaluation: {quiz_topic}")
        st.progress(0)
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("quiz_form", border=False):
            for i, q in enumerate(questions):
                level = q.get("bloom_level", "Remember")
                colors = BLOOM_COLORS.get(level, {"bg": "rgba(255,255,255,0.1)", "text": "#FFF"})
                
                # Question wrapper in HTML for premium feel, leaving radio logic in Streamlit
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0; border-bottom: none;">
                        <span class="bloom-badge" style="background-color: {colors['bg']}; color: {colors['text']}; border-color: {colors['text']}40;">
                            {level}
                        </span>
                        <h3 style="margin: 0; font-size: 1.1rem; line-height: 1.5;">{i+1}. {q['question']}</h3>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Inject a container for the radio button that matches the glass card style
                st.markdown("""<div class="glass-card" style="margin-top: 0; border-top-left-radius: 0; border-top-right-radius: 0; padding-top: 0; background: rgba(255, 255, 255, 0.01);">""", unsafe_allow_html=True)
                
                options = q.get("options", [])
                q_type = q.get("type", "multiple_choice")

                if q_type == "true_false":
                    options = ["True", "False", "[Leave Blank]"]
                    st.session_state["answers"][i] = st.radio(
                        f"q_{i}_label",
                        options,
                        index=2,
                        key=f"q_{i}",
                        label_visibility="collapsed"
                    )
                elif options:
                    if "[Leave Blank]" not in options:
                        options.append("[Leave Blank]")
                    st.session_state["answers"][i] = st.radio(
                        f"q_{i}_label",
                        options,
                        index=len(options)-1,
                        key=f"q_{i}",
                        label_visibility="collapsed"
                    )
                else:
                    st.session_state["answers"][i] = st.text_input(f"q_{i}_label", key=f"q_{i}", label_visibility="collapsed", placeholder="Type your answer here...")
                
                st.markdown("</div><br>", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Complete Evaluation", use_container_width=True, type="primary")
            
            if submitted:
                st.session_state["submitted"] = True
                st.rerun()

    # ── Results & Analytics ──────────────────────────────────────────────────
    else:
        answers = st.session_state.get("answers", {})
        correct, total = 0, 0
        bloom_scores = {level: {"correct": 0, "total": 0} for level in BLOOM_ORDER}

        for i, q in enumerate(questions):
            correct_answer = q.get("correct_answer", "")
            if not correct_answer: continue
            
            total += 1
            level = q.get("bloom_level", "Remember")
            user_ans = answers.get(i, "")
            
            is_blank = user_ans == "[Leave Blank]" or str(user_ans).strip() == ""
            is_correct = False if is_blank else str(user_ans).strip().lower() == str(correct_answer).strip().lower()
            
            if is_correct:
                correct += 1
                bloom_scores[level]["correct"] += 1
            bloom_scores[level]["total"] += 1

        pct = int((correct / total) * 100) if total else 0
        
        # Dynamic gradient based on score
        if pct >= 80:
            banner_bg = "linear-gradient(135deg, rgba(56, 239, 125, 0.2) 0%, rgba(17, 153, 142, 0.2) 100%)"
            banner_border = "rgba(56, 239, 125, 0.4)"
        elif pct >= 50:
            banner_bg = "linear-gradient(135deg, rgba(242, 201, 76, 0.2) 0%, rgba(242, 153, 74, 0.2) 100%)"
            banner_border = "rgba(242, 201, 76, 0.4)"
        else:
            banner_bg = "linear-gradient(135deg, rgba(244, 92, 67, 0.2) 0%, rgba(235, 51, 73, 0.2) 100%)"
            banner_border = "rgba(244, 92, 67, 0.4)"

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="score-banner" style="background: {banner_bg}; border-color: {banner_border}; backdrop-filter: blur(20px);">
                <p>Synthesis Accuracy</p>
                <h2>{pct}%</h2>
                <p>{correct} of {total} Nodes Verified</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

        tab1, tab2 = st.tabs(["Cognitive Metrics", "Node Inspection"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True)
            col_chart1, col_chart2 = st.columns(2)
            
            categories = []
            values = []
            for lvl in BLOOM_ORDER:
                if bloom_scores[lvl]["total"] > 0:
                    categories.append(lvl)
                    score = (bloom_scores[lvl]["correct"] / bloom_scores[lvl]["total"]) * 100
                    values.append(score)
                    
            if categories:
                # Transparent Plotly Config
                layout_opts = dict(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94A3B8', family='Inter'),
                    margin=dict(t=40, b=40, l=40, r=40)
                )

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=categories, fill='toself', 
                    name='Accuracy', line_color='#8B5CF6',
                    fillcolor='rgba(139, 92, 246, 0.3)'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 100], color='rgba(255,255,255,0.2)', gridcolor='rgba(255,255,255,0.1)'),
                        angularaxis=dict(gridcolor='rgba(255,255,255,0.1)')
                    ),
                    showlegend=False,
                    **layout_opts
                )
                col_chart1.plotly_chart(fig_radar, use_container_width=True)
                
                counts = [bloom_scores[lvl]["total"] for lvl in categories]
                colors = [BLOOM_COLORS[lvl]["hex"] for lvl in categories]
                
                fig_bar = px.bar(
                    x=categories, y=counts, 
                    color=categories, color_discrete_sequence=colors,
                    labels={"x": "", "y": "Questions"}
                )
                fig_bar.update_layout(
                    showlegend=False,
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                    **layout_opts
                )
                col_chart2.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True)
            for i, q in enumerate(questions):
                user_ans = answers.get(i, "")
                correct_ans = q.get("correct_answer", "")
                
                is_blank = user_ans == "[Leave Blank]" or str(user_ans).strip() == ""
                is_correct = False if is_blank else str(user_ans).strip().lower() == str(correct_ans).strip().lower()
                level = q.get("bloom_level", "Remember")
                
                icon = "⚪" if is_blank else ("🟢" if is_correct else "🔴")
                display_ans = "[Unanswered]" if is_blank else user_ans
                
                with st.expander(f"{icon} Node {i+1}: {q['question']}"):
                    st.markdown(f"**Your Input:** `{display_ans}`")
                    if not is_correct:
                        st.markdown(f"**Valid Signature:** `{correct_ans}`")
                    
                    st.markdown(f"**Cognitive Tier:** {level}")
                    st.markdown(f"**Engine Logic:** {q.get('explanation', 'No reasoning provided.')}")
                    
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("Reset Engine", use_container_width=True):
            for key in ["questions", "answers", "submitted", "quiz_topic"]:
                st.session_state.pop(key, None)
            st.rerun()


# ── Boot overlay: runs AFTER full UI renders, dims background ────────────────
if not st.session_state.get("warmup_done"):
    import time
    _overlay = st.empty()

    def _render_overlay(pct: int, label: str):
        bar_w = max(pct, 2)
        _overlay.markdown(f"""
        <div style="
            position: fixed; inset: 0; z-index: 99999;
            background: rgba(3, 7, 18, 0.75);
            backdrop-filter: blur(6px);
            -webkit-backdrop-filter: blur(6px);
            display: flex; align-items: center; justify-content: center;
        ">
            <div style="
                background: rgba(15, 23, 42, 0.95);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 20px;
                padding: 36px 44px;
                min-width: 320px;
                box-shadow: 0 24px 60px rgba(0,0,0,0.5);
                text-align: center;
            ">
                <p style="margin:0 0 6px; font-family:Inter,sans-serif; font-size:13px;
                          font-weight:700; letter-spacing:0.12em; text-transform:uppercase;
                          color:#64748B;">SYNAPSE AI</p>
                <h3 style="margin:0 0 28px; font-family:Inter,sans-serif; font-size:20px;
                           font-weight:700; color:#F1F5F9; letter-spacing:-0.02em;">
                      Initializing Engine
                </h3>
                <div style="
                    background: rgba(255,255,255,0.05);
                    border-radius: 999px;
                    height: 6px;
                    width: 100%;
                    overflow: hidden;
                    margin-bottom: 14px;
                ">
                    <div style="
                        width: {bar_w}%;
                        height: 100%;
                        border-radius: 999px;
                        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
                        box-shadow: 0 0 12px rgba(139,92,246,0.6);
                        transition: width 0.4s ease;
                    "></div>
                </div>
                <div style="display:flex; justify-content:space-between;
                            font-family:Inter,sans-serif; font-size:12px; color:#64748B; margin-bottom:10px;">
                    <span>{label}</span>
                    <span style="color:#8B5CF6; font-weight:700;">{pct}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    _render_overlay(0, "Loading embedding model...")
    _get_embed_model()
    _render_overlay(65, "Connecting to vector store...")
    _get_client()
    _render_overlay(100, "Engine ready")
    st.session_state["warmup_done"] = True
    time.sleep(0.5)
    st.rerun()
