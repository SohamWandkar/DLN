"""
LearnMate: Evidence-Grounded Course AI Tutor (G06)
Interactive Streamlit Prototype Demonstration
M.Tech AI-ML (First Year) | Deep Learning Networks | CCA2 & LCA-3
"""
import os
import sys
import json
import time
import streamlit as st

# Configure project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    PROCESSED_DIR, CORPUS_DIR, CONFORMAL_ALPHA, 
    DEFAULT_ABSTENTION_THRESHOLD, CALIBRATION_TEMPERATURE
)
from src.retriever import HybridRetriever
from src.evidence_verifier import EvidenceAgreementVerifier
from src.abstention_engine import CalibratedAbstentionEngine
from src.tutor_generator import TutorGenerator

st.set_page_config(
    page_title="LearnMate - Evidence-Grounded Course AI Tutor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #566573;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .grounded-box {
        background-color: #e8f8f5;
        border-left: 5px solid #27ae60;
        padding: 15px;
        border-radius: 6px;
        margin-top: 10px;
    }
    .abstain-box {
        background-color: #fef9e7;
        border-left: 5px solid #f39c12;
        padding: 15px;
        border-radius: 6px;
        margin-top: 10px;
    }
    .hallucinate-box {
        background-color: #fdedec;
        border-left: 5px solid #e74c3c;
        padding: 15px;
        border-radius: 6px;
        margin-top: 10px;
    }
    .citation-badge {
        display: inline-block;
        background-color: #2980b9;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    chunks_file = os.path.join(PROCESSED_DIR, "chunks_corpus.json")
    if not os.path.exists(chunks_file):
        st.error(f"Corpus chunks not found at {chunks_file}. Please run data_processor.py first.")
        return None, None, None, None, []
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)["chunks"]

    retriever = HybridRetriever(corpus)
    verifier = EvidenceAgreementVerifier()
    engine = CalibratedAbstentionEngine(alpha=CONFORMAL_ALPHA)
    generator = TutorGenerator()

    # Calibrate on validation split
    val_file = os.path.join(PROCESSED_DIR, "val_calibration_split.json")
    if os.path.exists(val_file):
        with open(val_file, "r", encoding="utf-8") as f:
            val_items = json.load(f)["items"]
        val_scores, val_labels = [], []
        for it in val_items:
            q = it["question"]
            hits = retriever.retrieve(q, top_k=1)
            txt = hits[0]["text"] if hits else ""
            res = verifier.verify_agreement(txt, q)
            val_scores.append(res["agreement_score"])
            val_labels.append(it["is_answerable"])
        engine.calibrate(val_scores, val_labels)

    return retriever, verifier, engine, generator, corpus

retriever, verifier, engine, generator, corpus = load_system()

# Sidebar Controls
st.sidebar.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
st.sidebar.title("LearnMate Controls")
st.sidebar.markdown("**Course:** Deep Learning Networks (LCA-3)")
st.sidebar.markdown("**Group:** G06 | Education Technology")

st.sidebar.divider()
system_mode = st.sidebar.radio(
    "Active System Architecture:",
    ["LearnMate (Evidence-Grounded + Calibrated Abstention)", 
     "Standard Dense RAG (Vulnerable to Hallucination)"],
    index=0
)

st.sidebar.divider()
alpha_val = st.sidebar.slider(
    "Conformal Risk Tolerance (alpha):",
    min_value=0.01, max_value=0.20, value=CONFORMAL_ALPHA, step=0.01,
    help="Maximum allowed ungrounded error rate. Lower values induce stricter abstention."
)
if engine is not None:
    engine.alpha = alpha_val

st.sidebar.markdown(f"**Calibrated Threshold ($\tau^*$):** `{engine.calibrated_threshold:.4f}`")

st.sidebar.divider()
st.sidebar.markdown("### Indexed Syllabus Modules")
module_names = sorted(list(set(c["module"] for c in corpus))) if corpus else []
for i, m in enumerate(module_names):
    st.sidebar.caption(f"• {m}")

# Main Layout
st.markdown("<div class='main-title'>LearnMate: Evidence-Grounded Course AI Tutor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Department of AI & Machine Learning | Course: Deep Learning Networks | Challenge G06</div>", unsafe_allow_html=True)

# Preset Query Buttons
st.markdown("##### Quick Test Benchmark Scenarios:")
col1, col2, col3, col4, col5 = st.columns(5)
preset_query = ""

with col1:
    if st.button("1. Nesterov NAG\n(In-Syllabus)", use_container_width=True):
        preset_query = "How does Nesterov Accelerated Gradient prevent oscillations?"
with col2:
    if st.button("2. Sigmoid Vanishing\n(In-Syllabus)", use_container_width=True):
        preset_query = "Why does saturated Sigmoid cause vanishing gradients in deep networks?"
with col3:
    if st.button("3. False Premise\n(Contradiction)", use_container_width=True):
        preset_query = "How does Batch Normalization eliminate the need for backpropagation during training?"
with col4:
    if st.button("4. Neural ODEs\n(Missing Topic)", use_container_width=True):
        preset_query = "What differential equation governs continuous-depth Neural ODEs?"
with col5:
    if st.button("5. Macroeconomics\n(Out-of-Domain)", use_container_width=True):
        preset_query = "How does the Taylor Rule calculate central bank monetary interest rates?"

query_input = st.text_input("Ask the Deep Learning AI Tutor a question:", value=preset_query, placeholder="e.g. Explain how Rotary Position Embeddings (RoPE) represent relative displacement...")

if st.button("Submit Question to Tutor", type="primary") or query_input:
    if not query_input.strip():
        st.warning("Please enter a question or click a scenario button above.")
    else:
        st.divider()
        start_t = time.time()
        
        # 1. Retrieve
        hits = retriever.retrieve(query_input, top_k=3)
        top_chunk = hits[0] if hits else {"text": "", "citation": "None"}
        
        # 2. Verify Agreement
        v_res = verifier.verify_agreement(top_chunk["text"], query_input)
        eas = v_res["agreement_score"]
        
        # 3. Decision
        decision = engine.evaluate_decision(eas, top_chunk, v_res)
        latency = (time.time() - start_t) * 1000

        # Display side-by-side or primary based on mode
        left_col, right_col = st.columns([1.5, 1])

        with left_col:
            st.subheader("Tutor Output & Response")
            
            if "Standard Dense RAG" in system_mode:
                # Standard RAG behavior
                naive_resp = generator.generate_naive_rag_response(query_input, top_chunk)
                st.markdown("<div class='hallucinate-box'>", unsafe_allow_html=True)
                st.markdown("#### Standard Uncalibrated RAG Response (No NLI Verifier)")
                st.markdown(naive_resp["response_text"])
                st.markdown(f"**Blind Citation Attached:** `[{naive_resp['citation']}]`")
                st.warning("Notice: Standard RAG answers blindly using parametric heuristics, even when evidence is completely absent or contradictory!")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                # LearnMate behavior
                resp = generator.generate_grounded_response(query_input, top_chunk, decision, v_res)
                
                if decision["should_abstain"]:
                    st.markdown("<div class='abstain-box'>", unsafe_allow_html=True)
                    st.markdown(resp["response_text"])
                    st.info(f"LearnMate actively prevented an ungrounded hallucination by enforcing the calibrated safety threshold ($\tau^* = {engine.calibrated_threshold:.2f}$).")
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='grounded-box'>", unsafe_allow_html=True)
                    st.markdown(resp["response_text"])
                    st.success("LearnMate verified that this claim is mathematically entailed by the official syllabus lecture notes.")
                    st.markdown("</div>", unsafe_allow_html=True)

        with right_col:
            st.subheader("Verification & Calibration Telemetry")
            
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Evidence Agreement Score (EAS)", f"{eas:.3f}", delta=f"Threshold: {engine.calibrated_threshold:.2f}")
            with m2:
                st.metric("Calibrated Confidence", f"{decision['calibrated_confidence']*100:.1f}%")

            m3, m4 = st.columns(2)
            with m3:
                st.metric("Contradiction Penalty", f"{v_res['contradiction']:.2f}")
            with m4:
                st.metric("Query Latency", f"{latency:.1f} ms")

            st.markdown("---")
            st.markdown("##### Retrieved Evidence Passage:")
            st.markdown(f"<span class='citation-badge'>{top_chunk['citation']}</span>", unsafe_allow_html=True)
            st.markdown(f"> *\"{top_chunk['text']}\"*")
            st.caption(f"Dense Similarity: {top_chunk['dense_score']:.3f} | BM25 Score: {top_chunk['bm25_score']:.2f} | RRF Score: {top_chunk['rrf_score']:.4f}")

st.divider()
st.markdown("""
<div style='text-align: center; color: #7f8c8d; font-size: 0.85rem;'>
LearnMate Research Prototype | M.Tech AI-ML | Academic Year 2026-27 | Deep Learning Networks (CCA2 & LCA-3)<br>
Conformal Prediction Risk Control (alpha=0.05) • Sentence-Transformers • Cross-Encoder NLI Evidence Agreement
</div>
""", unsafe_allow_html=True)
