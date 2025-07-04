import streamlit as st
from document_handler import parse_document
from summarizer import generate_summary
from qa_engine import answer_question
from logic_generator import generate_logic_question, evaluate_answer
from memory_manager import MemoryManager

st.set_page_config(
    page_title="REsearch AssistBuddy",
    layout="wide",
    page_icon="🔍"
)
memory = MemoryManager()

# ---- Modern Colorful Patches + Research Icons (no rectangles, patches away from UI) ----
st.markdown("""
    <style>
    body {
        min-height: 100vh;
        background: linear-gradient(120deg, #181c2f 0%, #232946 40%, #6247ea 80%, #e45858 100%);
        background-attachment: fixed;
        overflow-x: hidden;
    }
    /* Colorful blurred patches for pleasant blend, positioned away from main UI */
    .color-patch1, .color-patch2, .color-patch3, .color-patch4 {
        position: fixed;
        z-index: 0;
        pointer-events: none;
        border-radius: 40%;
        filter: blur(70px);
        opacity: 0.38;
        mix-blend-mode: lighten;
    }
    .color-patch1 {
        top: 2vh; left: 2vw; width: 220px; height: 120px;
        background: radial-gradient(circle, #6247ea 0%, #232946 100%);
    }
    .color-patch2 {
        top: 5vh; right: 2vw; width: 180px; height: 100px;
        background: radial-gradient(circle, #e45858 0%, #6247ea 100%);
    }
    .color-patch3 {
        bottom: 8vh; left: 2vw; width: 180px; height: 120px;
        background: radial-gradient(circle, #43e97b 0%, #38f9d7 100%);
    }
    .color-patch4 {
        bottom: 8vh; right: 2vw; width: 200px; height: 120px;
        background: radial-gradient(circle, #f4d35e 0%, #6247ea 100%);
    }
    /* Button and input effects */
    .stButton>button {
        background: linear-gradient(90deg, #6247ea, #e45858, #232946);
        color: #f4f4f4;
        border: 2px solid #6247ea;
        border-radius: 50px;
        padding: 0.9rem 2rem;
        font-weight: bold;
        box-shadow: 0px 0px 20px #6247ea, 0 0 10px #e45858 inset;
        transition: 0.4s cubic-bezier(.4,2,.6,1);
        outline: none;
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #e45858, #6247ea, #232946);
        box-shadow: 0px 0px 40px #e45858, 0 0 20px #6247ea inset;
        border: 2px solid #e45858;
        color: #fff;
        letter-spacing: 2px;
    }
    .stTextInput>div>div>input {
        background-color: #232946;
        color: #f4f4f4;
        border: 2px solid #6247ea;
        border-radius: 25px;
        padding: 0.75rem;
        box-shadow: 0 0 10px #6247ea inset;
        transition: border 0.3s, box-shadow 0.3s;
    }
    .stTextInput>div>div>input:focus {
        border: 2px solid #e45858;
        box-shadow: 0 0 20px #e45858;
    }
    .stRadio>div>label {
        color: #f4f4f4;
        font-weight: bold;
    }
    .stMarkdown {
        color: #f4f4f4;
    }
    .stFileUploader label {
        color: #f4f4f4;
    }
    .stFileUploader>div {
        border: 2px solid #6247ea !important;
        border-radius: 20px !important;
        box-shadow: 0 0 20px #6247ea, 0 0 40px #e45858 inset;
        background: rgba(35,41,70,0.85) !important;
        animation: borderGlow 2s infinite alternate;
        z-index: 10 !important;
        position: relative !important;
    }
    @keyframes borderGlow {
        0% { box-shadow: 0 0 20px #6247ea, 0 0 40px #e45858 inset; }
        100% { box-shadow: 0 0 40px #e45858, 0 0 60px #6247ea inset; }
    }
    /* Neon scrollbars */
    ::-webkit-scrollbar {
        width: 10px;
        background: #232946;
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(90deg, #6247ea, #e45858);
        border-radius: 10px;
        box-shadow: 0 0 10px #6247ea;
    }
    /* Make the bot name normal, not too bright */
    h1, h2, h3, h4 {
        color: #f4f4f4 !important;
        text-align: center;
        font-family: "Segoe UI", sans-serif;
        text-shadow: none !important;
        font-weight: 600;
    }
    /* Research icons (SVG) in background, faded and away from UI */
    .research-icon {
        position: fixed;
        z-index: 0;
        opacity: 0.10;
        pointer-events: none;
    }
    .icon1 { left: 8vw; top: 30vh; width: 120px; }
    .icon2 { right: 10vw; top: 60vh; width: 100px; }
    .icon3 { left: 45vw; bottom: 10vh; width: 140px; }
    </style>
    <div class="color-patch1"></div>
    <div class="color-patch2"></div>
    <div class="color-patch3"></div>
    <div class="color-patch4"></div>
    <svg class="research-icon icon1" viewBox="0 0 64 64" fill="#fff">
        <circle cx="32" cy="32" r="28" stroke="#6247ea" stroke-width="4" fill="#232946"/>
        <rect x="22" y="18" width="20" height="28" rx="6" fill="#e45858"/>
        <rect x="28" y="24" width="8" height="16" rx="2" fill="#fff"/>
    </svg>
    <svg class="research-icon icon2" viewBox="0 0 64 64" fill="#fff">
        <ellipse cx="32" cy="32" rx="28" ry="18" stroke="#43e97b" stroke-width="4" fill="#232946"/>
        <rect x="20" y="28" width="24" height="8" rx="4" fill="#43e97b"/>
    </svg>
    <svg class="research-icon icon3" viewBox="0 0 64 64" fill="#fff">
        <rect x="12" y="12" width="40" height="40" rx="10" stroke="#f4d35e" stroke-width="4" fill="#232946"/>
        <circle cx="32" cy="32" r="8" fill="#f4d35e"/>
    </svg>
""", unsafe_allow_html=True)

# ---- Logo + App Title ----
st.markdown("""
<div style="text-align:center;">
    <svg width="72" height="72" viewBox="0 0 24 24" fill="#6247ea" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10
        10-4.48 10-10S17.52 2 12 2zm0 3a2 2 0 110 4 2 2 0 010-4zm0
        14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08
        1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
    </svg>
    <h1>REsearch AssistBuddy</h1>
</div>
""", unsafe_allow_html=True)

# ---- Setup Section (Optional Switches) ----
st.markdown("#### ⚙️ Quick Settings")
switch1 = st.checkbox("Allow anonymous usage data for AI improvement")
switch2 = st.checkbox("Auto-launch AssistBuddy on startup", value=True)
switch3 = st.checkbox("Enable global shortcut", value=True)

if st.button("🚀 Launch Assistant"):
    st.experimental_rerun()  # Simulate proceed

# ---- Core Functionalities ----
uploaded_file = st.file_uploader("📂 Upload Research Document (PDF or TXT)", type=["pdf", "txt"])
if uploaded_file:
    with st.spinner("Parsing your document..."):
        doc_text = parse_document(uploaded_file)
        memory.store_context(doc_text)
        summary = generate_summary(doc_text)
        st.success("✅ Document Loaded!")

    st.subheader("📄 Document Summary")
    st.write(summary)

    mode = st.radio("🎛️ Select Interaction Mode", ["Ask Anything", "Challenge Me"], horizontal=True)

    if mode == "Ask Anything":
        question = st.text_input("🤔 Ask your question:")
        if question:
            answer, justification = answer_question(question, memory.get_context())
            st.markdown(f"**Answer:** {answer}")
            if justification:
                st.markdown(f"> _{justification}_")
            st.experimental_rerun()

    elif mode == "Challenge Me":
        if st.button("💡 Give me a logic-based question"):
            question, solution = generate_logic_question(memory.get_context())
            memory.store_temp_answer(solution)
            st.markdown(f"**Question:** {question}")

        user_input = st.text_input("📝 Your Answer:")
        if user_input and memory.temp_answer:
            evaluation = evaluate_answer(user_input, memory.temp_answer)
            st.markdown(f"**Evaluation:** {evaluation}")