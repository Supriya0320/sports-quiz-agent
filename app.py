"""
app.py
------
Front-end entry point. Coordinates database warm-up, user input, the RAG
generation pipeline, and renders an interactive quiz with instant feedback.

Run with:  streamlit run app.py
"""

import streamlit as st

from src.config import SUPPORTED_SPORTS, SUPPORTED_DIFFICULTIES, DEFAULT_NUM_QUESTIONS, validate_config
from src.database import setup_and_populate_db, get_collection_stats
from src.generator import compile_quiz_data, QuizGenerationError


@st.cache_resource
def prepare_knowledge_base():
    setup_and_populate_db()
    return get_collection_stats()


st.set_page_config(page_title="Sports Quiz Agent", page_icon="🏆", layout="centered")

config_warnings = validate_config()
db_stats = prepare_knowledge_base()

st.title("🏆 AI-Powered Sports Quiz Generator")
st.caption("RAG agent: ChromaDB (offline facts) + live web search + LLM generation")

if config_warnings:
    for w in config_warnings:
        st.warning(w)

st.sidebar.header("Quiz Settings")
sport_choice = st.sidebar.selectbox("Select Sport", SUPPORTED_SPORTS)
difficulty = st.sidebar.select_slider("Select Difficulty", options=SUPPORTED_DIFFICULTIES, value="Medium")
num_questions = st.sidebar.slider("Number of Questions", min_value=4, max_value=5, value=DEFAULT_NUM_QUESTIONS)

st.sidebar.divider()
st.sidebar.caption(f"📚 Knowledge base: {db_stats['total_facts']} offline facts stored in ChromaDB")

generate_clicked = st.sidebar.button("🎲 Generate Fresh Quiz", use_container_width=True, type="primary")

if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
    st.session_state.quiz_context = None
    st.session_state.quiz_sport = None
    st.session_state.quiz_difficulty = None
    st.session_state.answers = {}

if generate_clicked:
    with st.spinner("Fetching historical facts & scouring the live web..."):
        try:
            questions, context_used = compile_quiz_data(sport_choice, difficulty, num_questions)
            st.session_state.quiz_questions = questions
            st.session_state.quiz_context = context_used
            st.session_state.quiz_sport = sport_choice
            st.session_state.quiz_difficulty = difficulty
            st.session_state.answers = {}
            st.success(f"Generated {len(questions)} questions!")
        except QuizGenerationError as e:
            st.error(f"Could not generate quiz: {e}")

if st.session_state.quiz_questions:
    st.subheader(f"Quiz: {st.session_state.quiz_sport} ({st.session_state.quiz_difficulty})")

    for idx, q in enumerate(st.session_state.quiz_questions):
        st.markdown(f"**Q{idx + 1}. {q['question']}**")

        option_labels = [f"{letter}. {q['options'][letter]}" for letter in ("A", "B", "C", "D")]
        selected = st.radio(
            label=f"Choose an answer for question {idx + 1}",
            options=option_labels,
            index=None,
            key=f"radio_{idx}",
            label_visibility="collapsed",
        )

        if selected is not None:
            selected_letter = selected[0]
            correct_letter = q["correct_answer"]
            if selected_letter == correct_letter:
                st.success(f"✅ Correct! **{correct_letter}. {q['options'][correct_letter]}**")
            else:
                st.error(
                    f"❌ Not quite. Correct answer: **{correct_letter}. {q['options'][correct_letter]}**"
                )
            st.info(f"💡 Explanation: {q['explanation']}")

        st.divider()

    with st.expander("🔍 Inspect Ground Truth (RAG Context Used)"):
        st.code(st.session_state.quiz_context, language="markdown")

    with st.expander("📋 Copy as Plain Text"):
        lines = [f"Sport: {st.session_state.quiz_sport}", f"Difficulty: {st.session_state.quiz_difficulty}", ""]
        for idx, q in enumerate(st.session_state.quiz_questions):
            lines.append(f"Question {idx + 1}: {q['question']}")
            for letter in ("A", "B", "C", "D"):
                lines.append(f"{letter}. {q['options'][letter]}")
            lines.append(f"Correct Answer: {q['correct_answer']}. {q['options'][q['correct_answer']]}")
            lines.append(f"Explanation: {q['explanation']}")
            lines.append("")
        st.text_area("Quiz text", value="\n".join(lines), height=300)
else:
    st.write(
        "👈 Pick a sport and difficulty in the sidebar, then click **Generate Fresh Quiz** "
        "to create a grounded, fact-checked quiz."
    )