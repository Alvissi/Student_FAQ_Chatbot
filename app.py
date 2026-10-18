"""
app.py
------
Student FAQ Chatbot -- Streamlit web application.

This is the user-facing part of the research project. It loads a trained
machine learning model (TF-IDF + Multinomial Naive Bayes), lets a student
type a question, classifies the question's intent into one of three
categories, and returns a predefined sample answer for that category.

Run locally with:  streamlit run app.py
"""

import time
import pandas as pd
import streamlit as st

from train_model import load_or_train_model, BASE_DIR
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# If the model's top predicted probability is below this value, we treat the
# question as "not confidently recognized" and show the fallback message
# instead of possibly giving a wrong answer. This value was chosen by
# checking that clearly unrelated/off-topic questions in testing scored
# around 0.33 confidence, while genuine in-scope questions (even reworded
# ones) scored 0.43 and above. See README.md for details.
CONFIDENCE_THRESHOLD = 0.40

FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't identify your question. Please ask about "
    "enrollment, school requirements, or class schedules."
)

ANSWERS_PATH = os.path.join(BASE_DIR, "faq_answers.csv")

EXAMPLE_QUESTIONS = {
    "Enrollment": [
        "How do I enroll?",
        "What are the enrollment dates?",
        "How do returning students enroll?",
    ],
    "School Requirements": [
        "What documents should I submit?",
        "What are the school requirements?",
        "Do I need a good moral certificate?",
    ],
    "Class Schedules": [
        "Where can I see my class schedule?",
        "How do I check my subjects and schedule?",
        "Where do I find the room assignment for my class?",
    ],
}

# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Student Help Desk",
    page_icon="🎓",
    layout="centered",
)

CUSTOM_CSS = """
<style>
    .main {
        background-color: #FAFAFA;
    }
    .block-container {
        max-width: 760px;
        padding-top: 2rem;
    }
    h1 {
        color: #1A2A4C;
        font-weight: 700;
    }
    .subtitle {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .category-card {
        border: 1px solid #E2E5EA;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        background-color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    .chat-bubble-user {
        background-color: #1D3A8A;
        color: #FFFFFF;
        padding: 0.65rem 1rem;
        border-radius: 14px 14px 2px 14px;
        margin: 0.4rem 0;
        max-width: 85%;
        margin-left: auto;
        text-align: left;
    }
    .chat-bubble-bot {
        background-color: #F1F3F6;
        color: #111827;
        padding: 0.65rem 1rem;
        border-radius: 14px 14px 14px 2px;
        margin: 0.4rem 0;
        max-width: 85%;
        border: 1px solid #E2E5EA;
    }
    .research-note {
        color: #6B7280;
        font-size: 0.82rem;
        text-align: center;
        margin-top: 2rem;
        border-top: 1px solid #E5E7EB;
        padding-top: 0.8rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model and answers (cached so this only runs once per session start)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading chatbot model...")
def get_model():
    return load_or_train_model()


@st.cache_data(show_spinner=False)
def get_answers():
    df = pd.read_csv(ANSWERS_PATH)
    return dict(zip(df["category"], df["answer"]))


try:
    model = get_model()
    answers = get_answers()
    load_error = None
except Exception as exc:  # pragma: no cover - defensive UI guard
    model = None
    answers = {}
    load_error = str(exc)


def classify_question(question: str):
    """
    Runs the trained ML pipeline on a single question and returns
    (predicted_category, confidence, response_time_seconds).

    If the top confidence is below CONFIDENCE_THRESHOLD, the category is
    returned as None so the caller shows the fallback message instead.
    """
    start_time = time.perf_counter()
    probabilities = model.predict_proba([question])[0]
    classes = model.classes_
    best_index = probabilities.argmax()
    confidence = probabilities[best_index]
    predicted_category = classes[best_index]
    elapsed = time.perf_counter() - start_time

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence, elapsed
    return predicted_category, confidence, elapsed


def get_bot_reply(question: str) -> str:
    """Handles empty/invalid input and returns the final text to display."""
    question = (question or "").strip()

    if not question:
        return "Please type a question so I can try to help you."

    if model is None:
        return (
            "Sorry, the chatbot's model could not be loaded right now. "
            "Please try again later."
        )

    category, confidence, _ = classify_question(question)

    if category is None:
        return FALLBACK_MESSAGE

    return answers.get(category, FALLBACK_MESSAGE)


# ---------------------------------------------------------------------------
# Session state setup
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "text": (
                "Hi! I'm the Student Help Desk chatbot. Ask me about "
                "enrollment, school requirements, or class schedules."
            ),
        }
    ]

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown("# Student Help Desk")
st.markdown(
    '<div class="subtitle">Ask a question about enrollment, school '
    "requirements, or class schedules.</div>",
    unsafe_allow_html=True,
)

if load_error:
    st.error(f"The chatbot model failed to load: {load_error}")

# ---------------------------------------------------------------------------
# Supported categories with clickable example questions
# ---------------------------------------------------------------------------

st.markdown("#### Supported categories")
cat_cols = st.columns(3)
category_names = list(EXAMPLE_QUESTIONS.keys())

for col, category in zip(cat_cols, category_names):
    with col:
        st.markdown(
            f'<div class="category-card"><strong>{category}</strong></div>',
            unsafe_allow_html=True,
        )
        with st.popover(f"Examples", use_container_width=True):
            st.caption(f"Try one of these {category.lower()} questions:")
            for example in EXAMPLE_QUESTIONS[category]:
                if st.button(example, key=f"example_{category}_{example}"):
                    st.session_state.pending_question = example

st.divider()

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------

st.markdown("#### Chat")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-bubble-user">{msg["text"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-bot">{msg["text"]}</div>',
                unsafe_allow_html=True,
            )


def submit_question(question_text: str):
    """Adds the user's question and the bot's reply to the chat history."""
    question_text = (question_text or "").strip()
    st.session_state.messages.append({"role": "user", "text": question_text or "(empty message)"})

    with st.spinner("Thinking..."):
        reply = get_bot_reply(question_text)

    st.session_state.messages.append({"role": "bot", "text": reply})


# Handle a question triggered by clicking an example button above.
if st.session_state.pending_question:
    submit_question(st.session_state.pending_question)
    st.session_state.pending_question = None
    st.rerun()

# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------

with st.form(key="question_form", clear_on_submit=True):
    user_input = st.text_input(
        "Type your question here",
        placeholder="e.g. How do I enroll?",
        label_visibility="collapsed",
    )
    submit_col, clear_col = st.columns([3, 1])
    with submit_col:
        send_clicked = st.form_submit_button("Send", use_container_width=True)
    with clear_col:
        clear_clicked = st.form_submit_button("New chat", use_container_width=True)

if send_clicked:
    submit_question(user_input)
    st.rerun()

if clear_clicked:
    st.session_state.messages = [
        {
            "role": "bot",
            "text": (
                "Hi! I'm the Student Help Desk chatbot. Ask me about "
                "enrollment, school requirements, or class schedules."
            ),
        }
    ]
    st.rerun()

# ---------------------------------------------------------------------------
# Footer / research transparency note
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="research-note">Research prototype. Please verify official '
    "school information with the appropriate school office.</div>",
    unsafe_allow_html=True,
)
