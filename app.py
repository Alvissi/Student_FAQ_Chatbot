"""
app.py
------
Student FAQ Chatbot -- Streamlit web application.

This is the user-facing part of the research project. A student can either
pick a predefined question from a searchable dropdown or type their own
question. BOTH routes go through exactly the same pipeline (faq_engine.py):

  1. A trained machine learning model (TF-IDF + Multinomial Naive Bayes)
     classifies the question into one of three categories.
  2. TF-IDF + cosine similarity finds the most relevant FAQ entry inside that
     category, and that entry's own predefined answer is returned.
  3. If either step is not confident enough, a fallback message is shown.

Run locally with:  streamlit run app.py
"""

import html

import streamlit as st

from faq_engine import (
    CATEGORY_ORDER,
    FAQMatcher,
    answer_question,
    get_category_questions,
    load_knowledge_base,
)
from train_model import load_or_train_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# The confidence thresholds and fallback message live in faq_engine.py, so the
# app and the evaluation script always use identical settings.

# The dropdown's QUESTIONS are never hardcoded here -- they come from
# faq_knowledge_base.csv (see get_category_questions in faq_engine.py), so the
# dropdown can never show a question that has no answer.

CATEGORY_DESCRIPTIONS = {
    "Enrollment": "Questions about enrolling, registering, or re-enrolling for classes.",
    "School Requirements": "Questions about documents and papers needed for enrollment.",
    "Class Schedules": "Questions about viewing or checking your class schedule.",
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


@st.cache_resource(show_spinner=False)
def get_matcher():
    return FAQMatcher(load_knowledge_base())


try:
    model = get_model()
    matcher = get_matcher()
    load_error = None
except Exception as exc:  # pragma: no cover - defensive UI guard
    model = None
    matcher = None
    load_error = str(exc)


def get_bot_reply(question: str) -> str:
    """
    Single answer pipeline shared by the dropdown and the free-text box:
    classify the category, find the best FAQ in it, return that FAQ's answer.
    """
    question = (question or "").strip()

    if not question:
        return "Please type a question so I can try to help you."

    if model is None or matcher is None:
        return (
            "Sorry, the chatbot's model could not be loaded right now. "
            "Please try again later."
        )

    return answer_question(question, model, matcher).reply_text


def to_bubble_html(text: str) -> str:
    """Escapes text for safe display inside a chat bubble and keeps line breaks."""
    return html.escape(text).replace("\n", "<br>")


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
# Supported categories + one searchable dropdown of predefined questions
# ---------------------------------------------------------------------------

st.markdown("#### Supported categories")

if "active_category" not in st.session_state:
    st.session_state.active_category = CATEGORY_ORDER[0]

# A counter baked into the dropdown's widget key. Bumping it creates a brand-new
# (unselected) dropdown on the next run, which is how the dropdown is cleared
# after "Ask Question" without touching an already-instantiated widget's state
# directly (which Streamlit does not allow).
if "dropdown_reset_counter" not in st.session_state:
    st.session_state.dropdown_reset_counter = 0

category_questions = (
    get_category_questions(matcher.kb) if matcher is not None else {}
)

cat_cols = st.columns(3)
for col, category in zip(cat_cols, CATEGORY_ORDER):
    with col:
        is_active = category == st.session_state.active_category
        if st.button(
            category,
            key=f"cat_btn_{category}",
            width="stretch",
            type="primary" if is_active else "secondary",
        ):
            if not is_active:
                st.session_state.active_category = category
                st.rerun()

active_category = st.session_state.active_category

st.markdown(
    f'<div class="category-card"><strong>{active_category}</strong><br>'
    f'<span style="color:#6B7280; font-size:0.9rem;">'
    f"{CATEGORY_DESCRIPTIONS.get(active_category, '')}</span></div>",
    unsafe_allow_html=True,
)

dropdown_options = category_questions.get(active_category, [])
# The category AND the reset counter are part of the key, so the options and
# the selection always belong to the category currently shown.
dropdown_key = f"faq_dropdown_{active_category}_{st.session_state.dropdown_reset_counter}"

# Search happens INSIDE the dropdown: the student clicks it and types, and the
# native widget filters the options. filter_mode="contains" gives a
# case-insensitive partial-text match (needs Streamlit 1.56+). On an older
# Streamlit the keyword doesn't exist, so we fall back to the widget's own
# default typing-search instead of crashing.
dropdown_kwargs = dict(
    options=dropdown_options,
    index=None,
    placeholder="Search or select a question...",
    label_visibility="collapsed",
    key=dropdown_key,
)
try:
    selected_question = st.selectbox(
        "Select a predefined question", filter_mode="contains", **dropdown_kwargs
    )
except TypeError:
    selected_question = st.selectbox("Select a predefined question", **dropdown_kwargs)

if st.button(
    "Ask Question",
    key="ask_question_btn",
    width="stretch",
    disabled=selected_question is None,
):
    st.session_state.pending_question = selected_question
    st.session_state.dropdown_reset_counter += 1
    st.rerun()

st.caption(
    "Type in the box above to search the questions. If nothing matches, "
    "type your question in the chat box below."
)

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
                f'<div class="chat-bubble-user">{to_bubble_html(msg["text"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="chat-bubble-bot">{to_bubble_html(msg["text"])}</div>',
                unsafe_allow_html=True,
            )


def submit_question(question_text: str):
    """Adds the user's question and the bot's reply to the chat history."""
    question_text = (question_text or "").strip()
    st.session_state.messages.append({"role": "user", "text": question_text or "(empty message)"})

    with st.spinner("Thinking..."):
        reply = get_bot_reply(question_text)

    st.session_state.messages.append({"role": "bot", "text": reply})


# Handle a question sent from the dropdown's "Ask Question" button above.
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
        send_clicked = st.form_submit_button("Send", width="stretch")
    with clear_col:
        clear_clicked = st.form_submit_button("New chat", width="stretch")

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
