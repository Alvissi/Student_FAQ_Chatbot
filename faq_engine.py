"""
faq_engine.py
-------------
FAQ knowledge base + answer retrieval for the Student FAQ Chatbot.

This module has NO Streamlit code on purpose: app.py (the web app) and
evaluate_model.py (the research evaluation) both import it, so the exact same
answer-selection logic is used everywhere and can be tested without a browser.

The two-stage pipeline
----------------------
Stage 1 - INTENT CLASSIFICATION (machine learning, see train_model.py)
    The trained TF-IDF + Multinomial Naive Bayes pipeline predicts which of the
    three categories a question belongs to (Enrollment / School Requirements /
    Class Schedules) and how confident it is.

Stage 2 - FAQ MATCHING (TF-IDF + cosine similarity, this file)
    Inside the predicted category only, the question is compared against every
    FAQ entry's wording (its main question plus its alternative phrasings). The
    most similar entry wins, and that entry's own predefined answer is returned.

If either stage is not confident enough, the bot returns a fallback message
instead of guessing. No answer is ever generated: every answer is read from
faq_knowledge_base.csv.

IMPORTANT: the classifier only predicts the CATEGORY. It does not classify the
45 individual FAQs. Which specific FAQ answers a question is decided by Stage 2.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "faq_knowledge_base.csv")

CATEGORY_ORDER = ["Enrollment", "School Requirements", "Class Schedules"]

# ---------------------------------------------------------------------------
# Thresholds (see README section "How the thresholds were chosen")
# ---------------------------------------------------------------------------

# Stage 1: below this classifier probability the question is "not recognized".
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.40

# Stage 2: below this similarity the best FAQ is considered too weak a match
# to answer with.
RETRIEVAL_SCORE_THRESHOLD = 0.30

# Stage 2: if the best match is not a strong one and the runner-up FAQ scores
# within this margin of it, the question is treated as ambiguous and the bot
# offers both candidates instead of guessing.
STRONG_MATCH_SCORE = 0.75
AMBIGUITY_MARGIN = 0.03

FALLBACK_MESSAGE = (
    "I'm sorry, I couldn't identify your question. Please ask about "
    "enrollment, school requirements, or class schedules, or pick a "
    "question from the list above."
)

SAMPLE_NOTE = (
    "Sample guidance only. Please confirm school-specific details with the "
    "appropriate school office."
)

LIST_SEPARATOR = "|"


# ---------------------------------------------------------------------------
# Knowledge base loading and validation
# ---------------------------------------------------------------------------

def _split_list(value) -> List[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [p.strip() for p in str(value).split(LIST_SEPARATOR) if p.strip()]


def load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> pd.DataFrame:
    """
    Loads and validates faq_knowledge_base.csv.

    Columns: faq_id, category, question, alternative_phrasings (| separated),
             answer (a literal \\n marks a line break), related_faq_ids (| separated).

    Returns a DataFrame where 'alternative_phrasings' and 'related_faq_ids' are
    Python lists and 'answer' contains real line breaks. Raises ValueError with
    a clear message if the file is malformed, so a bad edit is caught
    immediately instead of silently producing wrong answers.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = ["faq_id", "category", "question", "alternative_phrasings",
                "answer", "related_faq_ids"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Knowledge base is missing columns: {missing}")

    for col in ["faq_id", "category", "question", "answer"]:
        df[col] = df[col].str.strip()

    # Drop fully blank rows (can appear after editing in a spreadsheet program).
    df = df[(df["faq_id"] != "") | (df["question"] != "")].reset_index(drop=True)

    if df.empty:
        raise ValueError("Knowledge base is empty.")

    bad_cat = sorted(set(df["category"]) - set(CATEGORY_ORDER))
    if bad_cat:
        raise ValueError(f"Unknown categories in knowledge base: {bad_cat}")

    for col in ["faq_id", "question", "answer"]:
        if (df[col] == "").any():
            raise ValueError(f"Knowledge base has an empty '{col}' value.")

    if df["faq_id"].duplicated().any():
        raise ValueError(
            f"Duplicate faq_id values: {df.loc[df['faq_id'].duplicated(), 'faq_id'].tolist()}"
        )

    q_norm = df["question"].str.lower()
    if q_norm.duplicated().any():
        raise ValueError(
            f"Duplicate questions: {df.loc[q_norm.duplicated(), 'question'].tolist()}"
        )

    a_norm = df["answer"].str.lower()
    if a_norm.duplicated().any():
        raise ValueError(
            "Two FAQ entries share an identical answer: "
            f"{df.loc[a_norm.duplicated(), 'faq_id'].tolist()}"
        )

    df["answer"] = df["answer"].str.replace("\\n", "\n", regex=False)
    df["alternative_phrasings"] = df["alternative_phrasings"].apply(_split_list)
    df["related_faq_ids"] = df["related_faq_ids"].apply(_split_list)

    known_ids = set(df["faq_id"])
    for _, row in df.iterrows():
        unknown = [r for r in row["related_faq_ids"] if r not in known_ids]
        if unknown:
            raise ValueError(f"{row['faq_id']} lists unknown related ids: {unknown}")
        if row["faq_id"] in row["related_faq_ids"]:
            raise ValueError(f"{row['faq_id']} lists itself as a related question.")

    return df


def get_category_questions(kb: pd.DataFrame) -> Dict[str, List[str]]:
    """
    {category: [main FAQ questions]} in knowledge-base order, without
    duplicates. This feeds the dropdown, so the dropdown can never show a
    question that has no answer.
    """
    result = {}
    for category in CATEGORY_ORDER:
        questions = kb.loc[kb["category"] == category, "question"].tolist()
        result[category] = list(dict.fromkeys(questions))
    return result


def knowledge_base_training_rows(kb: pd.DataFrame) -> pd.DataFrame:
    """
    Every wording in the knowledge base (main question + alternative
    phrasings) with its category, so the intent classifier also learns the
    vocabulary of the dropdown questions. Returns columns: question, category.
    """
    rows = []
    for _, r in kb.iterrows():
        for text in [r["question"]] + list(r["alternative_phrasings"]):
            rows.append({"question": text, "category": r["category"]})
    return pd.DataFrame(rows, columns=["question", "category"])


# ---------------------------------------------------------------------------
# Stage 2: FAQ matching inside a category
# ---------------------------------------------------------------------------

class FAQMatcher:
    """
    TF-IDF + cosine similarity over the knowledge base wordings.

    Each FAQ entry has several wordings (main question + alternatives). An
    entry's score for a user question is the highest similarity among its
    wordings, so rephrased questions can still match the right entry.

    The similarity is the average of two cosine similarities:
      * word TF-IDF (1-2 word phrases, English stop words removed), which
        rewards shared meaningful words, and
      * character TF-IDF (3-5 character pieces inside words), which lets
        "enrolling" match "enroll" and "transferring" match "transferee".
    Both are plain TF-IDF + cosine similarity; nothing is generated.
    """

    WORD_WEIGHT = 0.5

    def __init__(self, kb: pd.DataFrame):
        self.kb = kb.reset_index(drop=True)

        phrases, owners, categories = [], [], []
        for i, row in self.kb.iterrows():
            for text in [row["question"]] + list(row["alternative_phrasings"]):
                phrases.append(text)
                owners.append(i)
                categories.append(row["category"])

        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        self.word_matrix = self.word_vectorizer.fit_transform(phrases)
        self.char_matrix = self.char_vectorizer.fit_transform(phrases)
        self.owners = np.array(owners)
        self.categories = np.array(categories)

    def rank(self, text: str, category: str):
        """
        Returns [(kb_row_index, score), ...] for the FAQ entries of `category`,
        best first. Scores are between 0 and 1 (1.0 = identical wording).
        """
        word_sims = linear_kernel(
            self.word_vectorizer.transform([text]), self.word_matrix
        ).ravel()
        char_sims = linear_kernel(
            self.char_vectorizer.transform([text]), self.char_matrix
        ).ravel()
        sims = self.WORD_WEIGHT * word_sims + (1 - self.WORD_WEIGHT) * char_sims

        mask = self.categories == category
        best: Dict[int, float] = {}
        for owner, score in zip(self.owners[mask], sims[mask]):
            if score > best.get(owner, -1.0):
                best[owner] = float(score)
        return sorted(best.items(), key=lambda item: -item[1])


# ---------------------------------------------------------------------------
# Full pipeline: classify -> match -> answer (or fall back)
# ---------------------------------------------------------------------------

@dataclass
class BotResult:
    status: str                      # answered | ambiguous | low_confidence | weak_match | empty
    reply_text: str                  # plain text; the UI escapes it before display
    category: Optional[str] = None   # predicted category (None if not confident)
    classifier_confidence: float = 0.0
    faq_id: Optional[str] = None
    matched_question: Optional[str] = None
    match_score: float = 0.0
    related_questions: List[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.status == "answered"


def answer_question(text: str, classifier, matcher: FAQMatcher) -> BotResult:
    """
    The single entry point used by BOTH the dropdown and the free-text box.

    1. Classify the question's category with the trained ML pipeline.
    2. Find the most similar FAQ entry inside that category.
    3. Return that entry's own answer, or a fallback if either step is weak.
    """
    text = (text or "").strip()
    if not text:
        return BotResult(status="empty",
                         reply_text="Please type a question so I can try to help you.")

    # --- Stage 1: category -------------------------------------------------
    probabilities = classifier.predict_proba([text])[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    category = str(classifier.classes_[best_index])

    if confidence < CLASSIFIER_CONFIDENCE_THRESHOLD:
        return BotResult(status="low_confidence", reply_text=FALLBACK_MESSAGE,
                         classifier_confidence=confidence)

    # --- Stage 2: FAQ entry inside that category ----------------------------
    ranked = matcher.rank(text, category)
    if not ranked or ranked[0][1] < RETRIEVAL_SCORE_THRESHOLD:
        top_score = ranked[0][1] if ranked else 0.0
        return BotResult(status="weak_match", reply_text=FALLBACK_MESSAGE,
                         category=category, classifier_confidence=confidence,
                         match_score=top_score)

    best_idx, best_score = ranked[0]
    kb = matcher.kb

    if (len(ranked) > 1 and best_score < STRONG_MATCH_SCORE
            and best_score - ranked[1][1] < AMBIGUITY_MARGIN):
        first = kb.loc[best_idx, "question"]
        second = kb.loc[ranked[1][0], "question"]
        reply = (
            "Your question could match more than one topic. Did you mean:\n"
            f"- {first}\n- {second}\n"
            "Please pick one from the list above, or add more detail to your question."
        )
        return BotResult(status="ambiguous", reply_text=reply, category=category,
                         classifier_confidence=confidence, match_score=best_score,
                         related_questions=[first, second])

    entry = kb.loc[best_idx]
    id_to_question = dict(zip(kb["faq_id"], kb["question"]))
    related = [id_to_question[r] for r in entry["related_faq_ids"]]

    parts = []
    if best_score < STRONG_MATCH_SCORE:
        # Not a near-exact wording: show which FAQ was matched so the student
        # can immediately tell if the bot understood the question differently.
        parts.append(f'This looks closest to: "{entry["question"]}"')
    parts.append(entry["answer"])
    if related:
        parts.append("You might also ask:\n" + "\n".join(f"- {q}" for q in related))
    parts.append(SAMPLE_NOTE)

    return BotResult(
        status="answered",
        reply_text="\n\n".join(parts),
        category=category,
        classifier_confidence=confidence,
        faq_id=str(entry["faq_id"]),
        matched_question=str(entry["question"]),
        match_score=best_score,
        related_questions=related,
    )
