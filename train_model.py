"""
train_model.py
----------------
This file contains the machine learning pipeline for the Student FAQ Chatbot.

What it does, in plain terms:
1. Loads the labeled training questions from faq_training_data.csv, plus every
   wording (main question + alternative phrasings) from faq_knowledge_base.csv,
   so the classifier also learns the vocabulary of the dropdown questions.
2. Converts each question's text into numbers using TF-IDF
   (this measures how important each word is in a question).
3. Trains a Multinomial Naive Bayes classifier to learn which category
   (Enrollment, School Requirements, or Class Schedules) a question belongs to.
4. Saves the trained pipeline to disk (model.pkl) so the Streamlit app
   can load it instantly without retraining every time it starts.

This module is imported by both app.py (the web app) and evaluate_model.py
(the research evaluation script), so the exact same pipeline logic is used
everywhere -- there is only one place where the model is defined.
"""

import hashlib
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

from faq_engine import (
    KNOWLEDGE_BASE_PATH,
    knowledge_base_training_rows,
    load_knowledge_base,
)

# A fixed random seed makes results reproducible -- re-running training
# or evaluation will always split/shuffle data the same way.
RANDOM_SEED = 42

# File locations. Using paths relative to this file (not the current working
# directory) means the app works the same way locally and when deployed.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_PATH = os.path.join(BASE_DIR, "faq_training_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def load_training_data(
    path: str = TRAINING_DATA_PATH,
    include_knowledge_base: bool = True,
    kb_path: str = KNOWLEDGE_BASE_PATH,
) -> pd.DataFrame:
    """
    Loads and lightly validates the labeled dataset used by the classifier.

    Sources:
      1. faq_training_data.csv        (the original labeled questions)
      2. faq_knowledge_base.csv       (each FAQ's main question and alternative
                                       phrasings, labeled with the FAQ's category)

    Returns a DataFrame with columns: 'question', 'category'.
    Rows with missing or empty questions/categories are dropped, extra
    whitespace is trimmed, and exact duplicates (ignoring upper/lower case) are
    removed. If the same question text is labeled with two different
    categories, a ValueError is raised instead of silently picking one.
    """
    df = pd.read_csv(path)

    # Basic validation: make sure the expected columns exist.
    expected_columns = {"question", "category"}
    if not expected_columns.issubset(set(df.columns)):
        raise ValueError(
            f"Training data must contain columns {expected_columns}, "
            f"but found {set(df.columns)}"
        )
    df = df[["question", "category"]]

    if include_knowledge_base and os.path.exists(kb_path):
        kb_rows = knowledge_base_training_rows(load_knowledge_base(kb_path))
        df = pd.concat([df, kb_rows], ignore_index=True)

    # Clean up whitespace and drop any rows with missing values.
    df = df.dropna(subset=["question", "category"])
    df["question"] = df["question"].astype(str).str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df = df[(df["question"] != "") & (df["category"] != "")]

    # Remove exact duplicates, and refuse contradictory labels.
    key = df["question"].str.lower()
    labels_per_question = df.groupby(key)["category"].nunique()
    conflicts = labels_per_question[labels_per_question > 1].index.tolist()
    if conflicts:
        raise ValueError(
            f"These questions appear with more than one category: {conflicts}"
        )
    df = df.loc[~key.duplicated()].reset_index(drop=True)

    return df


def data_fingerprint(df: pd.DataFrame) -> str:
    """Short hash of the training data, used to detect an out-of-date model.pkl."""
    joined = "\n".join(sorted(f"{q}\t{c}" for q, c in zip(df["question"], df["category"])))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def build_pipeline() -> Pipeline:
    """
    Builds (but does not train) the scikit-learn pipeline:
    TF-IDF Vectorizer -> Multinomial Naive Bayes classifier.

    Using a Pipeline means the same vectorizer used for training is
    automatically reused for every future prediction, which prevents
    a common bug where training and prediction use different feature sets.
    """
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),  # single words AND two-word phrases
            ),
        ),
        (
            "classifier",
            MultinomialNB(),
        ),
    ])
    return pipeline


def train_and_save_model(
    data_path: str = TRAINING_DATA_PATH, model_path: str = MODEL_PATH
) -> Pipeline:
    """
    Trains the pipeline on the FULL labeled dataset and saves it to disk.

    This is used by the live chatbot app, where we want the best possible
    model trained on all available labeled examples. For a fair research
    evaluation with a held-out test set, see evaluate_model.py instead --
    that script trains on a separate 80% split so test data is never seen
    during training.
    """
    df = load_training_data(data_path)

    if df.empty:
        raise ValueError("Training data is empty after cleaning. Cannot train model.")

    pipeline = build_pipeline()
    pipeline.fit(df["question"], df["category"])

    # Remember which data this model was trained on, so load_or_train_model()
    # can tell when model.pkl is out of date.
    pipeline.data_fingerprint_ = data_fingerprint(df)

    joblib.dump(pipeline, model_path)
    print(f"Model trained on {len(df)} examples and saved to: {model_path}")

    return pipeline


def load_or_train_model(
    data_path: str = TRAINING_DATA_PATH, model_path: str = MODEL_PATH
) -> Pipeline:
    """
    Loads a previously saved model if one exists AND it was trained on the
    current data; otherwise trains a new one and saves it. This is what the
    Streamlit app calls on startup, so the model is always available --
    including on a fresh deployment where model.pkl doesn't exist yet -- and a
    stale model.pkl (from before the CSV files changed) is never used.
    """
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            current = data_fingerprint(load_training_data(data_path))
            if getattr(model, "data_fingerprint_", None) == current:
                return model
        except Exception:
            # If the saved file is corrupted or incompatible, retrain instead
            # of crashing the app.
            pass

    return train_and_save_model(data_path, model_path)


if __name__ == "__main__":
    # Running "python train_model.py" directly retrains and saves the model.
    train_and_save_model()
