"""
evaluate_model.py
------------------
Research evaluation script for the Student FAQ Chatbot's intent classifier.

This script is SEPARATE from the live chatbot (app.py) on purpose: for a
fair research evaluation, the model must be trained only on a training
split and tested on examples it has never seen. app.py, by contrast,
trains on the FULL dataset because in production we want to use every
labeled example we have.

What this script does:
1. Loads the labeled dataset (faq_training_data.csv).
2. Splits it into ~80% training / ~20% testing, STRATIFIED by category
   (so each category is proportionally represented in both splits).
3. Trains a fresh TF-IDF + Multinomial Naive Bayes pipeline on the
   training split only.
4. Predicts categories for the held-out test split.
5. Reports accuracy, macro-averaged precision/recall/F1, and a confusion
   matrix, using scikit-learn's real metric calculations (nothing here is
   hardcoded or invented).
6. Measures average prediction response time as a simple performance metric.

Run with:  python evaluate_model.py
"""

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from train_model import load_training_data, build_pipeline, RANDOM_SEED

TEST_SIZE = 0.20  # ~20% of the data held out for testing


def run_evaluation():
    print("=" * 70)
    print("STUDENT FAQ CHATBOT - MODEL EVALUATION")
    print("=" * 70)

    # 1. Load the full labeled dataset.
    df = load_training_data()
    print(f"\nTotal labeled examples loaded: {len(df)}")
    print("Examples per category:")
    print(df["category"].value_counts().to_string())

    if len(df) < 15:
        print(
            "\nWARNING: The dataset is very small. Evaluation results may not "
            "be statistically meaningful."
        )

    # 2. Stratified 80/20 split. Stratifying on 'category' ensures the
    #    proportion of each of the three categories is preserved in both
    #    the training set and the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        df["question"],
        df["category"],
        test_size=TEST_SIZE,
        stratify=df["category"],
        random_state=RANDOM_SEED,
    )

    print(f"\nTraining examples: {len(X_train)}")
    print(f"Testing examples:  {len(X_test)}")
    print(f"Random seed used for the split: {RANDOM_SEED}")

    # 3. Train a FRESH pipeline on the training split ONLY.
    #    This pipeline is independent from the one saved by app.py/train_model.py
    #    for production use -- it exists only to produce an honest evaluation.
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Predict on the held-out test split, while timing the predictions
    #    to report response time as a performance metric.
    start_time = time.perf_counter()
    y_pred = pipeline.predict(X_test)
    total_time = time.perf_counter() - start_time
    avg_response_time_ms = (total_time / len(X_test)) * 1000

    # 5. Compute metrics. Macro averaging treats all three categories
    #    as equally important, regardless of how many examples each has.
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    labels_sorted = sorted(df["category"].unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print("\n" + "-" * 70)
    print("RESULTS (measured on the held-out 20% test set)")
    print("-" * 70)
    print(f"Accuracy:            {accuracy:.4f}")
    print(f"Precision (macro):   {precision_macro:.4f}")
    print(f"Recall (macro):      {recall_macro:.4f}")
    print(f"F1-score (macro):    {f1_macro:.4f}")
    print(f"Avg. response time:  {avg_response_time_ms:.3f} ms per question")

    print("\nConfusion matrix")
    print("Rows = actual category, Columns = predicted category")
    header = " " * 24 + "".join(f"{label[:18]:>20}" for label in labels_sorted)
    print(header)
    for i, label in enumerate(labels_sorted):
        row = "".join(f"{cm[i][j]:>20}" for j in range(len(labels_sorted)))
        print(f"{label[:22]:<24}{row}")

    print("\nFull classification report (per category):")
    print(classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0))

    print("-" * 70)
    print(
        "NOTE: This dataset is small and manually created for a one-week "
        "college research case study. Results on this test set may not "
        "generalize to real, unseen student questions phrased very "
        "differently from the training examples. A larger and more diverse "
        "dataset, ideally collected from real student inquiries, would be "
        "needed to make stronger claims about real-world performance."
    )
    print("=" * 70)

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "avg_response_time_ms": avg_response_time_ms,
    }


if __name__ == "__main__":
    run_evaluation()
