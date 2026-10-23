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

7. (Part 2) Evaluates the FULL answer pipeline -- classifier + FAQ matching --
   on separate question sets that are never used for training or in the FAQ
   knowledge base: faq_dev_questions.csv (used once to choose thresholds) and
   faq_test_questions.csv (used only for reporting). This part reports
   CLASSIFICATION accuracy and FAQ ANSWER RELEVANCE separately, because they
   measure different things.

Run with:  python evaluate_model.py
"""

import os
import time
import numpy as np
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

from faq_engine import (
    AMBIGUITY_MARGIN,
    CLASSIFIER_CONFIDENCE_THRESHOLD,
    RETRIEVAL_SCORE_THRESHOLD,
    FAQMatcher,
    answer_question,
    load_knowledge_base,
)
from train_model import BASE_DIR, load_training_data, build_pipeline, RANDOM_SEED

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


DEV_QUESTIONS_PATH = os.path.join(BASE_DIR, "faq_dev_questions.csv")
TEST_QUESTIONS_PATH = os.path.join(BASE_DIR, "faq_test_questions.csv")


def _evaluate_question_set(name, path, classifier, matcher, kb, seen_wordings):
    """
    Runs the full pipeline on a labeled question set.
    expected_faq_id is a knowledge-base FAQ id for in-scope questions, or NONE
    for questions the bot should NOT answer (it should fall back instead).
    """
    qs = pd.read_csv(path)
    faq_category = dict(zip(kb["faq_id"], kb["category"]))

    overlap = [q for q in qs["question"] if q.strip().lower() in seen_wordings]
    print(f"\n{name}  ({len(qs)} questions, from {os.path.basename(path)})")
    if overlap:
        print(f"  WARNING: {len(overlap)} question(s) also appear in the training "
              f"data / knowledge base, so this set is NOT independent: {overlap}")

    in_scope = out_scope = 0
    cat_correct = faq_correct = wrong_answer = false_fallback = 0
    faq_correct_given_cat = cat_correct_total = 0
    oos_rejected = 0
    mistakes = []

    for question, expected in zip(qs["question"], qs["expected_faq_id"]):
        result = answer_question(question, classifier, matcher)

        if expected == "NONE":
            out_scope += 1
            if not result.answered:
                oos_rejected += 1
            else:
                mistakes.append((question, "should fall back", result.faq_id,
                                 result.classifier_confidence, result.match_score))
            continue

        in_scope += 1
        # Classification accuracy: raw top category, ignoring the fallback rule.
        raw_category = classifier.classes_[
            int(np.argmax(classifier.predict_proba([question])[0]))
        ]
        category_ok = raw_category == faq_category[expected]
        cat_correct += category_ok
        cat_correct_total += category_ok

        if result.answered and result.faq_id == expected:
            faq_correct += 1
            faq_correct_given_cat += category_ok
        elif result.answered:
            wrong_answer += 1
            mistakes.append((question, expected, result.faq_id,
                             result.classifier_confidence, result.match_score))
        else:
            false_fallback += 1
            mistakes.append((question, expected, f"({result.status})",
                             result.classifier_confidence, result.match_score))

    print(f"  In-scope questions: {in_scope}")
    print(f"    Category classification accuracy:    {cat_correct}/{in_scope} = {cat_correct / in_scope:.1%}")
    print(f"    Correct FAQ answer returned:         {faq_correct}/{in_scope} = {faq_correct / in_scope:.1%}")
    print(f"    Wrong FAQ answer returned:           {wrong_answer}/{in_scope} = {wrong_answer / in_scope:.1%}")
    print(f"    Fell back / asked to clarify:        {false_fallback}/{in_scope} = {false_fallback / in_scope:.1%}")
    if cat_correct_total:
        print(f"    Correct FAQ, given correct category: {faq_correct_given_cat}/{cat_correct_total} = {faq_correct_given_cat / cat_correct_total:.1%}")
    if out_scope:
        print(f"  Out-of-scope questions: {out_scope}")
        print(f"    Correctly refused (fallback):        {oos_rejected}/{out_scope} = {oos_rejected / out_scope:.1%}")
    if mistakes:
        print("  Questions that did not go as expected:")
        for q, exp, got, conf, score in mistakes:
            print(f"    - {q!r}\n        expected {exp}, got {got} "
                  f"(classifier conf {conf:.2f}, match score {score:.2f})")


def run_faq_pipeline_evaluation():
    print("\n" + "=" * 70)
    print("PART 2: FULL ANSWER PIPELINE (classifier + FAQ matching)")
    print("=" * 70)
    print(
        "Classification accuracy (did we pick the right CATEGORY?) is different\n"
        "from answer relevance (did we return the right FAQ ANSWER?). The ML\n"
        "classifier only predicts the category; the FAQ-matching stage picks the\n"
        "specific answer inside it, so both are reported below."
    )
    print(f"\nSettings: classifier confidence >= {CLASSIFIER_CONFIDENCE_THRESHOLD}, "
          f"match score >= {RETRIEVAL_SCORE_THRESHOLD}, ambiguity margin {AMBIGUITY_MARGIN}")

    kb = load_knowledge_base()
    df = load_training_data()
    # Same model the app serves: trained on the full labeled dataset. The
    # question sets below are separate files and are never part of it.
    classifier = build_pipeline()
    classifier.fit(df["question"], df["category"])
    matcher = FAQMatcher(kb)

    seen = set(df["question"].str.strip().str.lower())

    # A. Predefined (dropdown) questions: each should return its OWN answer.
    own = sum(
        answer_question(q, classifier, matcher).faq_id == fid
        for fid, q in zip(kb["faq_id"], kb["question"])
    )
    print(f"\nPredefined dropdown questions: {own}/{len(kb)} returned their own FAQ answer")
    print("  (These wordings are in the knowledge base, so this is a sanity check,\n"
          "   not a measure of generalization.)")

    # B. Custom questions the system has never seen.
    _evaluate_question_set("DEV SET (used once to choose the thresholds - optimistic)",
                           DEV_QUESTIONS_PATH, classifier, matcher, kb, seen)
    _evaluate_question_set("TEST SET (never used to tune anything)",
                           TEST_QUESTIONS_PATH, classifier, matcher, kb, seen)

    print("\n" + "-" * 70)
    print(
        "NOTE: These question sets are small (30 and 60 questions) and written by\n"
        "the developer, so the percentages above carry wide uncertainty and should\n"
        "not be presented as real-world performance. The Part 1 accuracy above is\n"
        "also optimistic: several rewordings of the same FAQ can land on both\n"
        "sides of the random 80/20 split, so the classifier sees near-duplicates\n"
        "of its test questions. The Part 2 test set (questions written separately,\n"
        "never used in training or in the knowledge base) is the more realistic\n"
        "estimate."
    )
    print("=" * 70)


if __name__ == "__main__":
    run_evaluation()
    run_faq_pipeline_evaluation()
