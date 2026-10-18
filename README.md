# Student FAQ Chatbot (Machine Learning-Based)

**Project title:** Development and Evaluation of a Machine Learning-Based Student FAQ
Chatbot for Selected Student Inquiries

A simple Streamlit web app that uses a real machine learning model (TF-IDF +
Multinomial Naive Bayes, from scikit-learn) to classify a student's question into one
of three categories — **Enrollment**, **School Requirements**, or **Class Schedules**
— and returns a predefined sample answer for that category. This is a research
prototype built for a college-level Artificial Intelligence 1 case study. It is
**not** connected to any real school system and does not use any paid API, external
LLM, or API key.

---

## 1. Project structure

```
faq_chatbot/
├── app.py                   # Streamlit web application (the chatbot UI)
├── train_model.py           # ML pipeline: load data, TF-IDF + Naive Bayes, save model
├── evaluate_model.py        # Research evaluation: stratified 80/20 split + metrics
├── faq_training_data.csv    # 90 labeled example questions (30 per category)
├── faq_answers.csv          # Predefined sample answer for each category
├── requirements.txt         # Python dependencies
├── .gitignore
└── README.md
```

`model.pkl` (the saved trained model) is **not** included in the repository — it is
generated automatically the first time the app or the training script runs, and is
listed in `.gitignore`. This keeps the repository small and guarantees the model is
always trained from the current CSV data, including on a fresh deployment.

## 2. How the classifier works (short explanation)

1. **TF-IDF Vectorizer** converts each question into a vector of numbers that
   represent how important each word (and word pair) is in that question, relative
   to the whole dataset.
2. **Multinomial Naive Bayes** is trained on these vectors, using the 90 labeled
   example questions, to learn which words are associated with each of the three
   categories.
3. When a student types a new question, the same TF-IDF transformation is applied,
   and the trained classifier predicts the most likely category, along with a
   confidence score (the model's predicted probability for that category).
4. If the top confidence score is **below 0.40**, the app does not trust the
   prediction and instead shows a fallback message asking the student to rephrase or
   try an example question. This threshold was chosen by testing clearly unrelated
   questions (which scored confidence ≈0.33, close to random guessing among 3
   categories) against genuine, even reworded, in-scope questions (which scored
   0.43 and above).
5. Once a category is predicted with enough confidence, the app looks up the
   matching **predefined sample answer** from `faq_answers.csv` — the model only
   classifies intent, it never generates or invents an answer.

No if-else keyword matching is used for classification — the category decision is
made entirely by the trained machine learning model.

---

## 3. Installing dependencies

You need Python 3.9 or later installed. From inside the `faq_chatbot` folder, run:

```bash
pip install -r requirements.txt
```

This installs Streamlit, scikit-learn, pandas, and joblib.

## 4. Running the app locally

From inside the `faq_chatbot` folder, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) — open it in your
browser. On the very first run, the app will automatically train the model from
`faq_training_data.csv` and save it as `model.pkl` (you'll briefly see a "Loading
chatbot model..." spinner). Every run after that loads the saved model instantly,
unless `model.pkl` is deleted or the CSV changes in a way that requires retraining.

## 5. Training the model manually (optional)

You do not have to do this separately — the app trains the model automatically if
`model.pkl` doesn't exist. But if you've changed `faq_training_data.csv` and want to
force a fresh model file, run:

```bash
python train_model.py
```

This loads all 90 labeled examples, trains the TF-IDF + Naive Bayes pipeline on the
**full** dataset, and saves it to `model.pkl`. (The app always trains on the full
dataset, because in production we want to use every labeled example available — this
is different from the evaluation script below, which intentionally holds out data.)

## 6. Evaluating the model (research metrics)

Run the separate evaluation script:

```bash
python evaluate_model.py
```

This script:

- Splits the 90 labeled examples into an **80% training set (72 examples)** and a
  **20% test set (18 examples)**, stratified so each category is proportionally
  represented in both splits.
- Trains a **fresh** model on the training split only — the test examples are never
  seen during this training.
- Predicts the category for each of the 18 held-out test questions.
- Reports: **Accuracy, Precision (macro), Recall (macro), F1-score (macro)**, a
  **confusion matrix**, a full per-category classification report, and the average
  prediction response time in milliseconds.
- Uses a fixed random seed (`random_state=42`, defined once in `train_model.py`) for
  the train/test split, so running it again reproduces the same split and the same
  results.

### Interpreting the results

- **Accuracy** — the percentage of the 18 test questions the model classified into
  the correct category.
- **Precision (macro)** — of all the times the model predicted a given category,
  what fraction were actually correct, averaged equally across all three categories
  (so a smaller category isn't drowned out by a larger one — here all three are
  equal size anyway).
- **Recall (macro)** — of all the questions that truly belonged to a given category,
  what fraction the model correctly found, again averaged equally across categories.
- **F1-score (macro)** — the harmonic mean of precision and recall, a single balanced
  summary number.
- **Confusion matrix** — a grid showing, for each actual category (rows), how many
  test questions were predicted into each category (columns). Numbers on the
  diagonal are correct predictions; numbers off the diagonal show which categories
  get confused with each other.

**Important limitation to report in your paper:** this dataset has only 90 manually
written examples (18 in each test run). A model trained on such a small, manually
created dataset will not generalize as reliably as one trained on hundreds or
thousands of real student questions. Do not present the measured accuracy as proof
that the system would perform equally well on real, unseen student inquiries — report
it honestly as a result measured on this specific small dataset, and name this as a
limitation of the study.

---

## 7. Uploading to GitHub

1. Create a new repository on GitHub (e.g. `student-faq-chatbot`). Do **not**
   initialize it with a README if you already have files locally — or if you do,
   you'll merge it in step 4.
2. From inside the `faq_chatbot` folder on your computer, initialize git and make
   your first commit:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Student FAQ Chatbot"
   ```
3. Connect your local folder to the GitHub repository (replace the URL with your
   own repository's URL):
   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/student-faq-chatbot.git
   git branch -M main
   ```
4. Push your code:
   ```bash
   git push -u origin main
   ```
5. Refresh your GitHub repository page in the browser to confirm all files
   (`app.py`, `train_model.py`, `evaluate_model.py`, both CSV files,
   `requirements.txt`, `.gitignore`, and this `README.md`) were uploaded. `model.pkl`
   should **not** appear — that's expected, since it's excluded by `.gitignore` and
   will be generated automatically after deployment.

## 8. Deploying to Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your
   GitHub account.
2. Click **"New app"**.
3. Select your repository (`student-faq-chatbot`), the branch (`main`), and set the
   main file path to `app.py`.
4. Click **"Deploy"**. Streamlit Cloud will install everything listed in
   `requirements.txt` and start the app. The very first load will take slightly
   longer than normal because the model trains automatically from
   `faq_training_data.csv` at that point (no local-only files or API keys are
   needed — everything required is in the repository).
5. Once deployment finishes, Streamlit Cloud gives you a public URL, usually in the
   form `https://YOUR-APP-NAME.streamlit.app`.

**Do not claim the app is "deployed" in your paper or demonstration until you have
actually completed these steps and confirmed the public URL loads correctly.**

### Testing the deployed app using the public URL

1. Open the public URL in a fresh, private/incognito browser window (to simulate a
   new user with no cached session).
2. Confirm the welcome message appears and the three category cards are visible.
3. Click an example question under one of the categories and confirm a reply
   appears in a chat bubble.
4. Type a new question manually into the input box, click **Send**, and confirm the
   reply is relevant to the category.
5. Type an unrelated question (e.g. "What's the weather today?") and confirm the
   fallback message appears instead of a wrong category answer.
6. Click **"New chat"** and confirm the conversation resets to the welcome message.
7. Reload the page (or open it in a second browser tab) and confirm the app starts a
   fresh session without errors.

---

## 9. Manual test checklist (for final demonstration)

Use these scenarios to confirm the chatbot behaves correctly before presenting:

1. **Typical enrollment question** — e.g. "How do I enroll?" → should return the
   Enrollment sample answer.
2. **Differently worded enrollment question** — e.g. "How can I register for
   classes?" → should still return the Enrollment sample answer.
3. **School requirements question** — e.g. "What documents should I submit?" →
   should return the School Requirements sample answer.
4. **Class schedule question** — e.g. "Where can I see my class schedule?" → should
   return the Class Schedules sample answer.
5. **Empty input** — click Send with nothing typed → should show a gentle prompt to
   type a question, not an error or a crash.
6. **Unrelated question** — e.g. "What's the weather today?" → should show the
   fallback message, not an incorrect category answer.
7. **Spelling/wording variation** — e.g. "How do I enrol? (sic)" or "wat r skool
   requirements" → observe whether the model still classifies reasonably; note any
   failures as a limitation in your paper, since the training data doesn't include
   intentional misspellings.
8. **Question that could be confused between categories** — e.g. "What do I need to
   bring on enrollment day?" (touches both Enrollment and School Requirements) →
   observe which category the model picks and discuss this as an example of
   real-world ambiguity in your paper.
9. **New session / page refresh** — reload the page or open a new browser tab →
   chat history should reset to a fresh welcome message without errors.
10. **Dependency check for deployment** — confirm `requirements.txt` installs
    cleanly in a brand-new environment (e.g. a fresh virtual environment or the
    Streamlit Cloud build log) with no missing-package errors.

---

## 10. Known limitations (for your research paper)

- The dataset is small (90 examples total, manually written by the developer), so
  measured accuracy may not reflect performance on real, unseen student phrasing,
  slang, or typos.
- The three categories are intentionally narrow in scope; questions outside
  Enrollment, School Requirements, and Class Schedules are expected to fall back,
  by design.
- Answers are fixed, clearly labeled **sample** answers, not verified official school
  policy — a real deployment would need the answers replaced with content confirmed
  by the actual school office.
- The 0.40 confidence threshold for the fallback was chosen empirically from a small
  set of manual test questions, not optimized through a separate validation
  procedure — this is a reasonable judgment call for a one-week prototype, but is
  worth naming explicitly as a limitation.
