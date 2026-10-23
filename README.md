# Student FAQ Chatbot (Machine Learning-Based)

**Project title:** AskMate: Development and Evaluation of a Machine Learning-Based Student FAQ
Chatbot for Selected Student Inquiries

A simple Streamlit web app that uses a real machine learning model (TF-IDF +
Multinomial Naive Bayes, from scikit-learn) to classify a student's question into one
of three categories — **Enrollment**, **School Requirements**, or **Class Schedules** —
and then finds the most relevant predefined FAQ inside that category and returns **that
FAQ's own answer**. This is a research prototype built for a college-level Artificial
Intelligence 1 case study. It is **not** connected to any real school system and does not
use any paid API, external LLM, or API key.

---

## 0. What changed in this update

1. **One searchable dropdown.** The separate search box is gone. Pick a category, then
   click the single dropdown and type to filter its questions (partial, case-insensitive
   — `filter_mode="contains"`). **Ask Question** sends the selection through the chatbot.
2. **45 distinct FAQs with their own answers.** `faq_knowledge_base.csv` holds 15 FAQs per
   category. Every entry has a unique question, 4 alternative phrasings, its own answer
   (numbered steps where it is a procedure), and related follow-up questions. The loader
   rejects the file if two entries share an answer.
3. **Answers are now chosen per question, not per category.** Dropdown and typed questions
   go through the same two-stage pipeline (section 2). `faq_answers.csv` (one answer per
   category) is no longer used and can be deleted.
4. **Safer fallbacks.** A low-confidence category, a weak FAQ match, or a near-tie between two
   FAQs ends in a fallback or a "Did you mean…" instead of a guess.
5. **Fixes found along the way:** `model.pkl` is now retrained automatically when the CSV
   data changes (before, a stale `model.pkl` would silently be reused); chat bubbles now
   HTML-escape what the user types; `use_container_width` (deprecated by Streamlit) was
   replaced with `width="stretch"`.

**Dependency change:** `streamlit>=1.56.0` is required, because `filter_mode` was added in
Streamlit 1.56 (released March 31, 2026), which needs Python 3.10 or later. On an older
Streamlit the app still runs — it falls back to the dropdown's default typing search.

### Honest testing status

- **Verified by running code:** the knowledge base loader and its checks; model training; all
  225 FAQ wordings returning their own FAQ; the evaluation scripts (results in section 6); the
  stale-`model.pkl` detection; and the full `app.py` script flow (category switching,
  dropdown → Ask Question → reply, typed questions, fallback, empty input, "New chat",
  history, HTML escaping) executed against a *stand-in* for Streamlit.
- **NOT verified:** the real Streamlit interface. The development environment could not install
  Streamlit, so the dropdown's actual search-as-you-type behavior, the "No results" text it
  shows when nothing matches, mobile layout, and Streamlit Community Cloud deployment have
  **not** been tested. Please run the checklist in section 9 in a browser before presenting.

---

## 1. Project structure

```
faq_chatbot/
├── app.py                    # Streamlit web application (the chatbot UI)
├── faq_engine.py             # FAQ knowledge base loader + answer retrieval (no Streamlit code)
├── train_model.py            # ML pipeline: load data, TF-IDF + Naive Bayes, save model
├── evaluate_model.py         # Research evaluation: classifier split + full-pipeline test
├── faq_knowledge_base.csv    # 45 FAQs: question, alternative phrasings, answer, related FAQs
├── faq_training_data.csv     # 90 original labeled example questions (30 per category)
├── faq_dev_questions.csv     # 30 held-out questions used to choose thresholds
├── faq_test_questions.csv    # 60 held-out questions used only for reporting
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

`model.pkl` (the saved trained model) is **not** included in the repository — it is
generated automatically the first time the app or the training script runs, and is
listed in `.gitignore`. If it exists but was trained on older data, it is detected and
retrained automatically.

### Editing the FAQ data

`faq_knowledge_base.csv` columns: `faq_id`, `category`, `question`,
`alternative_phrasings` (separated by `|`), `answer` (type a literal `\n` for a line
break), `related_faq_ids` (separated by `|`). The dropdown, the answers, the matcher and
the classifier's training data all read this one file, so they cannot drift apart. When you
add or edit an entry, run `python train_model.py` and `python evaluate_model.py`.

## 2. How a question is answered (two stages)

**Stage 1 — intent classification (machine learning).**
1. A **TF-IDF vectorizer** turns the question into numbers (word and word-pair importance).
2. **Multinomial Naive Bayes**, trained on 273 labeled questions (the 90 original ones plus
   every main question and alternative phrasing from the knowledge base, duplicates removed),
   predicts the category and a confidence score.
3. If confidence is **below 0.40**, the bot shows the fallback message.

**Stage 2 — FAQ matching (TF-IDF + cosine similarity).**
4. Inside the predicted category only, the question is compared with every FAQ's wordings.
   The score is the average of a word-level and a character-level TF-IDF cosine similarity
   (the character level lets "enrolling" match "enroll"). An FAQ's score is the best score
   among its wordings.
5. If the best score is **below 0.30**, the bot shows the fallback message. If the top two FAQs
   are within **0.03** of each other and the best is not a near-exact match, the bot asks
   "Did you mean…" with both candidates.
6. Otherwise the bot returns that FAQ's predefined answer, plus related questions and a
   "sample guidance" note. If the match is not near-exact, the reply first shows *which*
   FAQ it matched ("This looks closest to: …") so a wrong match is easy to spot.

The classifier only predicts the **category**. It does **not** independently classify the 45
individual FAQs; stage 2 picks the specific answer. No answer is ever generated — every reply
is text stored in `faq_knowledge_base.csv`. No if-else keyword rules are used for the category.

### How the thresholds were chosen

- **0.40 classifier confidence** (unchanged): with 3 categories, clearly off-topic questions
  usually score ≈0.34 (near random guessing). On the 65 in-scope dev and test questions, the
  lowest in-scope confidence was 0.44.
- **0.30 match score and 0.03 ambiguity margin:** chosen by trying values on the 30-question
  dev set only. Nearby values gave similar dev results, and with only 20 in-scope dev questions the choice is
  a reasonable judgment call, not an optimized constant.

---

## 3. Installing dependencies

You need **Python 3.10 or later** (required by Streamlit 1.56+). From inside the
`faq_chatbot` folder, run:

```bash
pip install -r requirements.txt
```

This installs Streamlit (1.56 or newer), scikit-learn, pandas, numpy, and joblib.

## 4. Running the app locally

```bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) — open it in your
browser. On the first run (or after the CSV data changes) the app trains the model and saves
`model.pkl`; you'll briefly see a "Loading chatbot model..." spinner.

## 5. Training the model manually (optional)

```bash
python train_model.py
```

This trains on the **full** labeled dataset (273 examples) and saves `model.pkl`. The app
does this automatically when needed, so this is only for forcing a fresh model.

## 6. Evaluating the model (research metrics)

```bash
python evaluate_model.py
```

Part 1 (classifier) and Part 2 (full answer pipeline) are reported separately because
**classification accuracy is not the same as FAQ answer relevance**: a question can be
assigned the right category and still receive a related-but-wrong FAQ answer.

### Part 1 — category classifier on a held-out 20% split

273 labeled examples → 218 training / 55 test, stratified by category, `random_state=42`. The
test examples are never used to train the model being evaluated.

| Metric | New data (273 examples, 55 test) | Original data (90 examples, 18 test) |
|---|---|---|
| Accuracy | 0.9091 | 0.7778 |
| Precision (macro) | 0.9102 | 0.7778 |
| Recall (macro) | 0.9084 | 0.7778 |
| F1 (macro) | 0.9088 | 0.7778 |
| Avg. response time | ≈0.02 ms per question | — |

Do **not** read the jump as proof of a better model. The two columns use different test sets,
and the new number is flattering: rewordings of the same FAQ can fall on both sides of the
random split, so the classifier sees near-duplicates of its test questions.

### Part 2 — full pipeline on questions the system has never seen

`faq_test_questions.csv` (45 in-scope + 15 out-of-scope questions) was written separately from
the training data and the knowledge base (the script warns if any question overlaps). It is
**the more realistic estimate**.

| Test set (60 questions) | Result |
|---|---|
| Category correct (in-scope, raw classifier) | 44/45 = 97.8% |
| **Correct FAQ answer returned** | **25/45 = 55.6%** |
| Wrong FAQ answer returned | 15/45 = 33.3% |
| Fell back or asked "Did you mean…" instead | 5/45 = 11.1% |
| Out-of-scope questions correctly refused | 13/15 = 86.7% |
| Predefined dropdown questions returning their own answer | 45/45 (sanity check, not generalization) |

The dev set (30 questions, used to pick thresholds, so optimistic) gave 13/20 correct, 4 wrong,
3 fallbacks, and 10/10 out-of-scope refused.

**What this means:** the classifier is good at the category, but choosing the exact FAQ from
free-typed wording is the weak point of the system — roughly one custom question in three
gets a related but wrong FAQ's answer. Typical failures are generic words ("documents",
"schedule") outweighing the distinctive ones, and paraphrases that share few words with
any stored wording. The two out-of-scope questions it answered anyway both contained words
from the dataset ("school", "times"). Selecting a question from the dropdown does not have this
problem (45/45). The most effective improvement is more and more varied wordings per FAQ —
ideally collected from real students.

**Transparency note:** the matcher settings were fixed using the dev set before the test set was
first run, and nothing was retuned afterwards. One extra idea (also scoring against answer text)
was tried on the dev set only after seeing the test results and was rejected as not clearly
beneficial. Any further change to the matcher or the data made after reading these results
would turn the test set into a tuning set.

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
   (`app.py`, `faq_engine.py`, `train_model.py`, `evaluate_model.py`, the four CSV
   files, `requirements.txt`, `.gitignore`, and this `README.md`) were uploaded. `model.pkl`
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

**Streamlit Community Cloud must install Streamlit 1.56 or newer (see `requirements.txt`) and
run on Python 3.10 or later; if the build log shows an older Streamlit, open the app's
*Advanced settings* and pick Python 3.11 or 3.12, then reboot the app.**

**Do not claim the app is "deployed" in your paper or demonstration until you have
actually completed these steps and confirmed the public URL loads correctly.**

### Testing the deployed app using the public URL

1. Open the public URL in a fresh, private/incognito browser window (to simulate a
   new user with no cached session).
2. Confirm the welcome message appears and the three category cards are visible.
3. Pick a question from the dropdown, click **Ask Question**, and confirm a reply
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

Run these in a real browser before presenting (the real Streamlit interface was not tested
during development — see section 0):

**Dropdown**
1. Each category button shows 15 questions in the dropdown; the placeholder reads
   "Search or select a question...".
2. Click the dropdown and type `status` → only matching options remain. Try `STATUS`, `stat`,
   and a word from the middle of a question.
3. Clear the search text → all 15 questions return.
4. Type something that matches nothing → the dropdown shows its built-in "no results" text
   (its wording is set by Streamlit and cannot be customized without a custom component).
5. Switch category → the options change and nothing from the previous category remains.
6. With nothing selected, **Ask Question** is disabled.
7. Select a question and click **Ask Question** → exactly one question/answer pair appears and
   the dropdown resets.
8. Pick two different questions in the same category → the two answers are different.

**Typed questions**
9. "How can I check my enrollment status?" → the enrollment-status answer.
10. A reworded question, e.g. "Is there a way to see whether my enrollment has been
    processed?" → note which FAQ is matched (shown in "This looks closest to…") and whether it
    is right; record misses as a limitation.
11. "What's the weather today?" → fallback message.
12. Click Send with nothing typed → a gentle prompt, no error.
13. "New chat" resets; reloading the page starts a fresh session.
14. Check the layout on a phone-width window.

**Deployment**
15. `requirements.txt` installs cleanly in a brand-new environment / the Streamlit Cloud build log.

---

## 10. Known limitations (for your research paper)

- The dataset is small and written by the developer (273 labeled questions, 45 FAQs), so
  measured results may not reflect real student phrasing, slang, or typos.
- **FAQ retrieval is the weakest stage**: on the held-out custom test questions only 55.6%
  returned the exact intended FAQ and 33.3% returned a different FAQ's answer. Category
  accuracy (97.8%) must not be reported as answer accuracy.
- The question sets used for Part 2 are small (30 and 60) and written by the same developer
  as the FAQs; confidence intervals are wide.
- Part 1 accuracy is optimistic because near-duplicate rewordings appear in both the training
  and test splits.
- The classifier's out-of-scope rejection relies on low confidence; questions that share
  words with the dataset ("school", "times") can slip through.
- The three categories are intentionally narrow; other topics are expected to fall back.
- Answers are generic **sample guidance**, not verified school policy. They deliberately
  contain no dates, fees, room numbers, office hours, or contacts. A real deployment would
  replace them with content confirmed by the actual school office.
- The 0.40 / 0.30 / 0.03 thresholds were chosen from small manual question sets, not through a
  rigorous validation procedure.
