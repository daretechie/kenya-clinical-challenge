```markdown
# Kenya Clinical Reasoning Challenge: To-Do List

Below is the comprehensive list of tasks we'll tackle:

1. **Data Validation & Integrity**

   - Verify that each row follows the prescribed prompt template (nurse intro → scenario → question).
   - Cross-check all SNOMED codes for correctness using the SNOMED CT Browser.
   - Ensure there’s no unintended PII (patient names, precise addresses).

2. **Exploratory Data Analysis (EDA)**

   - Inspect basic statistics: row count, column types, missing values.
   - Plot distributions of prompt lengths, years of experience, etc.
   - Generate word-frequency summaries or word clouds for clinical prompts.

3. **Text Preprocessing**

   - Clean and normalize the `Prompt` text (lowercase, remove extra whitespace/punctuation).
   - Tokenize, remove English stop-words, and lemmatize.
   - (Optionally) Split the prompt into “nurse profile” vs. “clinical scenario” segments.

4. **Feature Engineering**

   - Vectorize processed prompts (TF-IDF; later: transformer embeddings).
   - Encode categorical fields (`County`, `Health level`, `Nursing Competency`, `Clinical Panel`) via one-hot or label encoding.
   - Standardize/impute `Years of Experience`.

5. **Clinical & Local Context Enrichment**

   - Map medical terms to Kenyan usage (e.g., “salbutamol” vs. “albuterol”).
   - (Optionally) Append Kiswahili equivalents for key terms to support localization.
   - Incorporate county- or facility-level features reflecting resource constraints.

6. **Baseline Modeling**

   - Train a simple classifier (e.g., logistic regression on TF-IDF + categorical features) to predict clinical reasoning outcomes.
   - Evaluate with cross-validation (accuracy, F1-score).

7. **Advanced Modeling**

   - Fine-tune a clinical transformer (e.g., BioBERT or Med-BERT) on the prompt→response task.
   - Experiment with multi-target setups (e.g., jointly predicting clinician text and SNOMED codes).

8. **Response Formatting & SNOMED Integration**

   - Ensure model outputs follow Zindi’s required format: “Prompt → Final Answer (with SNOMED codes)”.
   - Automatically append or validate SNOMED CT codes in the predicted answers.

9. **Benchmarking & Validation**

   - Compare AI model outputs (GPT-4.0, LLAMA, GEMINI) against human clinician responses.
   - Measure agreement on key diagnostics and SNOMED coding rates.
   - Cross-validate clinical recommendations against Kenyan Ministry of Health protocols.

10. **Dataset Expansion & Ethical Checks**

    - Identify gaps (e.g., underrepresented conditions like malaria or neonatal sepsis) and augment if possible.
    - Confirm demographic balance (age, gender, county) to mitigate bias.

11. **Final Submission Preparation**
    - Build the final prediction pipeline for the test set.
    - Generate the submission file matching the SampleSubmission format.
    - Perform a dry run to catch formatting errors, then submit to Zindi.
```
