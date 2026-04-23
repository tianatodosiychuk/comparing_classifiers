# Comparing Classifiers — Bank Marketing (Module 17 Practical Application III)

Compare the performance of **k-Nearest Neighbors, Logistic Regression, Decision Trees, and Support
Vector Machines** on a real-world marketing problem: predicting which clients of a Portuguese bank
will subscribe to a term deposit after a telemarketing call.

**Dataset:** [UCI Bank Marketing Data Set](https://archive.ics.uci.edu/ml/datasets/bank+marketing) —
41,188 contacts, 20 features, 11.3% positive class.
Accompanying paper: *A Data-Driven Approach to Predict the Success of Bank Telemarketing* — Moro, Cortez & Rita (2014).

**Notebook:** [`prompt_III_submission.ipynb`](prompt_III_submission.ipynb) — contains the full
analysis with embedded outputs.

---

## Business problem

The bank runs expensive outbound telemarketing campaigns to sell term deposits, but only ~11% of
contacts convert. The goal is a classifier that **ranks clients by their probability of
subscribing**, using only information available *before* the call is placed — so the marketing
team can focus limited call capacity on the highest-propensity prospects.

The `duration` feature (call length in seconds) leaks the outcome and is dropped from the feature
set, in line with the dataset authors' guidance.

## Summary of findings (held-out test set, 20% stratified split)

| Model | Test ROC-AUC | Test F1 | Test Recall (yes) | Test Precision (yes) | Train time |
|---|---:|---:|---:|---:|---:|
| **Decision Tree (tuned)**       | **0.8025** | 0.475 | 0.661 | 0.371 |   7.6 s |
| **Logistic Regression (tuned)** | **0.8012** | 0.467 | 0.644 | 0.366 |  13.2 s |
| KNN (tuned, k = 75)             | 0.7936 | 0.343 | 0.228 | 0.684 |  10.5 s |
| SVM RBF (tuned)                 | 0.7844 | 0.490 | 0.632 | 0.400 |  52.8 s |
| *Majority-class baseline*       | 0.5000 | 0.000 | 0.000 |   —    |     0 s |

All four tuned models cluster tightly at ROC-AUC 0.78–0.80, far above the 0.50 baseline. At the
top 10% of ranked prospects, the best models capture ~47% of all subscribers — a **4.7× lift** over
random calling.

## Key findings

1. **Logistic Regression is the recommended production model** — tied with Decision Tree for best
   ROC-AUC, but trains in seconds, scores instantly, and its coefficients are directly
   interpretable for stakeholders.
2. **Macro-economic features dominate.** `nr.employed`, `euribor3m`, `emp.var.rate`, and
   `cons.price.idx` are the strongest predictors in every model — clients subscribe more often in
   periods of weakening employment and falling interest rates.
3. **Strongest individual client signals:** prior-campaign success (`poutcome = success` ⇒ ~65%
   conversion), mobile contact (`contact = cellular` ⇒ ~3× landline), and life-stage segments
   (students and retirees convert well above average).

## Actionable recommendations

- Rank prospects by model score — calling the top 10% captures roughly half of all subscribers at
  ~4.7× random efficiency.
- Lower the decision threshold from 0.5 to ~0.3 so the minority class is actually recalled.
- Prefer cellular over landline contact and always prioritise prior-campaign successes.
- Time campaigns to weak macroeconomic windows (falling Euribor, contracting employment).

## Next steps

- Benchmark ensembles (Random Forest / XGBoost / LightGBM) — typically +1–3 AUC points.
- Calibrate probabilities (Platt / isotonic) so scores represent true expected conversion rates.
- Replace the ad-hoc 0.3 threshold with a cost-based one driven by expected revenue per call.
- Monitor macro-feature drift and retrain quarterly.
- A/B test the model-ranked call list against a random-assignment control in the next campaign.

---

## How to run

1. Download `bank-additional-full.csv` from the
   [UCI Bank Marketing page](https://archive.ics.uci.edu/ml/datasets/bank+marketing) (unzip
   `bank-additional.zip` → `bank-additional/bank-additional-full.csv`).
2. Open `prompt_III_submission.ipynb` in **Jupyter / JupyterLab / VS Code** locally, or upload to
   **Google Colab**.
3. Place the CSV next to the notebook, or in a `data/` subfolder, or in Colab's `/content/` — the
   notebook auto-detects all three.
4. Dependencies (pre-installed on Colab): `pandas numpy scikit-learn matplotlib seaborn scipy`.
5. **Runtime → Run All.** End-to-end runtime: ~2–3 minutes.

## Methodology (CRISP-DM)

| Phase | What the notebook does |
|---|---|
| **Business Understanding** | Defines the objective as probability ranking for call prioritisation, not yes/no classification. |
| **Data Understanding** | Profiles dtypes, `unknown` categoricals, the `pdays == 999` sentinel, class imbalance, and visualises subscription-rate drivers + feature correlations. |
| **Data Preparation** | Drops `duration` (leakage). Replaces `pdays == 999` with NaN + adds `previously_contacted` indicator. One-hot encodes categoricals (keeps `unknown` as a level). Scales numerics. 80/20 stratified train/test split. |
| **Modeling** | (a) Simple Logistic Regression on client-only features. (b) Default-hyperparameter benchmark of all 4 classifiers on a 5k sub-sample. (c) `GridSearchCV` (5-fold stratified, `scoring='roc_auc'`) for LR / KNN / Decision Tree on full train; SVM tuned on 8k sub-sample and refit on 15k. `class_weight='balanced'` where supported. |
| **Evaluation** | Accuracy, precision, recall, F1, ROC-AUC on the held-out test set. ROC-curve overlay. Logistic Regression coefficients interpreted with a plot + prose. |
| **Deployment / Next Steps** | Concrete business recommendations plus refit cadence and A/B-test plan. |

