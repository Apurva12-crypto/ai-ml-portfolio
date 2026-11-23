# Interview knowledge base: time-split fraud scoring (IEEE-CIS)
*(Target: PayPal, Visa, Coinbase, Razorpay-style fraud / risk ML roles)*

## 0. Read this first: what is real and what is not
- The pipeline is complete and was smoke-tested **only on synthetic data** with the IEEE-CIS schema (`src/make_synthetic.py`). Those numbers are meaningless and are not reported anywhere.
- **No real IEEE-CIS result exists yet.** My workspace cannot reach Kaggle. Run `notebooks/02_paypal_fraud_kaggle.ipynb` on Kaggle (add the competition data; CPU is fine), then fill in section 3 from `results/metrics.json`. Until you do, **do not quote any metric** in interviews.
- Code was built with AI assistance. Be ready to explain everything; section 6 is written for that.

## 1. The 60-second pitch (use after you have run it)
"I built a fraud scorer on the IEEE-CIS data with a chronological train/validation/test split and a one-day embargo, so nothing from the future leaks into training. Features such as per-card spend history use only earlier transactions. A LightGBM model is early-stopped on validation PR-AUC, calibrated with isotonic regression, and the decision threshold is chosen on validation by minimising business cost (missed fraud amount + review cost + customer friction), then applied once to test. Each flagged transaction gets SHAP-based reason codes. I also checked drift with PSI and adversarial validation."

## 2. Design decisions and why
1. **Chronological split with embargo**: fraud is non-stationary and attackers adapt; random splits leak future behaviour and overstate performance. The 1-day embargo stops look-back features straddling the boundary. Train 70% / val 15% / test 15% by time.
2. **Past-only entity features**: no card ID exists, so I build a proxy `uid` = card1 + addr1 + estimated account start day (transaction day minus D1). Features: transactions seen so far, mean prior amount, amount vs card's usual spend, seconds since previous transaction, transactions in the last 24h. All computed with cumulative counts that exclude the current row.
3. **Encoders fit on train only**; unseen categories map to -1.
4. **Metrics**: PR-AUC is the headline (ROC-AUC looks good even for weak models at ~3.5% fraud). Also precision/recall at top 0.5/1/2% of traffic (reviewer capacity).
5. **Cost-based threshold**: FN costs the transaction amount; FP costs $3 review + 2% friction. These are **illustrative** assumptions; say so and show how the optimum moves when they change.
6. **Calibration**: isotonic on validation so scores read as probabilities; report Brier and ECE before and after.
7. **Explainability**: TreeExplainer SHAP, top-3 positive contributors mapped to readable reasons (e.g. "many transactions on this card in the last 24h"). Reason codes support analyst triage and regulatory adverse-action style explanations.
8. **Drift checks**: PSI per numeric feature and an adversarial-validation classifier (train vs test). AUC near 0.5 means the periods look alike; high AUC means drift.

## 3. Fill after your Kaggle run
| Metric | Value (from results/metrics.json) |
|---|---|
| Fraud rate train / val / test | |
| Baseline logistic regression PR-AUC | |
| LightGBM test ROC-AUC / PR-AUC | |
| Brier / ECE raw vs calibrated | |
| Test cost: approve all / threshold 0.5 / val-chosen | |
| Alert rate, precision, recall at val-chosen threshold | |
| Top SHAP features | |
| Adversarial validation AUC; top PSI features | |

## 4. Known weakness found during the smoke test (use it, it's a real insight)
My expanding-window features (`uid_prev_n`, `uid_prev_mean_amt`, time-gap) **drift by construction**: a card's history grows over time, so later periods look different from the training period (PSI and adversarial AUC were very high on the synthetic run). On real data check whether this hurts. The production fix is **rolling windows** (last 24h/7d/30d counts and means) so the feature meaning is stable. Mention this as a design improvement you identified.

## 5. Limitations to state yourself
- IEEE-CIS is anonymised (V columns, masked IDs) and a 6-month snapshot; no chargeback delay modelling (real labels arrive weeks later, so training labels for recent data are incomplete).
- Cost assumptions are illustrative. A real system tunes them with finance.
- No online serving test, no feedback loop, no adversarial robustness.
- `app/api.py` scores already-engineered rows; a production system needs a feature store computing the history features online.

## 6. Likely questions and strong answers
**Q: Why not a random train/test split?** It lets the model see transactions after the ones it is tested on, and fraud patterns evolve. Time splits give an honest estimate of how the model degrades going forward.
**Q: Why PR-AUC over ROC-AUC?** With ~3.5% positives, ROC-AUC is dominated by the many easy negatives. PR-AUC reflects precision at each recall level, which is what analysts experience.
**Q: What is target leakage here?** Using information unavailable at decision time: future transactions in aggregates, full-dataset encodings, or label-derived features. I used cumulative counts excluding the current row and fit encoders on train only.
**Q: How did you choose the threshold?** Minimising expected cost on validation, then freezing it for test. I also show reviewer-capacity views (precision at top 1%).
**Q: Class imbalance?** I did not resample; GBMs handle imbalance with PR-AUC early stopping, and calibration afterwards keeps probabilities meaningful. Resampling would distort probabilities. `scale_pos_weight` is an alternative to try.
**Q: Why isotonic calibration?** Monotone, non-parametric, fits well with enough validation data. Risk: overfits with few positives; Platt scaling is smoother.
**Q: How do SHAP reason codes work?** SHAP assigns each feature a contribution to the log-odds relative to a baseline using Shapley values from game theory; TreeSHAP computes them exactly and efficiently for trees. The top positive contributors become reasons.
**Q: How would you monitor this in production?** Score distribution and PSI per feature, alert rate, delayed-label precision/recall, calibration drift, and a champion/challenger setup.
**Q: How do you handle delayed labels?** Train on data old enough for labels to mature (e.g. 60-90 days), model label delay explicitly, and use proxy signals (disputes, manual review outcomes) for fast feedback.
**Q: Concept drift vs covariate drift?** Covariate: P(X) changes. Concept: P(y|X) changes (attackers adapt). Adversarial validation detects covariate shift only.
**Q: Fairness and compliance?** Review features that proxy protected attributes, provide reason codes, document model risk (e.g. SR 11-7 style validation), and monitor false-positive rates across segments.
**Q: What would you do with 2 more weeks?** Rolling-window features, graph features (shared devices and emails across cards), cost-sensitive training, a rules-vs-model comparison, and a latency budget for serving.

## 7. Files
`src/fraud.py` (whole pipeline), `src/make_synthetic.py` (smoke test only), `notebooks/02_paypal_fraud_kaggle.ipynb`, `app/api.py` (needs model files from your real run).
