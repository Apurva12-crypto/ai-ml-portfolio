# AI/ML portfolio: four evidence projects

Built with AI assistance (Claude). Everything is original code. Public datasets are credited; reference repos listed in `Repo_Evidence_Map.xlsx` were used for ideas only, not copied. **You must be able to explain every file: read each project's `KNOWLEDGE_BASE.md`.** If an interviewer asks how it was built, say it was built with AI assistance and that you can walk through every design decision.

| # | Project | Target roles | Status of results |
|---|---|---|---|
| 1 | `01_auric_selfaudit_rag` Self-auditing multi-hop retrieval with calibrated abstention | Auric Labs, RAG / agent reliability research | Real numbers, **synthetic testbed**; HotpotQA notebook untested |
| 2 | `02_paypal_fraud` Time-split fraud scoring, cost-based thresholds, SHAP reasons | PayPal, Visa, risk ML | Pipeline built; **real results need your Kaggle run** |
| 3 | `03_honeywell_rul` RUL prediction on NASA C-MAPSS (GBM vs LSTM vs Transformer) | Honeywell, aerospace / time-series ML | Real data, real results from this workspace |
| 4 | `04_handshake_pii_synth` PII scrubbing + synthetic-data quality filters | Handshake AI MTS, data / eval roles | Real numbers, **synthetic data**; spaCy notebook untested |

## Honest status
- Not run here: anything needing Kaggle, HuggingFace or an LLM API. Notebooks marked `UNTESTED` are templates.
- `docker compose` and the Oracle deployment were not tested (no Docker daemon, no cloud login). See `deploy/ORACLE_DEPLOY.md`.
- Never quote a number you did not produce or cannot reproduce.

## Run
```bash
pip install torch scikit-learn pandas numpy lightgbm shap faker fastapi uvicorn
python 04_handshake_pii_synth/src/pii.py      # from inside that folder
python 03_honeywell_rul/src/rul.py --ds FD001 # from inside that folder
python 01_auric_selfaudit_rag/src/audit.py    # from inside that folder
```
