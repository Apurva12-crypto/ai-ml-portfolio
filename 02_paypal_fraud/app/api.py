"""Scores ALREADY-ENGINEERED feature rows (same columns the training run used) and returns reason codes.
Needs results/lgbm.txt + results/meta.json produced by src/fraud.py on the real IEEE-CIS data."""
import json, os
import numpy as np, pandas as pd, lightgbm as lgb, shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
RES = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "results"))
app = FastAPI(title="Fraud scoring API", version="1.0")
booster = lgb.Booster(model_file=os.path.join(RES, "lgbm.txt")); meta = json.load(open(os.path.join(RES, "meta.json")))
expl = shap.TreeExplainer(booster)
class Tx(BaseModel): features: dict
@app.get("/health")
def health(): return {"ok": True, "threshold": meta["threshold"], "n_features": len(meta["cols"])}
@app.post("/score")
def score(tx: Tx):
    missing = [c for c in meta["cols"] if c not in tx.features and c not in meta["cats"]]
    row = pd.DataFrame([{c: tx.features.get(c, np.nan) for c in meta["cols"]}])
    for c in meta["cats"]: row[c] = pd.to_numeric(row[c], errors="coerce").fillna(-1).astype(int)   # caller sends label-encoded categoricals
    p = float(booster.predict(row)[0]); sv = expl.shap_values(row); sv = sv[1] if isinstance(sv, list) else sv
    top = np.argsort(-sv[0])[:3]
    return {"fraud_score": p, "flag": p >= meta["threshold"], "threshold": meta["threshold"], "top_reasons": [meta["cols"][i] for i in top], "missing_features": missing[:20]}
