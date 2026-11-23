"""Time-split fraud scoring: leak-safe features, LightGBM, PR-AUC, cost-based thresholding, calibration, SHAP reason codes.
Original implementation. Expects Kaggle IEEE-CIS Fraud Detection files (train_transaction.csv [+ train_identity.csv])."""
import json, os, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DAY = 86400
CATS = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType", "R_emaildomain", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

def load(path):
    tr = pd.read_csv(os.path.join(path, "train_transaction.csv"))
    idp = os.path.join(path, "train_identity.csv")
    if os.path.exists(idp): tr = tr.merge(pd.read_csv(idp), on="TransactionID", how="left")
    return tr.sort_values("TransactionDT").reset_index(drop=True)

def time_split(df, f_train=0.70, f_val=0.15, embargo_days=1):
    """Chronological split with an embargo gap so features that look back <= 1 day cannot straddle the boundary."""
    t = df["TransactionDT"].values; n = len(df)
    t_tr, t_va = t[int(f_train * n)], t[int((f_train + f_val) * n)]
    tr = df[t < t_tr]; va = df[(t >= t_tr + embargo_days * DAY) & (t < t_va)]; te = df[t >= t_va + embargo_days * DAY]
    return tr.copy(), va.copy(), te.copy()

def add_features(df):
    """Features computed with PAST-ONLY information within each entity (no look-ahead)."""
    d = df.copy()
    d["hour"] = (d["TransactionDT"] // 3600) % 24; d["dow"] = (d["TransactionDT"] // DAY) % 7
    d["log_amt"] = np.log1p(d["TransactionAmt"]); d["cents"] = (d["TransactionAmt"] % 1).round(3)
    start = (d["TransactionDT"] / DAY - d["D1"].fillna(0)).round() if "D1" in d else 0
    d["uid"] = d["card1"].astype(str) + "_" + d.get("addr1", pd.Series(0, index=d.index)).astype(str) + "_" + pd.Series(start, index=d.index).astype(str)
    g = d.groupby("uid", sort=False)
    d["uid_prev_n"] = g.cumcount()                                   # transactions seen so far
    cs = g["TransactionAmt"].cumsum() - d["TransactionAmt"]          # sum of strictly earlier amounts
    d["uid_prev_mean_amt"] = np.where(d["uid_prev_n"] > 0, cs / d["uid_prev_n"].clip(lower=1), np.nan)
    d["amt_vs_uid_mean"] = d["TransactionAmt"] / d["uid_prev_mean_amt"]
    d["uid_gap_s"] = g["TransactionDT"].diff()
    d["uid_n_last_day"] = g["TransactionDT"].transform(lambda s: pd.Series(np.searchsorted(s.values, s.values) - np.searchsorted(s.values, s.values - DAY, side="left"), index=s.index))
    return d.drop(columns=["uid"])

def prep_matrix(tr, others):
    """Label-encode categoricals with a mapping fit on TRAIN only; unseen -> -1."""
    drop = {"isFraud", "TransactionID", "TransactionDT"}
    cols = [c for c in tr.columns if c not in drop]
    maps = {c: {v: i for i, v in enumerate(tr[c].dropna().unique())} for c in cols if c in CATS or tr[c].dtype == object}
    def enc(df):
        X = df[cols].copy()
        for c, m in maps.items(): X[c] = X[c].map(m).fillna(-1).astype(int)
        return X
    return cols, [enc(tr)] + [enc(o) for o in others], list(maps)

def cost_curve(y, p, amt, review_cost=3.0, friction=0.02):
    """Business cost of a threshold: a missed fraud loses its amount; a flagged legit txn costs review + customer friction."""
    ths = np.unique(np.quantile(p, np.linspace(0.5, 0.9995, 400)))
    out = []
    for t in ths:
        f = p >= t; fn = ((~f) & (y == 1)); fp = (f & (y == 0))
        out.append((t, float(amt[fn].sum() + review_cost * fp.sum() + friction * amt[fp].sum()), float(f.mean())))
    return np.array(out)

def ece(p, y, bins=15):
    e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (p > lo) & (p <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)

def psi(a, b, bins=10):
    q = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3: return 0.0
    pa = np.histogram(a[~np.isnan(a)], q)[0] / max((~np.isnan(a)).sum(), 1) + 1e-4; pb = np.histogram(b[~np.isnan(b)], q)[0] / max((~np.isnan(b)).sum(), 1) + 1e-4
    return float(np.sum((pa - pb) * np.log(pa / pb)))

REASON = {"uid_gap_s": "unusually short time since card's last transaction", "uid_n_last_day": "many transactions on this card in the last 24h",
          "amt_vs_uid_mean": "amount far from this card's usual spend", "log_amt": "transaction amount", "TransactionAmt": "transaction amount",
          "P_emaildomain": "purchaser e-mail domain risk", "hour": "unusual time of day", "uid_prev_n": "card history length", "card1": "card identifier risk"}

def run(path, out="results", device_gpu=False, max_rounds=2000):
    os.makedirs(out, exist_ok=True)
    df = add_features(load(path)); tr, va, te = time_split(df)
    cols, (Xtr, Xva, Xte), cats = prep_matrix(tr, [va, te])
    ytr, yva, yte = tr["isFraud"].values, va["isFraud"].values, te["isFraud"].values
    R = {"rows": {"train": len(tr), "val": len(va), "test": len(te)}, "fraud_rate": {"train": float(ytr.mean()), "val": float(yva.mean()), "test": float(yte.mean())}, "n_features": len(cols)}
    # --- baseline: logistic regression on a few numeric features ---
    base_cols = [c for c in ["log_amt", "hour", "dow", "uid_prev_n", "amt_vs_uid_mean"] if c in cols]
    med = Xtr[base_cols].median().fillna(0); lr = LogisticRegression(max_iter=300).fit(Xtr[base_cols].fillna(med), ytr)
    pb = lr.predict_proba(Xte[base_cols].fillna(med))[:, 1]
    R["baseline_logreg"] = {"roc_auc": roc_auc_score(yte, pb), "pr_auc": average_precision_score(yte, pb)}
    # --- LightGBM, early stopping on VALIDATION (never on test) ---
    prm = dict(objective="binary", learning_rate=0.03, num_leaves=96, min_child_samples=60, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
               lambda_l2=5.0, metric="average_precision", verbose=-1, seed=0, n_jobs=-1)
    dtr = lgb.Dataset(Xtr, ytr, categorical_feature=cats); dva = lgb.Dataset(Xva, yva, reference=dtr)
    m = lgb.train(prm, dtr, max_rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    pv, pt = m.predict(Xva), m.predict(Xte)
    R["lightgbm"] = {"best_iteration": int(m.best_iteration), "val_pr_auc": average_precision_score(yva, pv),
                     "test_roc_auc": roc_auc_score(yte, pt), "test_pr_auc": average_precision_score(yte, pt),
                     "test_brier_raw": brier_score_loss(yte, pt), "test_ece_raw": ece(pt, yte)}
    # --- calibration: isotonic fitted on VALIDATION ---
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, yva); ptc = iso.predict(pt)
    R["lightgbm"].update({"test_brier_calibrated": brier_score_loss(yte, ptc), "test_ece_calibrated": ece(ptc, yte)})
    # --- cost-based threshold: pick on VALIDATION, apply to TEST ---
    amt_v, amt_t = va["TransactionAmt"].values, te["TransactionAmt"].values
    cv = cost_curve(yva, pv, amt_v); thr = float(cv[np.argmin(cv[:, 1]), 0])
    def cost_at(y, p, a, t): f = p >= t; return float(a[(~f) & (y == 1)].sum() + 3.0 * (f & (y == 0)).sum() + 0.02 * a[f & (y == 0)].sum())
    R["cost_policy"] = {"val_chosen_threshold": thr, "assumptions": "FN costs the txn amount; FP costs $3 review + 2% of amount friction (illustrative, change to your business numbers)",
        "test_cost_approve_all": float(amt_t[yte == 1].sum()), "test_cost_thr_0.5": cost_at(yte, pt, amt_t, 0.5), "test_cost_val_chosen": cost_at(yte, pt, amt_t, thr),
        "test_alert_rate_val_chosen": float((pt >= thr).mean()), "test_recall_val_chosen": float(((pt >= thr) & (yte == 1)).sum() / max(yte.sum(), 1)),
        "test_precision_val_chosen": float(((pt >= thr) & (yte == 1)).sum() / max((pt >= thr).sum(), 1))}
    for frac in (0.005, 0.01, 0.02):   # reviewer-capacity view
        k = max(int(frac * len(pt)), 1); idx = np.argsort(-pt)[:k]
        R["cost_policy"][f"precision_at_top_{frac:.1%}"] = float(yte[idx].mean()); R["cost_policy"][f"recall_at_top_{frac:.1%}"] = float(yte[idx].sum() / max(yte.sum(), 1))
    # --- drift & adversarial validation (train vs test) ---
    num = [c for c in cols if c not in cats][:40]
    R["psi_top"] = dict(sorted({c: psi(Xtr[c].values.astype(float), Xte[c].values.astype(float)) for c in num}.items(), key=lambda x: -x[1])[:8])
    Xa = pd.concat([Xtr.sample(min(len(Xtr), 50000), random_state=0), Xte.sample(min(len(Xte), 50000), random_state=0)]); ya = np.r_[np.zeros(min(len(Xtr), 50000)), np.ones(min(len(Xte), 50000))]
    cut = len(Xa) // 2; idx = np.random.RandomState(0).permutation(len(Xa)); a_tr, a_te = idx[:cut], idx[cut:]
    ma = lgb.train(dict(objective="binary", verbose=-1, num_leaves=31, learning_rate=0.1), lgb.Dataset(Xa.iloc[a_tr], ya[a_tr]), 80)
    R["adversarial_validation_auc_train_vs_test"] = float(roc_auc_score(ya[a_te], ma.predict(Xa.iloc[a_te])))
    # --- SHAP reason codes ---
    import shap
    ex = shap.TreeExplainer(m); samp = Xte.iloc[np.argsort(-pt)[:300]]
    sv = ex.shap_values(samp); sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=cols).sort_values(ascending=False)
    R["shap_top_features"] = {k: float(v) for k, v in imp.head(15).items()}
    codes = []
    for i in range(min(10, len(samp))):
        top = np.argsort(-sv[i])[:3]; codes.append({"score": float(pt[np.argsort(-pt)[i]]), "reasons": [REASON.get(cols[j], cols[j]) for j in top], "features": [cols[j] for j in top]})
    R["example_reason_codes"] = codes
    json.dump(R, open(f"{out}/metrics.json", "w"), indent=1, default=float)
    m.save_model(f"{out}/lgbm.txt"); json.dump({"cols": cols, "cats": cats, "threshold": thr}, open(f"{out}/meta.json", "w"))
    return R

if __name__ == "__main__":
    import sys; print(json.dumps(run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results"), indent=1, default=float)[:3500])
