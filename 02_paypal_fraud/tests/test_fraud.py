"""Smoke tests on SYNTHETIC data only. These check pipeline mechanics (no leakage, no crash); they say nothing about real fraud performance."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import fraud, make_synthetic

def small(n=6000):
    return make_synthetic.make(n=n, seed=1).sort_values("TransactionDT").reset_index(drop=True)

def test_time_split_ordered_with_embargo():
    df = small(); tr, va, te = fraud.time_split(df)
    assert tr.TransactionDT.max() < va.TransactionDT.min() - fraud.DAY + 1
    assert va.TransactionDT.max() < te.TransactionDT.min() - fraud.DAY + 1
    assert len(tr) > len(va) > 0 and len(te) > 0

def test_features_use_only_past():
    df = small(); f = fraud.add_features(df)
    # first transaction of each entity has no history
    assert (f.loc[f.uid_prev_n == 0, "uid_prev_mean_amt"].isna()).all()
    # changing a row's own amount must not change its own history features
    d2 = df.copy(); i = 3000; d2.loc[i, "TransactionAmt"] *= 100
    f2 = fraud.add_features(d2)
    assert f.loc[i, "uid_prev_n"] == f2.loc[i, "uid_prev_n"]
    same = np.isclose(f.loc[i, "uid_prev_mean_amt"], f2.loc[i, "uid_prev_mean_amt"], equal_nan=True)
    assert same
    # and a future row's change must not alter any earlier row
    assert np.allclose(f.loc[:i - 1, "uid_prev_mean_amt"].fillna(-1), f2.loc[:i - 1, "uid_prev_mean_amt"].fillna(-1))

def test_encoder_fit_on_train_only():
    df = small(); tr, va, te = fraud.time_split(df)
    tr = fraud.add_features(tr); va = fraud.add_features(va)
    tr.loc[tr.index[0], "P_emaildomain"] = "only-in-train.test"
    va.loc[va.index[0], "P_emaildomain"] = "only-in-val.test"
    cols, (Xtr, Xva), cats = fraud.prep_matrix(tr, [va])
    assert Xva["P_emaildomain"].iloc[0] == -1

def test_cost_curve_and_ece():
    r = np.random.RandomState(0); y = (r.rand(2000) < .05).astype(int); p = np.clip(y * .6 + r.rand(2000) * .4, 0, 1); a = r.rand(2000) * 100 + 5
    c = fraud.cost_curve(y, p, a); assert c.shape[1] == 3 and (c[:, 1] >= 0).all()
    assert fraud.ece(p, y) >= 0

def test_end_to_end_tiny(tmp_path):
    d = tmp_path / "data"; d.mkdir(); small(12000).to_csv(d / "train_transaction.csv", index=False)
    out = tmp_path / "res"; out.mkdir()
    R = fraud.run(str(d), str(out), max_rounds=30)
    assert (out / "metrics.json").exists() and "cost_policy" in R
