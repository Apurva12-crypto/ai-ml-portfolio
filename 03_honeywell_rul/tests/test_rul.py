import os, sys, json, numpy as np
HERE = os.path.dirname(__file__); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import rul

def test_nasa_score_asymmetry():
    y = np.array([50.0]); assert rul.nasa_score(y, y) == 0
    assert rul.nasa_score(y, y + 10) > rul.nasa_score(y, y - 10)   # late is penalised more

def test_rmse():
    assert abs(rul.rmse(np.array([0., 0.]), np.array([3., 4.])) - np.sqrt(12.5)) < 1e-9

def test_load_prep_windows():
    tr, te, r = rul.load("FD001", os.path.join(ROOT, "data"))
    assert len(r) == te.unit.nunique() == 100
    tr = rul.add_rul(tr); assert tr.rul.max() <= rul.RUL_CAP and tr.rul.min() == 0
    p = rul.Prep(1).fit(tr); X = p.transform(tr)
    assert X.shape[0] == len(tr) and np.isfinite(X).all()
    W, y, u = rul.windows(tr, X); assert W.shape[1:] == (rul.WIN, X.shape[1]) and len(W) == len(y)
    Wt, _, ut = rul.windows(te, p.transform(te), last_only=True); assert len(Wt) == 100

def test_committed_metrics_consistent():
    for ds in ("FD001", "FD002", "FD003", "FD004"):
        d = json.load(open(os.path.join(ROOT, "results", f"metrics_{ds}.json"))); m = d["models"]
        assert m["lstm"]["test_rmse"] < m["gbm_baseline"]["test_rmse"]
        assert d["leakage_demo"]["val_rmse_random_window_split"] < d["leakage_demo"]["val_rmse_engine_split"]
        y, mu, sd = np.load(os.path.join(ROOT, "results", f"pred_{ds}_lstm.npy"))
        assert abs(rul.rmse(y, mu) - m["lstm"]["test_rmse"]) < 1e-3     # saved predictions reproduce the reported RMSE
        assert abs(rul.nasa_score(y, mu) - m["lstm"]["test_nasa_score"]) < 1.0

def test_tiny_training_smoke(tmp_path):
    cwd = os.getcwd(); os.chdir(ROOT)
    try: rul.run("FD001", seeds=1, epochs=1, out=str(tmp_path))
    finally: os.chdir(cwd)
    assert (tmp_path / "metrics_FD001.json").exists()
