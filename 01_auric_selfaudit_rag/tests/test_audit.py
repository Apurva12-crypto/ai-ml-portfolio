import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import audit

HERE = os.path.dirname(__file__)

def test_ece_perfect_and_bad():
    c = np.array([1.0] * 50 + [0.0] * 50); y = np.array([1] * 50 + [0] * 50)
    assert audit.ece(c, y) < 1e-9
    assert audit.ece(np.ones(100), np.zeros(100)) > 0.9

def test_aurc_prefers_good_ranking():
    y = np.array([1, 1, 1, 0, 0]); good = np.array([.9, .8, .7, .2, .1]); bad = good[::-1]
    assert audit.aurc(good, y) < audit.aurc(bad, y)

def test_rrf_combines():
    out = audit.rrf([[1, 2, 3], [1, 3, 2]], k=2)
    assert out[0] == 1 and len(out) == 2

def test_smoke_run(tmp_path):
    out = str(tmp_path); r = audit.run(seed=0, K=3, out=out)
    assert os.path.exists(os.path.join(out, "metrics.json"))

def test_committed_metrics_sane():
    d = json.load(open(os.path.join(HERE, "..", "results", "metrics.json")))
    c = d["calibration_abstention_fused"]
    assert c["AURC_calibrated"] < c["AURC_raw"]
    assert 0 <= c["abstain_rate_test"] <= 1
