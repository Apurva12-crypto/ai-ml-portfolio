import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import synth

def test_small_run_catches_defects(tmp_path, monkeypatch):
    r, kept = synth.run(n=300, defect_rate=0.4, seed=3)
    assert r["clean_rejected"] <= 0.05 * r["clean_examples"]
    assert r["kept_defect_rate"] <= 0.1
