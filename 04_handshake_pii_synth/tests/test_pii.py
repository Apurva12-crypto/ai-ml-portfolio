import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pii

HERE = os.path.dirname(__file__)

def types(text):
    return {l for _, _, l, _ in pii.detect(text)}

def test_luhn():
    assert pii.luhn("4111111111111111")
    assert not pii.luhn("4111111111111112")

def test_card_detected_and_invalid_rejected():
    assert "CREDIT_CARD" in types("Card 4111 1111 1111 1111 please")
    assert "CREDIT_CARD" not in types("Order number 4111 1111 1111 1112")

def test_ssn_rules():
    assert pii.ssn_ok("123-45-6789")
    assert not pii.ssn_ok("000-12-3456") and not pii.ssn_ok("666-12-3456") and not pii.ssn_ok("900-12-3456")

def test_ip_range():
    assert pii.ip_ok("10.0.0.255") and not pii.ip_ok("300.1.1.1")

def test_aadhaar_verhoeff_roundtrip():
    body = "23456789012"
    full = body + str(pii.verhoeff_check_digit(body))
    assert pii.verhoeff_ok(full)
    bad = body + str((int(full[-1]) + 1) % 10)
    assert not pii.verhoeff_ok(bad)

def test_email_redacted():
    t = "Mail me at jane.doe@example.com today."
    out = pii.redact(t, pii.detect(t))
    assert "jane.doe@example.com" not in out

def test_version_string_not_ip():
    assert "IP_ADDRESS" not in types("Upgraded to version 3.10.4.2 of the build.")

def test_committed_ablation_consistent():
    r = json.load(open(os.path.join(HERE, "..", "results", "pii_metrics.json")))
    a = r["ablation_micro"]
    assert a["regex+validators+context"]["precision"] >= a["regex+validators"]["precision"] >= a["regex_only"]["precision"]
    m = r["ablation_micro"]["regex+validators+context"]
    assert abs(m["tp"] / (m["tp"] + m["fp"]) - m["precision"]) < 1e-3
    assert abs(m["tp"] / (m["tp"] + m["fn"]) - m["recall"]) < 1e-3
