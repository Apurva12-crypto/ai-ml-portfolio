"""SYNTHETIC IEEE-CIS-shaped data. For pipeline smoke tests ONLY - never report its metrics as a benchmark."""
import numpy as np, pandas as pd
def make(n=80000, seed=0):
    r = np.random.RandomState(seed)
    ncard = n // 6; card = r.randint(0, ncard, n); risky = r.rand(ncard) < 0.04
    t = np.sort(r.randint(86400, 86400 * 180, n))
    amt = np.exp(r.normal(4.0, 1.1, n)).round(2)
    fraud_p = 0.012 + 0.25 * risky[card] * (r.rand(n) < 0.35) + 0.02 * (amt > 600)
    y = (r.rand(n) < np.clip(fraud_p, 0, 0.9)).astype(int)
    amt = np.where(y == 1, amt * np.exp(r.normal(0.5, 0.6, n)), amt).round(2)
    df = pd.DataFrame({"TransactionID": np.arange(n), "isFraud": y, "TransactionDT": t, "TransactionAmt": amt,
        "ProductCD": r.choice(list("WCRHS"), n, p=[.6, .1, .15, .08, .07]), "card1": card, "card2": r.randint(100, 600, n).astype(float),
        "card4": r.choice(["visa", "mastercard", "discover", "american express"], n), "card6": r.choice(["debit", "credit"], n),
        "addr1": r.randint(100, 500, n).astype(float), "P_emaildomain": np.where((y == 1) & (r.rand(n) < .4), "anonymous.com", r.choice(["gmail.com", "yahoo.com", "hotmail.com", "icloud.com"], n)),
        "D1": np.maximum(0, (t / 86400 - r.randint(0, 150, n))).round(), "C1": r.poisson(1.5 + 3 * y), "V1": r.normal(0, 1, n) + 0.8 * y,
        "DeviceType": r.choice(["desktop", "mobile", None], n, p=[.4, .4, .2])})
    return df
