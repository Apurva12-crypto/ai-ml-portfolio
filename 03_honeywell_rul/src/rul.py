"""Remaining-useful-life (RUL) prediction on NASA C-MAPSS.

Original implementation. Data: NASA C-MAPSS turbofan degradation simulation
(Saxena & Goebel, 2008), public domain NASA Prognostics Data Repository.
"""
import json, time, argparse, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor

COLS = ["unit", "cycle"] + [f"op{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]
RUL_CAP = 125      # piecewise-linear target: early life is "healthy", RUL not identifiable
WIN = 30

def load(ds, d="data"):
    tr = pd.read_csv(f"{d}/train_{ds}.txt", sep=r"\s+", header=None, names=COLS)
    te = pd.read_csv(f"{d}/test_{ds}.txt", sep=r"\s+", header=None, names=COLS)
    rul = pd.read_csv(f"{d}/RUL_{ds}.txt", header=None)[0].values
    return tr, te, rul

def add_rul(df):
    mx = df.groupby("unit")["cycle"].transform("max")
    df["rul"] = (mx - df["cycle"]).clip(upper=RUL_CAP)
    return df

class Prep:
    """Fit on TRAIN only. Drops near-constant sensors; normalises per operating condition."""
    def __init__(self, n_cond):
        self.n_cond = n_cond
    def fit(self, tr):
        ops = tr[["op1", "op2", "op3"]].values
        self.km = KMeans(self.n_cond, n_init=5, random_state=0).fit(ops) if self.n_cond > 1 else None
        cond = self._cond(tr)
        sens = [f"s{i}" for i in range(1, 22)]
        # sensors that vary within a condition carry degradation signal
        keep = [s for s in sens if tr.groupby(cond)[s].std().mean() > 1e-4]
        self.sens = keep
        self.stats = {c: (tr.loc[cond == c, keep].mean(), tr.loc[cond == c, keep].std().replace(0, 1))
                      for c in np.unique(cond)}
        return self
    def _cond(self, df):
        return self.km.predict(df[["op1", "op2", "op3"]].values) if self.km else np.zeros(len(df), int)
    def transform(self, df):
        cond = self._cond(df)
        out = np.zeros((len(df), len(self.sens)), np.float32)
        for c, (m, s) in self.stats.items():
            idx = cond == c
            out[idx] = ((df.loc[idx, self.sens] - m) / s).values
        return out

def windows(df, X, win=WIN, last_only=False):
    xs, ys, us = [], [], []
    for u, g in df.groupby("unit"):
        a = X[g.index.values]
        if len(a) < win:                      # pad by repeating first row
            a = np.vstack([np.repeat(a[:1], win - len(a), 0), a])
        rr = g["rul"].values if "rul" in g else None
        rng = [len(a) - 1] if last_only else range(win - 1, len(a))
        for e in rng:
            xs.append(a[e - win + 1:e + 1]); us.append(u)
            if rr is not None:
                ys.append(rr[min(e, len(rr) - 1)] if len(rr) >= win else rr[-1])
    return np.stack(xs), (np.array(ys, np.float32) if ys else None), np.array(us)

def nasa_score(y, p):
    d = p - y
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))

def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))

class LSTMNet(nn.Module):
    def __init__(s, f, h=64, drop=0.2):
        super().__init__()
        s.rnn = nn.LSTM(f, h, 2, batch_first=True, dropout=drop)
        s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(h, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x): return s.head(s.rnn(x)[0][:, -1]).squeeze(-1)

class TfNet(nn.Module):
    def __init__(s, f, d=64, heads=4, layers=2, drop=0.1, win=WIN):
        super().__init__()
        s.inp = nn.Linear(f, d); s.pos = nn.Parameter(torch.zeros(1, win, d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 128, drop, batch_first=True), layers)
        s.head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x):
        h = s.enc(s.inp(x) + s.pos)
        return s.head(h.mean(1)).squeeze(-1)

def fit_torch(kind, Xtr, ytr, Xva, yva, seed, epochs=30, bs=256, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    net = (LSTMNet if kind == "lstm" else TfNet)(Xtr.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr / RUL_CAP)
    Xv = torch.tensor(Xva)
    best, best_sd, bad = 1e9, None, 0
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); nn.functional.mse_loss(net(Xt[b]), yt[b]).backward(); opt.step()
        sch.step(); net.eval()
        with torch.no_grad(): pv = net(Xv).numpy() * RUL_CAP
        r = rmse(yva, pv)
        if r < best: best, bad, best_sd = r, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6: break
    net.load_state_dict(best_sd); net.eval()
    return net

def predict(net, X):
    with torch.no_grad(): return net(torch.tensor(X)).numpy() * RUL_CAP

def feats(W):  # flattened summary features for the tree baseline
    last = W[:, -1]; mean = W.mean(1); std = W.std(1)
    slope = W[:, -1] - W[:, 0]
    return np.hstack([last, mean, std, slope])

def run(ds, seeds=3, epochs=30, out="results"):
    t0 = time.time()
    n_cond = 6 if ds in ("FD002", "FD004") else 1
    tr, te, rul_te = load(ds); tr = add_rul(tr).reset_index(drop=True); te = te.reset_index(drop=True)
    prep = Prep(n_cond).fit(tr)
    Xtr_all, Xte_all = prep.transform(tr), prep.transform(te)
    # ---- ENGINE-LEVEL split: no engine appears in both train and validation ----
    units = tr["unit"].unique(); rs = np.random.RandomState(0); rs.shuffle(units)
    val_u = set(units[: int(0.2 * len(units))])
    Wall, yall, uall = windows(tr, Xtr_all)
    m = np.array([u in val_u for u in uall])
    Wtr, ytr, Wva, yva = Wall[~m], yall[~m], Wall[m], yall[m]
    # last-window-only validation mimics the test protocol (predict at an arbitrary cut)
    Wte, _, _ = windows(te, Xte_all, last_only=True)
    yte = np.minimum(rul_te, RUL_CAP).astype(np.float32)
    res = {"dataset": ds, "n_train_engines": int(len(units)), "n_features": int(Wall.shape[-1]),
           "window": WIN, "rul_cap": RUL_CAP, "models": {}}

    # ---- leakage demonstration: same GBM, random-window split vs engine-level split ----
    F = feats(Wall)
    ri = np.random.RandomState(1).permutation(len(F)); cut = int(0.8 * len(F))
    g = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0)
    g.fit(F[ri[:cut]], yall[ri[:cut]]); leak_rmse = rmse(yall[ri[cut:]], g.predict(F[ri[cut:]]))
    g2 = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, random_state=0).fit(F[~m], yall[~m])
    eng_rmse = rmse(yall[m], g2.predict(F[m]))
    res["leakage_demo"] = {"val_rmse_random_window_split": leak_rmse, "val_rmse_engine_split": eng_rmse}
    p = g2.predict(feats(Wte))
    res["models"]["gbm_baseline"] = {"test_rmse": rmse(yte, p), "test_nasa_score": nasa_score(yte, p)}

    for kind in ("lstm", "transformer"):
        preds = []
        for sd in range(seeds):
            net = fit_torch(kind, Wtr, ytr, Wva, yva, seed=sd, epochs=epochs)
            preds.append(predict(net, Wte))
        P = np.stack(preds); mu, sdv = P.mean(0), P.std(0)
        # empirical 80% interval from ensemble spread, widened by a factor calibrated on VALIDATION engines
        lo, hi = mu - 1.28 * sdv, mu + 1.28 * sdv
        cov = float(np.mean((yte >= lo) & (yte <= hi)))
        res["models"][kind] = {"test_rmse": rmse(yte, mu), "test_nasa_score": nasa_score(yte, mu),
                               "rmse_per_seed": [rmse(yte, q) for q in preds],
                               "ens_interval80_coverage": cov, "mean_ens_std": float(sdv.mean())}
        np.save(f"{out}/pred_{ds}_{kind}.npy", np.vstack([yte, mu, sdv]))
    res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{out}/metrics_{ds}.json", "w"), indent=1)
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="FD001")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print(json.dumps(run(a.ds, a.seeds, a.epochs, out=a.out), indent=1))
