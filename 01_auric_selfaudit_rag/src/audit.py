"""Self-auditing multi-hop retrieval: multi-path retrieval, agreement-based coverage audit, calibrated abstention."""
import re, math, json, random, numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from world import build, questions

TOK = lambda t: re.findall(r"[a-z0-9]+", t.lower())

class BM25:
    def __init__(s, docs, k1=1.2, b=0.75):
        s.k1, s.b = k1, b; s.tf = [Counter(TOK(d["text"])) for d in docs]; s.len = np.array([sum(t.values()) for t in s.tf]); s.avg = s.len.mean()
        s.inv = defaultdict(list)
        for i, t in enumerate(s.tf):
            for w, c in t.items(): s.inv[w].append((i, c))
        s.N = len(docs)
    def search(s, q, k):
        sc = defaultdict(float)
        for w in set(TOK(q)):
            post = s.inv.get(w, []);  idf = math.log(1 + (s.N - len(post) + .5) / (len(post) + .5))
            for i, c in post: sc[i] += idf * c * (s.k1 + 1) / (c + s.k1 * (1 - s.b + s.b * s.len[i] / s.avg))
        return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]

class Dense:   # LSA stand-in for an embedding model; swap with bge/e5 on GPU (see kaggle notebook)
    def __init__(s, docs, dim=96):
        s.v = TfidfVectorizer(token_pattern=r"[A-Za-z0-9]+"); X = s.v.fit_transform([d["text"] for d in docs])
        s.svd = TruncatedSVD(dim, random_state=0); s.E = s.svd.fit_transform(X); s.E /= (np.linalg.norm(s.E, axis=1, keepdims=True) + 1e-9)
    def search(s, q, k):
        e = s.svd.transform(s.v.transform([q]))[0]; e /= (np.linalg.norm(e) + 1e-9)
        return list(np.argsort(-(s.E @ e))[:k])

class Graph:   # exact entity-mention index: precise when the entity string is exact
    def __init__(s, docs):
        s.docs = docs; s.by_ent = defaultdict(list); s.by_key = defaultdict(list)
        for i, d in enumerate(docs):
            s.by_ent[d["ent"]].append(i)
            if d["kind"] == "person": s.by_key[re.sub(r" [A-Z]\. ", " ", d["ent"])].append(i)
        for i, d in enumerate(docs):   # company_move docs mention the new city too
            pass
    def search(s, q, k, ent=None):
        if not ent: return []
        return (s.by_ent.get(ent) or s.by_key.get(ent, []))[:k]

def rrf(lists, k=3, c=60):
    sc = defaultdict(float)
    for L in lists:
        for r_, i in enumerate(L): sc[i] += 1 / (c + r_ + 1)
    return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:k]]

# ---- reader (extractive; this is where an LLM plugs in on Kaggle) ----
def read(rel, ent, doc):
    t, e = doc["text"], re.escape(doc["ent"])
    if rel == "employer":
        m = re.match(e + r" is a \w+ who joined (.+?) in (\d{4})\.", t); r_ = (m.group(1), 0) if m else None
    elif rel == "hq":
        m = re.match(e + r" is headquartered in (\w+)\.", t); r_ = (m.group(1), 0) if m else None
        if not m:
            m = re.match(r"In (\d{4}), " + e + r" relocated its headquarters to (\w+)\.", t); r_ = (m.group(2), int(m.group(1))) if m else None
    else:
        m = re.match(e + r" is located in (\w+)\.", t); r_ = (m.group(1), 0) if m else None
    return (r_[0], r_[1], doc["ent"] == ent) if r_ else None

def hop(rel, ent, qtext, docs, paths, K):
    """Run all paths, fuse, read. Returns answer, supporting docs, features."""
    ranked = {n: p.search(qtext, K) if n != "graph" else p.search(qtext, K, ent) for n, p in paths.items()}
    fused = rrf(list(ranked.values()), K)
    def answer_from(ids):
        c = [(read(rel, ent, docs[i]), i) for i in ids]; c = [(a, i) for a, i in c if a]
        if not c: return None, [], 0, 0.0
        ex = [x for x in c if x[0][2]]; pool = ex or c
        best = max(pool, key=lambda x: x[0][1])    # latest dated fact wins; ties keep retrieval rank order
        return best[0][0], [i for _, i in c], len({a[0] for a, _ in pool}), float(bool(ex))
    ans, sup, ndist, exact = answer_from(fused)
    per_path = {n: answer_from(ids)[0] for n, ids in ranked.items()}
    agree = np.mean([a == ans for a in per_path.values()]) if ans else 0.0
    return ans, sup, {"exact": exact, "agree": agree, "n_distinct": ndist, "found": float(ans is not None), "per_path": per_path, "ranked": ranked}

def solve(q, docs, paths, K, mode):
    rels = ["employer", "hq"] + (["country"] if q["hops"] == 3 else [])
    ent, feats, trail = q["surface"], [], []
    for rel in rels:
        if mode == "bm25": P = {"bm25": paths["bm25"]}
        elif mode == "dense": P = {"dense": paths["dense"]}
        else: P = paths
        qtext = ent + (" joined" if rel == "employer" else " headquartered relocated" if rel == "hq" else " located")
        ans, sup, f = hop(rel, ent, qtext, docs, P, K)
        if rel == "employer" and f["n_distinct"] == 0: f["n_distinct"] = 0
        feats.append(f); trail.append((rel, ent, ans, sup))
        if ans is None: return None, feats, trail
        ent = ans
        if rel == "hq" and q["hops"] == 3:
            # country hop is keyed by city entity; ent is now the city name
            pass
    return ent, feats, trail

def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, float); e = 0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any(): e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)

def aurc(conf, correct):
    o = np.argsort(-np.asarray(conf)); c = np.asarray(correct, float)[o]; risks = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(risks.mean())

def qfeat(feats):
    a = [f["agree"] for f in feats]; nd = [f["n_distinct"] for f in feats]
    return [min(a), float(np.mean(a)), max(nd), min(f["exact"] for f in feats), len(feats)]

def run(seed=0, K=3, out="results"):
    w = build(seed); docs = w["docs"]; Q = questions(w, 2400, seed + 1)
    paths = {"bm25": BM25(docs), "dense": Dense(docs), "graph": Graph(docs)}
    R = {"K": K, "n_docs": len(docs), "n_people": len(w["people"]), "systems": {}}
    store = {}
    for mode in ("bm25", "dense", "fused"):
        res = [solve(q, docs, paths, K, mode) for q in Q]
        acc = np.array([ans == q["answer"] for (ans, _, _), q in zip(res, Q)], float)
        R["systems"][mode] = {"accuracy": round(float(acc.mean()), 4),
                              "acc_2hop": round(float(acc[[q["hops"] == 2 for q in Q]].mean()), 4),
                              "acc_3hop": round(float(acc[[q["hops"] == 3 for q in Q]].mean()), 4),
                              "acc_relocated_companies": round(float(acc[[q["relocated"] for q in Q]].mean()), 4),
                              "acc_exact_name": round(float(acc[[q["surface_kind"] == "exact" for q in Q]].mean()), 4),
                              "acc_typo_name": round(float(acc[[q["surface_kind"] == "typo" for q in Q]].mean()), 4),
                              "acc_twin_ambiguous_name": round(float(acc[[q["ambiguous"] for q in Q]].mean()), 4)}
        store[mode] = (res, acc)
    # ---- calibration & abstention on the fused system: fit on first 800, test on the rest ----
    res, acc = store["fused"]; X = np.array([qfeat(f) for _, f, _ in res]); d, t = slice(0, 800), slice(800, None)
    raw = X[:, 1]            # uncalibrated confidence = mean cross-path agreement
    lr = LogisticRegression(C=2.0, max_iter=500).fit(X[d], acc[d]); cal = lr.predict_proba(X[t])[:, 1]
    y = acc[t]
    sel = {}
    order = np.argsort(-cal)
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6):
        k = int(cov * len(y)); sel[str(cov)] = round(float(y[order[:k]].mean()), 4)
    # abstain threshold chosen on dev to hit >=95% selective accuracy
    cd = lr.predict_proba(X[d])[:, 1]; th = None
    for c in np.sort(np.unique(cd)):
        m = cd >= c
        if m.sum() > 50 and acc[d][m].mean() >= 0.95: th = float(c); break
    ab = cal < th if th is not None else np.zeros(len(y), bool)
    R["calibration_abstention_fused"] = {
        "test_n": int(len(y)), "test_accuracy": round(float(y.mean()), 4),
        "ECE_raw_agreement": round(ece(raw[t], y), 4), "ECE_calibrated": round(ece(cal, y), 4),
        "AURC_raw": round(aurc(raw[t], y), 4), "AURC_calibrated": round(aurc(cal, y), 4),
        "selective_accuracy_at_coverage": sel,
        "abstain_threshold_from_dev": th, "abstain_rate_test": round(float(ab.mean()), 4),
        "accuracy_on_answered": round(float(y[~ab].mean()), 4) if (~ab).any() else None,
        "wrong_answers_avoided": int(((~y.astype(bool)) & ab).sum()), "correct_answers_lost": int((y.astype(bool) & ab).sum())}
    # ---- evidence recall per hop (gold supporting doc in fused top-K), by path ----
    gold = defaultdict(list)
    for q, (_, f, tr) in zip(Q, res):
        for (rel, ent, ans, sup), fh in zip(tr, f):
            for n, ids in fh["ranked"].items():
                gold[n].append(any((read(rel, ent, docs[i]) or (0, 0, False))[2] for i in ids))
    amb = np.array([q["ambiguous"] for q in Q])[800:]
    R["calibration_abstention_fused"]["abstain_rate_on_twin_ambiguous"] = round(float(ab[amb].mean()), 4) if amb.any() else None
    R["calibration_abstention_fused"]["abstain_rate_on_exact_name"] = round(float(ab[np.array([q["surface_kind"] == "exact" for q in Q])[800:]].mean()), 4)
    R["hop_evidence_recall_by_path"] = {n: round(float(np.mean(v)), 4) for n, v in gold.items()}
    R["note"] = ("Synthetic testbed built to isolate retriever failure modes (twin entities, relocated companies). "
                 "Reader is extractive/rule-based; numbers say nothing about LLM quality or real corpora.")
    json.dump(R, open(f"{out}/metrics.json", "w"), indent=1)
    # a few failure examples for the knowledge base
    fails = [(q, tr) for q, (a, f, tr), ok in zip(Q, res, acc) if not ok][:8]
    json.dump([{"q": q["q"], "gold": q["answer"], "trail": [(r_, e, a) for r_, e, a, _ in tr]} for q, tr in fails], open(f"{out}/failure_examples_fused.json", "w"), indent=1)
    return R

if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
