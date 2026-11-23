import nbformat as nbf, pathlib
R = pathlib.Path(__file__).parent
def nb(cells, name):
    n = nbf.v4.new_notebook(); n.cells = [nbf.v4.new_code_cell(c) if k == "code" else nbf.v4.new_markdown_cell(c) for k, c in cells]
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(n, R / "notebooks" / name)
def wf(path, dest): return f"%%writefile {dest}\n" + (R / path).read_text()

# 1) PayPal fraud - REAL data on Kaggle
nb([("md", "# Fraud scoring on IEEE-CIS (time-split, cost-based threshold, SHAP)\n**Before running:** Add the competition data: Add Input -> Competitions -> *IEEE-CIS Fraud Detection* (accept rules). Settings -> Internet On is not required. CPU is fine; ~15-25 min."),
    ("code", "!pip -q install shap lightgbm"),
    ("code", wf("02_paypal_fraud/src/fraud.py", "fraud.py")),
    ("code", "import fraud, json\nR = fraud.run('/kaggle/input/ieee-fraud-detection', out='/kaggle/working/results')\nprint(json.dumps({k: R[k] for k in ['rows','fraud_rate','baseline_logreg','lightgbm','cost_policy']}, indent=1, default=float))"),
    ("code", "print(json.dumps({k: R[k] for k in ['psi_top','adversarial_validation_auc_train_vs_test','shap_top_features','example_reason_codes']}, indent=1, default=float))")], "02_paypal_fraud_kaggle.ipynb")

# 2) Honeywell RUL - GPU optional, needs C-MAPSS files as a Kaggle dataset
nb([("md", "# RUL on NASA C-MAPSS\n**Before running:** create a Kaggle Dataset containing the 12 files `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt` (public NASA data, in the project's `data/` folder) and add it as input. GPU optional."),
    ("code", wf("03_honeywell_rul/src/rul.py", "rul.py")),
    ("code", "import glob, os, shutil, json, rul\np = glob.glob('/kaggle/input/**/train_FD001.txt', recursive=True)[0]; d = os.path.dirname(p)\nos.makedirs('data', exist_ok=True); os.makedirs('results', exist_ok=True)\nfor f in glob.glob(d + '/*FD00*.txt'): shutil.copy(f, 'data/')\nfor ds in ['FD001','FD002','FD003','FD004']:\n    r = rul.run(ds, seeds=5, epochs=40); print(ds, json.dumps(r['models'], indent=1))")], "03_honeywell_rul_kaggle.ipynb")

# 3) Auric: retrieval-only real-data eval on HotpotQA (UNTESTED template)
nb([("md", "# Retrieval reliability on HotpotQA (distractor) - UNTESTED TEMPLATE\nI could not run this in my build environment (no HuggingFace access). Needs Internet On. Measures supporting-fact recall@k for BM25, dense (bge-small) and RRF, and whether cross-path agreement predicts retrieval success (the signal the self-audit method relies on)."),
    ("code", "!pip -q install rank_bm25 sentence-transformers datasets scikit-learn"),
    ("code", '''import numpy as np, re
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
ds = load_dataset("hotpot_qa", "distractor", split="validation").select(range(1500))
enc = SentenceTransformer("BAAI/bge-small-en-v1.5")
K = 5
def rrf(ls, k=K, c=60):
    s = {}
    for L in ls:
        for r, i in enumerate(L): s[i] = s.get(i, 0) + 1 / (c + r + 1)
    return [i for i, _ in sorted(s.items(), key=lambda x: -x[1])[:k]]
rows = []
for ex in ds:
    titles = ex["context"]["title"]; paras = [" ".join(s) for s in ex["context"]["sentences"]]
    gold = set(ex["supporting_facts"]["title"])
    bm = BM25Okapi([re.findall(r"\\w+", p.lower()) for p in paras]); sc = bm.get_scores(re.findall(r"\\w+", ex["question"].lower()))
    L1 = list(np.argsort(-sc)[:K])
    E = enc.encode(paras, normalize_embeddings=True); q = enc.encode([ex["question"]], normalize_embeddings=True)[0]
    L2 = list(np.argsort(-(E @ q))[:K]); F = rrf([L1, L2])
    rec = lambda L: len(gold & {titles[i] for i in L}) / len(gold)
    rows.append((rec(L1), rec(L2), rec(F), len(set(L1) & set(L2)) / K, float(np.sort(sc)[-1] - np.sort(sc)[-2]), float(np.sort(E @ q)[-1] - np.sort(E @ q)[-2]), rec(F) == 1.0))
A = np.array(rows, float); print("supporting-fact recall@5  bm25 %.3f  dense %.3f  rrf %.3f" % tuple(A[:, :3].mean(0)))
X, y = A[:, 3:6], A[:, 6]
p = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=5, method="predict_proba")[:, 1]
print("AUC of agreement features predicting 'all gold paragraphs retrieved':", round(roc_auc_score(y, p), 3))''')], "01_auric_hotpotqa_retrieval_UNTESTED.ipynb")

# 4) Handshake: add spaCy NER for PERSON (UNTESTED template)
nb([("md", "# Closing the PERSON recall gap with spaCy NER - UNTESTED TEMPLATE\nThe rule-based detector has PERSON recall ~0.65 on held-out templates. This notebook adds spaCy NER on the same synthetic benchmark and reports the change. Needs Internet On."),
    ("code", "!pip -q install faker spacy && python -m spacy download en_core_web_sm"),
    ("code", wf("04_handshake_pii_synth/src/pii.py", "pii.py")),
    ("code", '''import spacy, random, pii
from faker import Faker
nlp = spacy.load("en_core_web_sm")
fk = Faker(["en_US", "en_IN"]); Faker.seed(31337); rnd = random.Random(31337)   # NEW seed: not used in any earlier error analysis
docs = [pii.make_doc(pii.TEST_T, fk, rnd) for _ in range(1500)]
def hybrid(text, **kw):
    spans = [s for s in pii.detect(text) if s[2] != "PERSON"]
    for e in nlp(text).ents:
        if e.label_ == "PERSON": spans.append((e.start_char, e.end_char, "PERSON", e.text))
    return sorted(spans)
orig = pii.detect; base = pii.score(docs)["PERSON"]; pii.detect = lambda t, **kw: hybrid(t)
print("rules only:", base); print("rules + spaCy NER:", pii.score(docs)["PERSON"]); pii.detect = orig''')], "04_handshake_pii_spacy_UNTESTED.ipynb")
print("ok")
