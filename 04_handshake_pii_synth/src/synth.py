"""Synthetic instruction data + quality-filter pipeline, evaluated by defect injection. Original implementation."""
import random, re, json, math, os
from collections import Counter
from faker import Faker
from pii import detect

# ---------- 1. verifiable task generator (gold answer known by construction) ----------
def make_task(rnd, fk):
    k = rnd.choice(["arith", "unit", "pct", "date"])
    if k == "arith":
        a, b, c = rnd.randint(3, 40), rnd.randint(2, 12), rnd.randint(5, 90)
        q = f"A shop sells {a} boxes with {b} pens each, then adds {c} loose pens. How many pens in total?"
        return {"kind": k, "prompt": q, "answer": a * b + c,
                "response": f"Each box has {b} pens, so {a} boxes hold {a*b} pens. Adding {c} loose pens gives {a*b+c} pens in total. The answer is {a*b+c}."}
    if k == "unit":
        km = rnd.randint(2, 300)
        q = f"A trip is {km} kilometres long. How many metres is that?"
        return {"kind": k, "prompt": q, "answer": km * 1000,
                "response": f"One kilometre is 1000 metres, so {km} km is {km*1000} metres. The answer is {km*1000}."}
    if k == "pct":
        p, base = rnd.choice([10, 20, 25, 40, 50]), rnd.choice([80, 120, 200, 360, 500])
        ans = base * p // 100
        q = f"What is {p}% of {base}?"
        return {"kind": k, "prompt": q, "answer": ans,
                "response": f"{p}% means {p}/100, so {p}/100 times {base} equals {ans}. The answer is {ans}."}
    d = rnd.randint(2, 45); n = rnd.randint(1, 28)
    q = f"A parcel ships {d} days after the {n}th of a 30-day month. On which day of the month does it ship if the month wraps around?"
    ans = (n + d - 1) % 30 + 1
    return {"kind": k, "prompt": q, "answer": ans,
            "response": f"Counting {d} days from day {n} in a 30-day month and wrapping gives day {ans}. The answer is {ans}."}

# ---------- 2. defect injection (ground truth for judging the judge) ----------
DEFECTS = ["wrong_answer", "truncated", "pii_leak", "off_topic", "duplicate"]
def inject(ex, defect, rnd, fk, pool):
    ex = dict(ex); ex["defect"] = defect
    if defect == "wrong_answer":
        wrong = ex["answer"] + rnd.choice([-10, -3, -1, 1, 2, 7, 10])
        ex["response"] = re.sub(r"\d+\.?$", f"{wrong}.", ex["response"].rstrip()) if False else ex["response"].replace(f"The answer is {ex['answer']}.", f"The answer is {wrong}.")
    elif defect == "truncated":
        ex["response"] = ex["response"][: rnd.randint(18, max(19, len(ex["response"]) // 2))]
    elif defect == "pii_leak":
        ex["response"] += f" Contact {fk.first_name()} {fk.last_name()} at {fk.email()} or call {rnd.randint(200,999)}-{rnd.randint(200,999)}-{rnd.randint(1000,9999)} for details."
        ex["response"] = ex["response"].replace("Contact", "Contact", 1)
    elif defect == "off_topic":
        ex["response"] = rnd.choice(["The best way to brew coffee is to use freshly ground beans and water just below boiling.",
                                     "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
                                     "Remember to back up your photos regularly to avoid losing precious memories."])
    elif defect == "duplicate" and pool:
        src = rnd.choice(pool); ex["prompt"], ex["response"], ex["answer"] = src["prompt"] + " ", src["response"], src["answer"]
    return ex

# ---------- 3. filters ----------
def shingles(t, k=3):
    w = re.findall(r"\w+", t.lower()); return {" ".join(w[i:i + k]) for i in range(max(1, len(w) - k + 1))}
def jacc(a, b): return len(a & b) / max(len(a | b), 1)
def tok(t): return Counter(re.findall(r"[a-z0-9]+", t.lower()))
def cos(a, b):
    num = sum(a[k] * b[k] for k in a if k in b); den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0
STOP = set("a an the of to in is are on and for how what many which does do if then it its at as by with".split())

class RuleJudge:
    """Offline judge: deterministic checks. Stands in for the LLM judge on structure/verifiable defects."""
    def __init__(self, dup_thresh=0.8, rel_thresh=0.12):
        self.seen = []; self.dt = dup_thresh; self.rt = rel_thresh
    def verdict(self, ex):
        r = ex["response"]; reasons = []
        if not re.search(r"[.!?]$", r.strip()) or "The answer is" not in r: reasons.append("truncated")
        m = re.search(r"The answer is (-?\d+)", r)
        if m and int(m.group(1)) != ex["answer"]: reasons.append("wrong_answer")
        if detect(r): reasons.append("pii_leak")
        pc = {w: c for w, c in tok(ex["prompt"]).items() if w not in STOP}; rc = {w: c for w, c in tok(r).items() if w not in STOP}
        if cos(pc, rc) < self.rt: reasons.append("off_topic")
        sh = shingles(ex["prompt"] + " " + r)
        if any(jacc(sh, s) >= self.dt for s in self.seen): reasons.append("duplicate")
        self.seen.append(sh)
        return reasons

class LLMJudge:
    """Drop-in LLM judge (not executed in the offline benchmark). Needs a client with .complete(prompt)->str.
    Use it for what rules cannot see: tone, helpfulness, subtle reasoning errors. Calibrate it against RuleJudge labels first."""
    TEMPLATE = ("You are a strict data-quality reviewer. Given PROMPT and RESPONSE, list every defect from "
                "[wrong_answer, truncated, pii_leak, off_topic, duplicate, other] as JSON {{\"defects\": [...], \"reason\": \"...\"}}.\n"
                "PROMPT: {p}\nRESPONSE: {r}")
    def __init__(self, client): self.client = client
    def verdict(self, ex):
        out = self.client.complete(self.TEMPLATE.format(p=ex["prompt"], r=ex["response"]))
        try: return json.loads(out).get("defects", [])
        except Exception: return ["unparseable_judge_output"]

def run(n=3000, defect_rate=0.4, seed=11):
    rnd = random.Random(seed); fk = Faker(); Faker.seed(seed)
    data, pool = [], []
    seen_prompts = set()
    while len(data) < n:
        ex = make_task(rnd, fk)
        if ex["prompt"] in seen_prompts: continue          # generation-time dedup of clean data
        seen_prompts.add(ex["prompt"])
        if rnd.random() < defect_rate: ex = inject(ex, rnd.choice(DEFECTS), rnd, fk, pool[-200:])   # duplicates copy a CLEAN earlier record
        else: ex["defect"] = None; pool.append(dict(ex))
        data.append(ex)
    # order matters for the duplicate check: originals must precede copies
    J = RuleJudge(); conf = {d: Counter() for d in DEFECTS}; stats = {d: {"tp": 0, "fn": 0} for d in DEFECTS}; fp = Counter(); clean_n = clean_rej = 0
    kept = []
    for ex in data:
        v = J.verdict(ex)
        if ex["defect"] is None:
            clean_n += 1; clean_rej += bool(v)
            for d in v: fp[d] += 1
        else:
            stats[ex["defect"]]["tp" if ex["defect"] in v else "fn"] += 1
            for d in v: conf[ex["defect"]][d] += 1
        if not v: kept.append(ex)
    res = {"n": n, "defect_rate": defect_rate, "clean_examples": clean_n, "clean_rejected": clean_rej,
           "false_reject_rate": round(clean_rej / clean_n, 4),
           "per_defect_recall": {d: round(s["tp"] / max(s["tp"] + s["fn"], 1), 3) for d, s in stats.items()},
           "false_flags_on_clean_by_reason": dict(fp),
           "flags_fired_per_injected_defect": {d: dict(c) for d, c in conf.items()},
           "defective_examples_that_survived": sum(1 for e in kept if e["defect"]),
           "kept": len(kept), "kept_defect_rate": round(sum(1 for e in kept if e["defect"]) / max(len(kept), 1), 4),
           "note": "Synthetic defect-injection benchmark; defects are simple and independent, so recall here is an upper bound on real-world performance."}
    return res, kept[:30]

if __name__ == "__main__":
    res, sample = run()
    json.dump(res, open("results/synth_filter_metrics.json", "w"), indent=1)
    json.dump(sample, open("results/synth_kept_sample.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
