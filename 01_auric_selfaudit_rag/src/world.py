"""Synthetic multi-hop world with the retrieval failure modes that matter: near-duplicate entities and conflicting/time-sensitive facts."""
import random, string

SYL = ["ar","vex","lin","dor","mi","sa","ko","tel","bran","qui","zen","ur","pol","tav","eri","mon","gal","fen","ost","rik"]
def word(r, n=None): return "".join(r.choice(SYL) for _ in range(n or r.randint(2, 3))).capitalize()
ROLES = ["engineer", "analyst", "designer", "researcher", "auditor", "architect"]

def build(seed=0, n_people=1500, n_companies=220, n_cities=70, n_countries=14, twin_rate=0.35, relocate_rate=0.3):
    r = random.Random(seed); names = set()
    def uniq(f):
        while True:
            w = f()
            if w not in names: names.add(w); return w
    countries = [uniq(lambda: word(r, 2) + "ia") for _ in range(n_countries)]
    cities = [{"name": uniq(lambda: word(r, 2) + r.choice(["ton", "ville", "burg", "port"])), "country": r.choice(countries)} for _ in range(n_cities)]
    companies = []
    for _ in range(n_companies):
        hq = r.choice(cities); c = {"name": uniq(lambda: word(r, 2) + " " + r.choice(["Labs", "Systems", "Works", "Dynamics"])), "hq0": hq}
        c["hq"] = hq
        if r.random() < relocate_rate:
            c["hq"] = r.choice([x for x in cities if x is not hq]); c["moved"] = r.randint(2019, 2023)
        companies.append(c)
    people = []
    while len(people) < n_people:
        first, last = word(r, 2), word(r, 2); comp = r.choice(companies)
        base = {"first": first, "last": last, "company": comp, "role": r.choice(ROLES), "year": r.randint(2005, 2022)}
        letters = r.sample(string.ascii_uppercase, 2 if r.random() < twin_rate else 1)
        for L in letters:   # twins share first+last, differ only in the middle initial and employer
            p = dict(base, mid=L, full=f"{first} {L}. {last}"); 
            if L != letters[0]: p["company"] = r.choice([c for c in companies if c is not comp])
            if p["full"] not in names: names.add(p["full"]); people.append(p)
    docs = []
    for p in people:
        docs.append({"id": len(docs), "kind": "person", "ent": p["full"], "text": f"{p['full']} is a {p['role']} who joined {p['company']['name']} in {p['year']}. {p['last']} enjoys hiking."})
    for c in companies:
        docs.append({"id": len(docs), "kind": "company", "ent": c["name"], "text": f"{c['name']} is headquartered in {c['hq0']['name']}. It was founded in {r.randint(1980, 2012)}."})
        if "moved" in c:
            docs.append({"id": len(docs), "kind": "company_move", "ent": c["name"], "text": f"In {c['moved']}, {c['name']} relocated its headquarters to {c['hq']['name']}."})
    for ct in cities:
        docs.append({"id": len(docs), "kind": "city", "ent": ct["name"], "text": f"{ct['name']} is located in {ct['country']}. It has {r.randint(20, 900)} thousand residents."})
    r.shuffle(docs)
    for i, d in enumerate(docs): d["id"] = i
    return {"people": people, "companies": companies, "cities": cities, "docs": docs}

def questions(w, n=2000, seed=1):
    r = random.Random(seed); qs = []
    for _ in range(n):
        p = r.choice(w["people"]); hops = r.choice([2, 3]); c = p["company"]
        if hops == 2: q, a = f"In which city is the company that employs {p['full']} headquartered today?", c["hq"]["name"]
        else: q, a = f"In which country is the company that employs {p['full']} headquartered today?", c["hq"]["country"]
        kind = r.choices(["exact", "no_initial", "typo"], [0.65, 0.2, 0.15])[0]
        surf = p["full"]
        if kind == "no_initial": surf = f"{p['first']} {p['last']}"
        if kind == "typo":
            i = r.randrange(len(p["last"])); surf = p["full"].replace(p["last"], p["last"][:i] + r.choice("xyzq") + p["last"][i + 1:])
        n_twins = sum(1 for x in w["people"] if x["first"] == p["first"] and x["last"] == p["last"])
        q = q.replace(p["full"], surf)
        qs.append({"q": q, "surface": surf, "surface_kind": kind, "ambiguous": kind == "no_initial" and n_twins > 1, "answer": a, "hops": hops, "person": p["full"], "company": c["name"], "twin": any(x["first"] == p["first"] and x["last"] == p["last"] and x is not p for x in w["people"][:0]),
                   "relocated": "moved" in c})
    return qs
