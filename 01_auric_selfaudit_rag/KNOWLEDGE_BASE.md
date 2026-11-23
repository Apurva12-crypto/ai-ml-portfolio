# Interview knowledge base: self-auditing multi-hop retrieval with calibrated abstention
*(Target: Auric Labs research role and similar RAG / agent-reliability roles)*

## 0. Read this first: what is real and what is not
- This is a **controlled synthetic testbed** I built to isolate retrieval failure modes: twin entities (same first and last name, different middle initial and employer), relocated companies (conflicting, time-sensitive facts) and noisy name mentions (typos, missing initial).
- The **reader is extractive and rule-based**, not an LLM. So these numbers say nothing about LLM answer quality or real corpora. They test retrieval, evidence selection, agreement-based confidence and abstention.
- Do **not** present these numbers as results on HotpotQA or as the figures on your resume. `notebooks/01_auric_hotpotqa_retrieval_UNTESTED.ipynb` is a template for a real-data check; I could not run it.
- Code was built with AI assistance. Be ready to explain everything.

## 1. The 60-second pitch
"I studied when multi-hop retrieval fails and whether a system can know it is failing. I built a synthetic corpus with near-duplicate entities and conflicting dated facts, ran three retrieval paths (BM25, a dense LSA index, an exact entity index), fused them with reciprocal rank fusion, and used cross-path agreement as a confidence signal. A logistic calibrator on top cut area under the risk-coverage curve by roughly 2x and let the system abstain on about 18% of questions, raising accuracy on answered ones from 88.6% to 93.8%. A surprising finding: simple fusion did not beat BM25, and I report that."

## 2. Setup
- 1,500 people (35% have a twin), 220 companies (30% relocated), 70 cities, 14 countries; 1,846 documents; 2,400 questions (2-hop: city of employer HQ; 3-hop: country). Each question has name form: exact 65%, missing initial 20% (ambiguous if a twin exists), typo 15%.
- Retrieval: top-K = 3 documents per hop (small K creates pressure).
- Paths: BM25 (own implementation, k1=1.2, b=0.75); dense = TF-IDF + truncated SVD (LSA) as a stand-in for an embedding model; graph = exact-entity index, with a first+last-name fallback that returns all twins.
- Fusion: reciprocal rank fusion, score = sum 1/(60 + rank).
- Evidence rule: prefer docs whose entity string matches exactly; among candidates the **latest dated** fact wins (relocation overrides the founding HQ).
- Confidence features: minimum and mean cross-path agreement, maximum number of distinct candidate answers, minimum exact-match flag, number of hops. Logistic regression fit on the first 800 questions, evaluated on the other 1,600.

## 3. Real results (world seed 0; seeds 1 and 2 in `results/seed1`, `seed2`)
| System | Accuracy | Exact-name | Typo | Twin-ambiguous |
|---|---|---|---|---|
| BM25 | 0.897 | 1.000 | 0.704 | 0.449 |
| Dense (LSA) | 0.654 | 0.805 | 0.352 | 0.331 |
| Fused (RRF) | 0.893 | 1.000 | 0.654 | 0.472 |

Across 3 world seeds fused accuracy was 0.893 / 0.903 / 0.900 versus BM25 0.897 / 0.910 / 0.915, so **fusion did not help**, and the weak dense path hurt typo handling.

Calibration and abstention (fused, seed 0, 1,600 test questions): ECE raw agreement 0.038 -> calibrated 0.028; **AURC 0.0405 -> 0.0208**; selective accuracy 88.6% at full coverage, 94.6% at 80%, 98.0% at 70%; abstain threshold chosen on dev for 95% target gives 17.9% abstention and 93.8% accuracy on answered questions; 101 wrong answers avoided, 186 correct answers lost. AURC across seeds: raw 0.041 to 0.049 versus calibrated 0.017 to 0.021.

## 4. What went wrong and what I learned
1. **First version had 100% accuracy**: the exact-entity path trivially solved every question, so nothing could fail. I added noisy mentions and twins, which exposed real failure modes. A benchmark that cannot fail teaches nothing.
2. **Fusion hurts when one path is weak**: RRF gives the dense path equal weight, and it pulls twin or noisy docs into the top-3. Fixes to try: learned or per-path weights, a reranker, or dropping paths whose self-agreement is low.
3. **Abstention is imperfect on exactly the case it should catch**: it abstained on only 38% of twin-ambiguous questions (and 0% of exact-name questions, which is right). My features do not directly encode "two entities matched this name". Next step: add a feature for the number of distinct entities matching the query surface form.
4. **Cost of abstaining**: 186 correct answers lost to avoid 101 wrong ones at that threshold. Choosing a threshold is a business decision (cost of a wrong answer vs cost of a refusal).
5. Hop-level evidence recall (exact-entity documents in the top-3): bm25 0.862, dense 0.777, graph 0.862. BM25 and graph tie because this metric counts only exact-entity documents.

## 5. Limitations to state yourself
- Synthetic and templated: question types are parsed with rules; no LLM decomposition.
- One corpus family, 3 seeds, no confidence intervals beyond that.
- LSA is not a neural embedding model. A real dense retriever may behave differently on near-duplicates.
- Agreement is correlated across paths (they read the same text), so it can be confidently wrong when all paths miss in the same way.

## 6. Likely questions and strong answers
**Q: Why does multi-hop retrieval fail?** (1) The second-hop query depends on the first answer, so a miss compounds; (2) near-duplicate entities collapse in embedding space; (3) chunking splits evidence; (4) conflicting sources need provenance and recency. Errors multiply across hops: 0.9^3 = 0.73.
**Q: What is reciprocal rank fusion?** Score each doc by sum over rankers of 1/(c + rank), c=60. Needs no score calibration across rankers. Weakness: treats all rankers as equally good.
**Q: What is ECE?** Bin predictions by confidence; ECE = sum over bins of (bin share) x |mean confidence - accuracy|. Lower is better calibrated. Limitations: depends on the number of bins, noisy with small bins.
**Q: What is AURC / a risk-coverage curve?** Sort answers by confidence, answer the top x%, plot error rate vs coverage. AURC is the area under it; lower is better. It measures how well confidence ranks correct above incorrect, which is what abstention needs.
**Q: Why calibrate on a different split from the test?** Fitting and scoring on the same data overstates calibration. I fit on 800 and tested on 1,600.
**Q: How would you estimate retrieval recall without labels?** Use independent paths: if different retrievers agree, evidence is more likely found. Better: sample documents from the corpus, build questions with known evidence, and measure recall on them. Caveat: paths that share failure modes agree wrongly.
**Q: How would you swap in an LLM?** The `read()` function is the plug-in point: replace it with an LLM that gets the retrieved passages and returns an answer plus cited spans, keep the agreement features, and add the LLM's own signals (self-consistency across samples, token log-probs). Calibrate the confidence again.
**Q: Why abstain at all?** A wrong answer delivered with high confidence is worse than none in high-stakes settings. Selective prediction trades coverage for reliability.
**Q: How do you handle conflicting sources?** Keep both with provenance and dates; I pick the latest dated fact for time-sensitive questions and expose the conflict as a feature. Better: present both when the question does not fix a time.
**Q: Why did fusion not beat BM25 here? Is that a bug?** I checked: exact-name questions are 100% for both, so the gap sits in typo and twin cases where the dense path adds noise. It is a legitimate finding about equal-weight fusion on this data, not evidence that fusion is bad in general.
**Q: What would you do next?** (1) Run the HotpotQA template on a GPU. (2) Learned fusion weights. (3) A twin-count feature. (4) Conformal prediction for a coverage guarantee. (5) Replace the reader with an LLM and measure end-to-end.

## 7. Files
`src/world.py` (data generator), `src/audit.py` (retrievers, reader, calibration, metrics), `results/metrics.json`, `results/seed1|seed2`, `results/failure_examples_fused.json`.
