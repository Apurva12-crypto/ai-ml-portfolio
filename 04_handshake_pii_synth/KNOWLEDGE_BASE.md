# Interview knowledge base: PII scrubbing + synthetic-data quality pipeline
*(Target: Handshake AI MTS and similar data / LLM-evaluation roles)*

## 0. Read this first: what is real and what is not
- Everything here ran in my workspace. All data is **synthetic** (Faker, Luhn/Verhoeff-valid identifiers) with gold span labels I generated.
- Synthetic benchmarks are easy and my generator and detector were designed by the same person, so near-perfect scores prove the **pipeline and the method**, not real-world accuracy. Say this before the interviewer does.
- Code was built with AI assistance. Be able to explain every function below. If asked, say so plainly.

## 1. The 60-second pitch
"I built a PII detector and a synthetic-data quality pipeline. The detector layers regex candidates, checksum validators (Luhn for cards, Verhoeff for Aadhaar, SSN rules, IP octet ranges) and same-clause context rules. I measured each layer with an ablation on held-out templates. Precision went from 0.69 to 0.90 with validators and to 1.0 with context, while recall rose from 0.70 to 0.90. The weak point is names (PERSON recall 0.65) because rules cannot do that job, so production needs an NER model. The second part generates verifiable tasks, injects known defects and measures whether the filters catch them."

## 2. Architecture
1. **Candidates**: one regex per type (email, card, SSN, Aadhaar, PAN, IP, phone, DOB). High recall, low precision.
2. **Validators** kill look-alikes: Luhn checksum for cards (13+ digits), Verhoeff for 12-digit Aadhaar, SSN area/group/serial rules (no 000/666/9xx, no 00 group, no 0000 serial), IP octets <= 255.
3. **Context**: DOB and phone only count if a cue word ("born", "call", "mobile") sits in the **same clause** (split at `. ; newline`). Names are taken after cues ("my name is", "Dear", "Regards,", "Mr.").
4. **Overlap resolution**: priority table (card/Aadhaar/SSN/PAN/email > IP > DOB/PERSON > phone), longer span wins.
5. **Synthetic-data pipeline**: task generator with a programmatic gold answer (arithmetic, unit conversion, percentages, date wrap) -> defect injector -> filter stack (verifier, PII filter, truncation, topical overlap, near-duplicate) -> metrics. An `LLMJudge` class is a drop-in for subjective quality; it was **not run** (no API access in the build).

## 3. Real results (all on synthetic data)
**PII, held-out templates (1,500 docs, 5,002 gold spans, final seed 2024)**

| Stage | Precision | Recall | F1 |
|---|---|---|---|
| Regex only | 0.694 | 0.700 | 0.697 |
| + validators | 0.897 | 0.700 | 0.786 |
| + context rules | **1.000** | **0.896** | **0.945** |

Per type in the final run: card, DOB, email, Aadhaar, PAN, IP, phone, SSN all F1 = 1.0 on this data. **PERSON: precision 1.0, recall 0.653.**

**Quality filter (3,000 examples, 40% injected defects)**: recall 1.0 for wrong answer, truncation, PII leak, off-topic and duplicate; 0 false rejections of clean items; surviving defect rate 0%.

## 4. What went wrong (this is your best interview material)
1. **I burned my test set twice and said so.** v1 test (seed 7): micro F1 0.857, with IP precision 0.40 because version strings like `3.10.4.2` are valid IPv4. I added a version/build/release context rule. v2 (fresh seed 99): F1 0.888; error analysis showed two more problems. v3 (fresh seed 2024): F1 0.945. Each round used a new seed because tuning on a test set you have looked at inflates results. The first-run numbers are kept in `results/pii_metrics_v1_first_run.json`.
2. **Two bugs were in my benchmark, not the detector.** (a) Faker's `name()` sometimes returned "Dr." as the first word, so gold labels were wrong. (b) I appended "Notes:" directly after a signature, so the name regex swallowed the word "Notes". Fixes: generate names from first/last name, put notes on a new line. Lesson: when a metric looks bad, audit the labels first.
3. **Phone false positives came from context leakage**: the 40-char look-back window crossed into the previous sentence and picked up "WhatsApp" for a tracking number. Fix: truncate the window at sentence boundaries.
4. **Filter bugs, v1 -> final**: off-topic recall was 0.0 because my own `if "truncated" not in reasons` guard suppressed it (off-topic text has no "The answer is", so it was always flagged truncated first). Duplicate recall was 0.734 because the benchmark copied from records that had themselves been corrupted, which is undetectable by construction. I fixed the benchmark to copy only clean records. Files: `synth_filter_metrics_v1_with_bugs.json` vs `synth_filter_metrics.json`.
5. A flag worth discussing: off-topic items were also flagged "duplicate" 48 times, because I used three canned off-topic replies. Real off-topic data is more diverse.

## 5. Limitations to state yourself
- Synthetic, template-based text. Real text has typos, OCR noise, multilingual names, free-form addresses (not covered at all).
- Rule-based PERSON detection is weak. A hybrid (rules for structured IDs, NER for names) is the right design; `notebooks/04_handshake_pii_spacy_UNTESTED.ipynb` tests that on Kaggle but I have not run it.
- Filters hit 100% because injected defects are crisp. Real defects (subtle reasoning errors, tone, hallucinated citations) need an LLM judge calibrated against human labels.
- No locale coverage beyond US and Indian formats.

## 6. Likely questions and strong answers
**Q: Why validators and not just regex?** Digit patterns collide with order IDs, tracking numbers and versions. Checksums remove ~90% of random 16-digit strings (Luhn catches 9 in 10) and about 90% of random 12-digit Aadhaar-shaped numbers (Verhoeff accepts 1 in 10 by chance). Ablation: validators lifted precision 0.69 -> 0.90 at no recall cost.
**Q: How does the Luhn algorithm work?** From the right, double every second digit, subtract 9 if the result exceeds 9, sum everything, valid if sum % 10 == 0.
**Q: What is Verhoeff?** A checksum using dihedral group D5 multiplication and permutation tables; it catches all single-digit errors and adjacent transpositions, which Luhn does not. Used for Aadhaar.
**Q: Precision or recall first for PII?** Depends on the use. For training-data scrubbing, recall (a leak is costly); for user-facing masking, balance with precision so you don't destroy the text. Say which you'd pick and why.
**Q: How do you evaluate a span detector?** Exact span match per type, then micro and per-type P/R/F1; also overlap-based scoring if boundaries are fuzzy. I used exact match, which is strict.
**Q: How would you improve PERSON?** NER (spaCy, a fine-tuned transformer), a gazetteer for locale names, and a model that sees context windows. Measure recall on held-out names the model never saw.
**Q: How do you avoid overfitting rules to your test set?** Dev templates vs held-out templates, and a fresh seed after every error-analysis round; I report the progression.
**Q: Why programmatic gold answers for synthetic data?** Verifiable tasks let you reject wrong samples without a judge model and give a ground truth to measure the judge against.
**Q: What can an LLM judge catch that rules can't?** Fluency, helpfulness, subtle reasoning flaws, tone, hallucinated facts. Its weaknesses: position and length bias, self-preference, inconsistency. Calibrate against human labels and measure agreement (Cohen's kappa) before trusting it.
**Q: How do you detect near-duplicates at scale?** Word 3-gram shingles + Jaccard here (O(n^2) in my demo); production uses MinHash + LSH.
**Q: How would you measure the effect of filtering on a model?** Fine-tune a small model on filtered vs unfiltered data and compare on a held-out benchmark. I did not do this; say so.
**Q: Privacy risk of the pipeline itself?** Detection logs should not store raw matches; hash or drop them. The demo API should never receive real PII.

## 7. Cheat sheet
- Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = harmonic mean.
- Micro averages pool counts across types; macro averages per-type scores.
- Held-out discipline: tune on dev, report on data you have not looked at.
- Files: `src/pii.py`, `src/synth.py`, `results/*.json`, `app/api.py` (redaction API).
