# Interview knowledge base: remaining-useful-life prediction on NASA C-MAPSS
*(Target: Honeywell Aerospace and similar predictive-maintenance / time-series ML roles)*

## 0. Read this first: what is real and what is not
- **Everything here is real**: NASA C-MAPSS turbofan simulation data (public NASA Prognostics Repository, Saxena and Goebel 2008), trained and tested in this workspace, 3 seeds per model, results in `results/`.
- It is **simulated** engine data, not field data. Say so. Real fleets have missing sensors, maintenance events and censoring.
- Code was built with AI assistance and is original (not copied from the reference repos). Be able to explain it.

## 1. The 60-second pitch
"I predicted remaining useful life for turbofan engines on all four C-MAPSS subsets. I compared a gradient-boosting baseline on window summary features against an LSTM and a small Transformer, with per-operating-condition normalisation for the multi-regime subsets and engine-level validation so no engine appears in both train and validation. I also showed why that matters: a random window split gives validation RMSE of 7.6 to 14 where the honest engine-level number is 16 to 18. The LSTM ensemble reached 13.3 to 14.9 RMSE on the four official test sets and beat the baseline everywhere. The honest weakness: seed-ensemble uncertainty intervals covered only 19-44% of true values at a nominal 80%, so I'd add conformal prediction."

## 2. Method and why
1. **Piecewise-linear target (cap 125 cycles)**: early in life degradation is invisible, so exact RUL is unlearnable; capping stops the model chasing noise and is standard in the literature.
2. **Sensor selection**: drop sensors with ~zero variance within an operating condition (fit on train only).
3. **Per-condition normalisation**: FD002/FD004 run under 6 operating regimes that shift sensor means more than degradation does. I cluster the three operating settings with k-means (k=6, fit on train) and standardise each sensor within its cluster. FD001/FD003 have one regime.
4. **Windows**: 30 cycles; test engines shorter than 30 are padded by repeating the first row. Test uses the **last window** of each test engine, matching the official protocol.
5. **Models**: HistGradientBoosting on [last, mean, std, delta] window features; 2-layer LSTM (hidden 64, dropout 0.2); Transformer encoder (d=64, 4 heads, 2 layers, learned positions, mean pooling). AdamW, cosine schedule, early stopping on **validation engines** (20% of train engines, split by engine).
6. **Ensembling**: 3 seeds averaged; spread used for an uncertainty interval.
7. **Metrics**: RMSE and the asymmetric **NASA score** (late predictions are penalised more than early ones: exp(d/10)-1 for d>=0, exp(-d/13)-1 for d<0).

## 3. Real results (test sets, RMSE in cycles; brackets = 95% bootstrap CI over test engines)
| Dataset | Train engines | GBM | LSTM ensemble | Transformer ensemble | NASA score GBM / LSTM / Transformer |
|---|---|---|---|---|---|
| FD001 | 100 | 16.3 | 14.4 (12.0-16.8) | 15.8 (12.9-18.6) | 565 / 418 / 530 |
| FD002 | 260 | 17.1 | 14.6 (13.0-16.2) | 14.6 (13.0-16.1) | 1581 / 1192 / 1118 |
| FD003 | 100 | 16.5 | 13.3 (10.5-15.9) | 14.9 (12.4-17.3) | 803 / 316 / 463 |
| FD004 | 249 | 17.8 | 14.9 (13.1-16.7) | 16.0 (14.0-18.0) | 1963 / 1205 / 1380 |

Wide intervals mean differences between LSTM and Transformer are **not statistically clear**; the consistent finding is that both deep models beat the GBM baseline on RMSE and NASA score in all four subsets, and the LSTM was better than or equal to the Transformer in every subset.

**Leakage demo** (same GBM, validation RMSE):

| Dataset | Random window split | Engine-level split |
|---|---|---|
| FD001 | 10.1 | 17.7 |
| FD002 | 14.0 | 17.9 |
| FD003 | 7.6 | 16.2 |
| FD004 | 12.5 | 17.6 |

Random splits put neighbouring windows of the same engine on both sides, so validation looks far better than reality.

**Uncertainty (nominal 80% interval from seed spread) actual coverage:**

| Dataset | LSTM | Transformer |
|---|---|---|
| FD001 | 24% | 30% |
| FD002 | 21% | 44% |
| FD003 | 30% | 31% |
| FD004 | 19% | 39% |

## 4. What went wrong / honest findings
1. **Seed-ensemble intervals are far too narrow.** They capture model (epistemic) variance only; most error is irreducible noise and shared bias across seeds. Fix: split-conformal prediction using validation residuals, or heteroscedastic output heads. I did not implement this; I chose to report the failure.
2. **Validation RMSE (~16-18) is worse than test RMSE (~13-15)** because validation scores every window of an engine (including early-life windows, where the capped target is hard) whereas the test protocol scores one window per engine near the end. Do not compare the two numbers directly.
3. **The Transformer did not clearly beat the LSTM.** With ~100 training engines a small LSTM is as good or better; attention needs more data or longer windows.
4. **I tuned nothing heavily**: single hyperparameter setting, 25-30 epochs, 3 seeds. Published results with tuning and longer windows are better on some subsets; do not claim state of the art.
5. A first smoke run with 1 seed gave interval coverage of 0% because spread was zero. A reminder that uncertainty from one run is meaningless.

## 5. Limitations to state yourself
- Simulated data, 4 fault/regime combinations; no censoring or maintenance events.
- No hyperparameter search, no feature selection ablation, no per-engine failure-mode analysis.
- NASA score is dominated by a few badly late predictions; report it with the RMSE and look at the worst engines.

## 6. Likely questions and strong answers
**Q: Why cap RUL at 125?** Before degradation starts, sensors carry no information about time to failure, so a linear target forces the model to guess. A piecewise target says "healthy beyond ~125 cycles".
**Q: Why engine-level validation?** Windows from one engine are highly correlated. If some are in train and some in validation, the model memorises the engine. Split by group (engine). My demo shows 7.6-14 vs 16-18 RMSE.
**Q: Explain the NASA score.** Asymmetric exponential penalty: late predictions (predicting more life than remains) cost exp(d/10)-1, early ones exp(-d/13)-1. Safety logic: late is worse than early. It is sensitive to outliers.
**Q: Why per-condition normalisation?** In FD002/FD004, altitude/Mach/throttle regimes shift sensor values far more than wear. Normalising within regime isolates the degradation trend.
**Q: LSTM vs Transformer here?** Both capture temporal order; the LSTM has a stronger inductive bias for short sequences and small data. With 100-250 training engines the extra capacity of attention doesn't pay off; I'd test longer windows, convolutional front-ends, or pretraining.
**Q: How would you get honest uncertainty?** Split-conformal on validation residuals gives distribution-free coverage under exchangeability; or quantile regression; or MC dropout plus a calibrated scale. Check coverage on test engines.
**Q: How would you deploy this?** Stream sensor cycles, maintain a rolling 30-cycle buffer, score each cycle, smooth the RUL estimate, alert when the interval's lower bound crosses a maintenance threshold. Monitor drift per regime.
**Q: What about censored data?** Real engines are removed before failure. Use survival models (Cox, discrete-time hazard, DeepSurv) or treat the removal time as a lower bound.
**Q: Why did GBM do OK?** Hand-built window features (last, mean, std, slope) capture the degradation trend; deep models win mainly through better handling of the full sequence and regime effects.
**Q: Where would a Honeywell-type team push further?** Fleet-level transfer across engine types, physics-informed features, anomaly detection for unseen faults, explainability for maintenance engineers.

## 7. Files
`src/rul.py` (everything), `data/` (NASA files), `results/metrics_FD00x.json`, `results/pred_FD00x_*.npy` (y, mean, std per test engine), `make_kb.py` (regenerates the tables above).
