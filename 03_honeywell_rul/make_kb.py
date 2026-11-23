import json, numpy as np, os
rows, leak, cov = [], [], []
for ds in ["FD001","FD002","FD003","FD004"]:
    p = f"results/metrics_{ds}.json"
    if not os.path.exists(p): continue
    r = json.load(open(p)); m = r["models"]
    ci = {}
    for k in ("lstm","transformer"):
        y, mu, sd = np.load(f"results/pred_{ds}_{k}.npy"); rs = np.random.RandomState(0); n = len(y)
        b = [np.sqrt(np.mean((y[i]-mu[i])**2)) for i in (rs.randint(0,n,n) for _ in range(2000))]
        ci[k] = np.percentile(b, [2.5, 97.5])
    rows.append(f"| {ds} | {r['n_train_engines']} | {m['gbm_baseline']['test_rmse']:.1f} | {m['lstm']['test_rmse']:.1f} ({ci['lstm'][0]:.1f}-{ci['lstm'][1]:.1f}) | {m['transformer']['test_rmse']:.1f} ({ci['transformer'][0]:.1f}-{ci['transformer'][1]:.1f}) | {m['gbm_baseline']['test_nasa_score']:.0f} / {m['lstm']['test_nasa_score']:.0f} / {m['transformer']['test_nasa_score']:.0f} |")
    leak.append(f"| {ds} | {r['leakage_demo']['val_rmse_random_window_split']:.1f} | {r['leakage_demo']['val_rmse_engine_split']:.1f} |")
    cov.append(f"| {ds} | {m['lstm']['ens_interval80_coverage']:.0%} | {m['transformer']['ens_interval80_coverage']:.0%} |")
T = open("KB_TEMPLATE.md").read()
T = T.replace("{{RESULTS}}", "\n".join(rows)).replace("{{LEAK}}", "\n".join(leak)).replace("{{COV}}", "\n".join(cov))
open("KNOWLEDGE_BASE.md", "w").write(T); print(T[:3000])
