"""Train UniMatch's models from everything in datasets/.   Usage:  python train.py
 1) Admission model  - Random Forest classifier on tseamcet.csv + 2022 + 2021 last-rank statements
 2) Marks -> rank    - Random Forest regressor on marks_vs_rank.xlsx (same approach as your notebook)"""
import glob, json, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
from common import features
from data_prep import load_all, DIST

rng = np.random.default_rng(42)
rec, cls, used = load_all(); print("Loaded:", used)
K = ["college_code", "branch_code"]; G = K + ["caste", "gender", "region"]
groups = rec.groupby(G)["rank"].agg(med="median", lo="min", hi="max", n="count").reset_index()
sm = cls["strength"].median(); cls["strength_f"] = cls["strength"].fillna(sm)
ORDER = ["OC", "EWS", "BC_A", "BC_B", "BC_C", "BC_D", "BC_E", "SC", "ST"]
enc = {"caste": [c for c in ORDER if c in set(rec.caste)] + sorted(set(rec.caste) - set(ORDER)), "gender": sorted(rec.gender.unique()),
       "region": sorted(rec.region.unique()), "branch": sorted(rec.branch_code.unique())}

# ---- 1) admission model ----
d = rec.merge(groups[G + ["med"]], on=G).merge(cls[K + ["fee", "strength_f"]].rename(columns={"fee": "cfee"}), on=K)
d = d.loc[d.index.repeat(4)].reset_index(drop=True)                      # 4 simulated students per record
s = d["med"] * np.exp(rng.normal(0, 0.7, len(d))); y = (s <= d["rank"]).astype(int)
X = features(s, d["med"], d.caste, d.gender, d.region, d.branch_code, d.cfee, d.strength_f, enc)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=15, n_jobs=-1, random_state=42).fit(Xtr, ytr)
p = rf.predict(Xte)
metrics = {"accuracy": accuracy_score(yte, p), "precision": precision_score(yte, p), "recall": recall_score(yte, p), "f1": f1_score(yte, p),
           "records": int(len(rec)), "colleges": int(rec.college_code.nunique()), "sources": used}

# ---- 2) marks -> rank (log-rank target so the wide rank range is handled well) ----
mf = (glob.glob("datasets/marks_vs_rank*.xlsx") or [None])[0]
if mf:
    mk = pd.read_excel(mf).dropna(); mk["rank"] = mk["rank"].astype(float)
    a, b, c, e = train_test_split(mk[["marks"]].values, np.log(mk["rank"].values), test_size=0.2, random_state=2)
    mr = RandomForestRegressor(n_estimators=300, random_state=2).fit(a, c)
    metrics["marks_r2"] = float(r2_score(e, mr.predict(b)))
    mr.fit(mk[["marks"]].values, np.log(mk["rank"].values)); joblib.dump(mr, "models/marks_rf.joblib")
    metrics["marks_range"] = [float(mk.marks.min()), float(mk.marks.max())]
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})

joblib.dump(rf, "models/rf.joblib", compress=3); groups.to_csv("models/groups.csv", index=False); cls.drop(columns="strength_f").to_csv("models/classes.csv", index=False)
dists = sorted({x for x in cls.dist if x in DIST}); places = sorted({p for p in cls.place if p})
bn = cls.drop_duplicates("branch_code").sort_values("branch")
json.dump({"enc": enc, "metrics": metrics, "districts": [{"code": c, "name": DIST[c]} for c in sorted(dists, key=lambda c: DIST[c])], "places": places,
           "branches": [{"code": r.branch_code, "name": r.branch} for r in bn.itertuples()]}, open("models/meta.json", "w"))
print("Saved models to models/")

# ---- 3) JEE model (datasets/JEE_Rank_2016_2024.csv) ----
import train_jee; train_jee.main()
