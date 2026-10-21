"""Train the JEE (JoSAA) Random Forest.   Usage:  python train_jee.py [path/to/JEE_Rank_2016_2024.csv]
Each CSV row = closing/opening rank of one program, seat type and gender pool in one year (2016-2024).
Rows are grouped per (institute, program, seat type, pool); students are simulated around each group's recent
closing ranks and labelled admitted / not admitted, and the Random Forest learns P(admission)."""
import glob, json, sys, numpy as np, pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import jee_model as jm

def main(csv=None):
    csv = csv or (glob.glob("datasets/*JEE*Rank*.csv") or [None])[0]
    if not csv: print("JEE dataset not found in datasets/ - skipping JEE training"); return
    d = pd.read_csv(csv); d.columns = [c.strip() for c in d.columns]; d = d.dropna(subset=["Institute", "Academic_Program_Name", "Seat_Type"])
    for c in ["Closing_Rank", "Opening_Rank"]: d[c] = pd.to_numeric(d[c], errors="coerce")       # preparatory ranks like '4P' become NaN and are dropped
    d = d.dropna(subset=["Closing_Rank"]); d = d[(d.Closing_Rank > 0) & ~d.Seat_Type.str.contains("PwD")]
    d["inst"] = d.Institute.map(jm.canon); d["prog"] = d.Academic_Program_Name; d["seat"] = d.Seat_Type
    d["pool"] = np.where(d.Gender.str.startswith("Female"), "F", "N"); d["fam"] = d.prog.map(jm.family); d["type"] = d.inst.map(jm.itype)
    d["dur"] = d.prog.str.extract(r"\((\d) Years")[0].astype(float).fillna(4); d = d.sort_values("Year")
    K = ["inst", "prog", "seat", "pool"]; rows = []
    for key, x in d.groupby(K):
        c = x.Closing_Rank.values; y = x.Year.values
        rows.append(key + (float(np.median(c[-3:])), float(c[-1]), int(y[-1]), float(x.Opening_Rank.values[-1]), float(c.min()), float(c.max()), len(c),
                           " · ".join(f"{int(a)}: {int(b):,}" for a, b in zip(y[-4:], c[-4:])), x.fam.iloc[0], x.type.iloc[0], x.dur.iloc[0]))
    groups = pd.DataFrame(rows, columns=K + ["ref", "last", "lastyear", "lastopen", "lo", "hi", "n", "trend", "fam", "type", "dur"])
    enc = {"seat": sorted(d.seat.unique()), "pool": ["N", "F"], "fam": [f[0] for f in jm.FAMS] + ["OTHER"], "inst": sorted(d.inst.unique())}
    t = d.merge(groups[K + ["ref"]], on=K); t = t.loc[t.index.repeat(np.where(t.Year >= 2022, 2, 1) * 3)].reset_index(drop=True)   # recent years count double
    s = t.ref * np.exp(np.random.default_rng(42).normal(0, 0.7, len(t))); y = (s <= t.Closing_Rank).astype(int)
    X = jm.features(s, t.ref, t.seat, t.pool, t.fam, t.inst, t.dur, enc)
    a, b, c, e = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=10, n_jobs=-1, random_state=42).fit(a, c); p = rf.predict(b)
    fc = d.drop_duplicates(["prog"]).fam.value_counts()
    metrics = {"accuracy": accuracy_score(e, p), "precision": precision_score(e, p), "recall": recall_score(e, p), "f1": f1_score(e, p), "records": int(len(d)),
               "institutes": int(d.inst.nunique()), "years": [int(d.Year.min()), int(d.Year.max())]}
    print("JEE:", {k: round(v, 3) if isinstance(v, float) else v for k, v in metrics.items()})
    joblib.dump(rf, "models/jee_rf.joblib", compress=3); groups.to_csv("models/jee_groups.csv", index=False)
    json.dump({"enc": enc, "metrics": metrics, "types": sorted(set(d.type)), "families": [[c, l] for c, l, _ in jm.FAMS if c in fc.index]}, open("models/jee_meta.json", "w"))
    print("Saved JEE model to models/")

if __name__ == "__main__": main(sys.argv[1] if len(sys.argv) > 1 else None)
