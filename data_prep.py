"""Loads and merges every dataset in datasets/ into one table of historical closing ranks.
  tseamcet.csv                       long format (one row per seat-group closing rank)
  tseamcet_2022_2.csv                2022 last-rank statement (wide: one column per category/gender)
  TSEAMCET_2021_*.xlsx               2021 first-phase last-rank statement (wide, converted from PDF)"""
import glob, os, re, numpy as np, pandas as pd
D = "datasets"
CATS = [(c, g) for c in ["OC", "BC_A", "BC_B", "BC_C", "BC_D", "BC_E", "SC", "ST", "EWS"] for g in "MF"]
DIST = {"HNK": "Hanamkonda", "HYD": "Hyderabad", "JGN": "Jangaon", "JTL": "Jagtial", "KGM": "Bhadradri Kothagudem", "KHM": "Khammam", "KMR": "Kamareddy",
        "KRM": "Karimnagar", "MBN": "Mahabubnagar", "MDL": "Medchal-Malkajgiri", "MED": "Medak", "NLG": "Nalgonda", "NZB": "Nizamabad", "PDL": "Peddapalli",
        "RR": "Rangareddy", "SDP": "Siddipet", "SRC": "Rajanna Sircilla", "SRD": "Sangareddy", "SRP": "Suryapet", "WGL": "Warangal", "WNP": "Wanaparthy", "YBG": "Yadadri Bhuvanagiri"}
SMALL = {"AND", "OF", "FOR", "THE", "IN", "AT"}
KEEP = {"CBIT", "GRIET", "VNRVJIET", "MLRIT", "MGIT", "CMRIT", "CMRTC", "CMREC", "CMRCET", "BVRIT", "MRCET", "SNIST", "KMIT", "GNITS", "VJIT", "NGIT", "MVSR",
        "JNTUH", "JNTU", "IARE", "VBIT", "MRIT", "SRIT", "AIML", "IOT", "CSE", "ECE", "EEE", "SVS", "TKR", "VCE", "MRM"}

def smart_title(s):
    out = []
    for w in re.sub(r"\s+", " ", str(s)).strip().split(" "):
        core = re.sub(r"[^A-Z]", "", w.upper())
        if core in SMALL: out.append(w.lower())
        elif core in KEEP or (core and not re.search("[AEIOU]", core)): out.append(w.upper())
        else: out.append(re.sub(r"[A-Za-z]+", lambda m: m.group(0).capitalize(), w))
    r = " ".join(out); return r[:1].upper() + r[1:]

def _wide(df, code, branch, extra):
    """wide table -> (records, class metadata)"""
    cols = list(df.columns[9:27]); recs = []
    for (c, g), col in zip(CATS if False else _order(), cols):
        t = pd.DataFrame({"college_code": df[code].values, "branch_code": df[branch].values, "caste": c, "gender": g, "region": "OU",
                          "rank": pd.to_numeric(df[col], errors="coerce").values, "fee": pd.to_numeric(df[extra["fee"]], errors="coerce").values})
        recs.append(t)
    r = pd.concat(recs); r["strength"] = np.nan; return r.dropna(subset=["rank"])

def _order():  # column order used by the 2021/2022 statements
    return [("OC", "M"), ("OC", "F")] + [(c, g) for c in ["BC_A", "BC_B", "BC_C", "BC_D", "BC_E", "SC", "ST"] for g in "MF"] + [("EWS", "M"), ("EWS", "F")]

def _pick(pattern):
    f = sorted(glob.glob(os.path.join(D, pattern))); return f[0] if f else None

def load_all():
    recs, metas, used = [], [], {}
    # ---- 2022 statement (cleanest names / districts) ----
    f = _pick("*2022*.csv")
    if f:
        b = pd.read_csv(f); b.columns = [re.sub(r"\s+", " ", c).strip() for c in b.columns]
        b = b[b["INST CODE"].notna()].copy()
        for c in ["INST CODE", "BRANCH"]: b[c] = b[c].astype(str).str.strip().str.upper()
        r = _wide(b, "INST CODE", "BRANCH", {"fee": "TUITION FEE"}); r["src"] = "2022"; recs.append(r)
        metas.append(pd.DataFrame({"college_code": b["INST CODE"], "branch_code": b["BRANCH"], "college": b["INSTITUTE NAME"].map(smart_title),
            "branch": b["BRANCH NAME"].map(smart_title), "place": b["PLACE"].map(smart_title), "dist": b["DIST"].astype(str).str.strip().str.upper(),
            "type": b["TYPE"], "estb": pd.to_numeric(b["YEAR OF ESTB"], errors="coerce"), "affil": b["AFFILIATED"].map(lambda s: re.sub(r"\s+", " ", str(s)).strip()),
            "fee": pd.to_numeric(b["TUITION FEE"], errors="coerce"), "strength": np.nan, "pri": 1}))
        used["2022 statement"] = len(r)
    # ---- main long-format dataset ----
    f = _pick("tseamcet.csv")
    if f:
        a = pd.read_csv(f); a.columns = [c.strip() for c in a.columns]
        for c in ["rank", "fee", "students_per_class"]: a[c] = pd.to_numeric(a[c], errors="coerce")
        for c in ["caste", "gender", "region", "branch_code", "college_code"]: a[c] = a[c].astype(str).str.strip().str.upper()
        a = a.dropna(subset=["rank"]); a = a[a["rank"] > 0]
        full = a["college"].astype(str).str.replace(r"^\s*\w+\s*-\s*", "", regex=True).str.strip()
        nm = full.apply(lambda s: s.rsplit(",", 1)[0] if "," in s else s); pl = full.apply(lambda s: s.rsplit(",", 1)[1] if "," in s else "")
        a["college"] = nm.map(smart_title); a["place"] = pl.map(smart_title)
        r = a[["college_code", "branch_code", "caste", "gender", "region", "rank", "fee"]].copy(); r["strength"] = a["students_per_class"]; r["src"] = "main"; recs.append(r)
        metas.append(a.groupby(["college_code", "branch_code"]).agg(college=("college", "first"), branch=("branch", "first"), place=("place", "first"),
            fee=("fee", "median"), strength=("students_per_class", "median")).reset_index().assign(pri=2, dist="", type=None, estb=np.nan, affil=None))
        metas[-1]["branch"] = metas[-1]["branch"].map(smart_title); used["main dataset"] = len(r)
    # ---- 2021 first-phase statement (PDF-converted; names are garbled so only codes/ranks/places are used) ----
    f = _pick("*2021*.xlsx")
    if f:
        x = pd.read_excel(f, header=None); x = x[x[0].astype(str).str.fullmatch(r"[A-Z0-9]{3,6}") & x[7].astype(str).str.fullmatch(r"[A-Z0-9]{2,5}")].copy()
        x[0] = x[0].str.upper(); x[7] = x[7].str.upper()
        x = x.rename(columns={0: "c", 7: "b"}); x.columns = [str(c) for c in x.columns]
        r = _wide(x, "c", "b", {"fee": "27"}); r["src"] = "2021"; recs.append(r)
        metas.append(pd.DataFrame({"college_code": x["c"], "branch_code": x["b"], "college": None, "branch": None, "place": x["2"].map(smart_title),
            "dist": x["3"].astype(str).str.strip().str.upper(), "type": x["5"], "estb": pd.to_numeric(x["6"], errors="coerce"), "affil": None,
            "fee": pd.to_numeric(x["27"], errors="coerce"), "strength": np.nan, "pri": 3}))
        used["2021 statement"] = len(r)
    if not recs: raise SystemExit("No dataset found in datasets/ - copy tseamcet.csv (and the 2021/2022 files) there.")
    rec = pd.concat(recs, ignore_index=True)
    for c in ["caste", "gender", "region", "branch_code", "college_code"]: rec[c] = rec[c].astype(str).str.strip().str.upper()
    rec = rec[rec["rank"] > 0]
    K = ["college_code", "branch_code"]
    meta = pd.concat(metas, ignore_index=True).sort_values("pri").drop(columns="pri")
    cls = meta.groupby(K).first().reset_index()              # first non-null value per column, in priority order
    cn = meta.dropna(subset=["college"]).drop_duplicates("college_code").set_index("college_code")["college"]
    dd = meta[meta["dist"].astype(str).str.len() > 0].drop_duplicates("college_code").set_index("college_code")["dist"]
    cls["college"] = cls["college_code"].map(cn).fillna(cls["college_code"])
    cls["dist"] = cls["college_code"].map(dd).fillna("")
    bn = meta.dropna(subset=["branch"]).drop_duplicates("branch_code").set_index("branch_code")["branch"]
    cls["branch"] = cls["branch"].fillna(cls["branch_code"].map(bn)).fillna(cls["branch_code"])
    fee = rec.groupby(K)["fee"].median(); cls["fee"] = cls["fee"].fillna(cls.set_index(K).index.map(fee).to_series(index=cls.index))
    cls["fee"] = cls["fee"].fillna(rec["fee"].median()); cls["place"] = cls["place"].fillna("")
    cls = cls[cls.set_index(K).index.isin(rec.set_index(K).index)]
    return rec, cls, used
