"""JEE (JoSAA) Random-Forest predictor. Trained by train_jee.py on datasets/JEE_Rank_2016_2024.csv.
Falls back to the approximate table in jee_data.py for institute types that have no training data (e.g. NITs / IIITs)."""
import json, os, re, numpy as np, pandas as pd, joblib
import jee_data

FAMS = [("CSE", "Computer Science & AI", ["computer", "data science", "artificial intelligence", "information technology"]),
        ("ECE", "Electronics & Communication", ["electronics", "communication", "vlsi"]), ("EEE", "Electrical", ["electrical", "power"]),
        ("MECH", "Mechanical & Production", ["mechanical", "production", "manufacturing", "industrial"]), ("CIVIL", "Civil & Structural", ["civil", "structural", "infrastructure"]),
        ("CHEM", "Chemical", ["chemical"]), ("AERO", "Aerospace / Naval / Ocean", ["aerospace", "aeronautical", "naval", "ocean"]),
        ("MATH", "Mathematics & Computing", ["mathematics", "statistics", "computational"]), ("MAT", "Materials & Metallurgy", ["metallurgical", "materials", "ceramic", "polymer"]),
        ("ENGPHY", "Engineering Physics", ["physics", "engineering science"]), ("BIO", "Bio & Biotechnology", ["bio"])]
SEAT = {"OPEN": "OPEN", "EWS": "EWS", "OBC": "OBC-NCL", "SC": "SC", "ST": "ST"}
norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())

def family(prog):
    p = re.sub(r"\s*\(.*$", "", str(prog)).lower()
    for code, _, kws in FAMS:
        if any(k in p for k in kws): return code
    return "OTHER"
def canon(name):
    name = str(name).strip()
    return "Indian Institute of Technology (ISM) Dhanbad" if name == "Indian School of Mines Dhanbad" else name
def itype(name):
    n = name.lower()
    return "IIIT" if "information technology" in n else "NIT" if "national institute of technology" in n else "IIT" if "indian institute of technology" in n else "OTHER"
def short_inst(name): return name.replace("Indian Institute of Technology", "IIT").replace("National Institute of Technology", "NIT")
def short_prog(p):
    p = p.replace("Bachelor and Master of Technology (Dual Degree)", "B.Tech + M.Tech Dual Degree").replace("Integrated Master of Technology", "Integrated M.Tech")
    return re.sub(r"\((\d) Years, (Bachelor of Technology)\)", r"(\1 yr B.Tech)", p).replace("Bachelor of Technology", "B.Tech").replace(" Years", " yr")
def features(s, ref, seat, pool, fam, inst, dur, enc):
    idx = lambda v, l: np.array([l.index(x) if x in l else -1 for x in v])
    s = np.asarray(s, float); ref = np.asarray(ref, float)
    return np.column_stack([np.log(s / ref), np.log(s), idx(seat, enc["seat"]), idx(pool, enc["pool"]), idx(fam, enc["fam"]), idx(inst, enc["inst"]), np.asarray(dur, float)])

M = None
def load():
    global M
    if not os.path.exists("models/jee_rf.joblib"): return False
    G = pd.read_csv("models/jee_groups.csv"); meta = json.load(open("models/jee_meta.json"))
    G["dem"] = 1 - G.groupby(["seat", "pool"])["ref"].rank(pct=True)
    info = {norm(r[1]): r for r in jee_data.R}
    M = {"rf": joblib.load("models/jee_rf.joblib"), "G": G, "meta": meta, "info": info}; return True
def available(): return M["meta"] if M else None

def predict(d, search=False, q=""):
    exam = d.get("exam", "main"); want = ("IIT",) if exam == "adv" else ("NIT", "IIIT")
    cat = d.get("category", "OPEN"); pct = float(d.get("pct") or 100)
    warn = "" if pct >= (65 if cat in ("SC", "ST") else 75) else "Below the 75% (65% SC/ST) Class 12 criterion: eligible only if you are in your board's top-20 percentile."
    if not M or not any(t in M["meta"]["types"] for t in want):
        res, w = jee_data.predict(d, search, q)
        return res, (w + " " if w else "") + "Note: NIT / IIIT predictions use approximate cutoff data (no training file for them yet)."
    try: rank = float(d.get("rank") or 0)
    except ValueError: rank = 0
    try: crank = float(d.get("crank") or 0)
    except ValueError: crank = 0
    note = ""; g = M["G"][M["G"].type.isin(want)].copy()
    if q:
        toks = q.lower().split(); txt = (g.inst + " " + g.inst.map(short_inst) + " " + g.prog).str.lower(); g = g[txt.apply(lambda s: all(t in s for t in toks))]
    if not search:
        if d.get("branch"): g = g[g.fam == d["branch"]]
        try: maxf = float(d.get("max_fee") or 99)
        except ValueError: maxf = 99
        g = g[g.inst.map(lambda n: (M["info"].get(norm(short_inst(n))) or [0] * 5)[4] <= maxf)]
    if g.empty: return [], warn
    if rank:
        seats = {"OPEN", SEAT.get(cat, "OPEN")}; g = g[g.seat.isin(seats)]
        gender = d.get("gender", "M"); g = g[(g.pool == "N") | (gender in ("F", "N"))]
        g["srank"] = np.where(g.seat == "OPEN", rank, crank)
        if cat != "OPEN" and not crank: note = "Category rank not entered, so only OPEN seats (using your AIR) were considered. Enter your category rank to include reserved seats."
        g = g[g.srank > 0]
        if g.empty: return [], (warn + " " + note).strip()
        X = features(g.srank, g.ref, g.seat, g.pool, g.fam, g.inst, g.dur, M["meta"]["enc"]); g["p"] = M["rf"].predict_proba(X)[:, 1]
        g = g.sort_values("p", ascending=False).drop_duplicates(["inst", "prog"])
        if not search:
            keep = g[g.p >= .05]
            if len(keep) >= 15: g = keep
            else: g = g.head(25); note = (note + " Your rank is beyond the closing ranks of most programs for these filters, so these are the closest options (low chances).").strip()
        w0, wd, wp, wf = {"b": (.2, .6, .3, .1), "p": (.2, .3, .6, .1), "f": (.2, .3, .1, .6), "a": (.7, .5, .3, .2)}.get(d.get("priority", "b"), (.2, .6, .3, .1))
        g["plc"] = g.inst.map(lambda n: (M["info"].get(norm(short_inst(n))) or [0] * 8)[5] or 70); g["feesc"] = g.inst.map(lambda n: 1 - min((M["info"].get(norm(short_inst(n))) or [0] * 5)[4] or 2.3, 4) / 4)
        pe = np.minimum(g.p / .6, 1) ** 1.5      # chance counts fully once it is "likely" (>=60%); after that, college quality decides
        g["sc"] = 100 * pe * (w0 + (1 - w0) * (wd * g.dem + wp * g.plc / 100 + wf * g.feesc) / (wd + wp + wf)); g = g.sort_values("sc", ascending=False)
    else:
        g = g[(g.seat == "OPEN") & (g.pool == "N")].drop_duplicates(["inst", "prog"]).sort_values(["inst", "prog"])
    out = []
    for r in g.head(60 if search else 40).itertuples():
        inf = M["info"].get(norm(short_inst(r.inst))); p = getattr(r, "p", None)
        pool = "Female-only (incl. supernumerary)" if r.pool == "F" else "Gender-neutral"
        det = [["Program", short_prog(r.prog)], ["Seat pool used", f"{r.seat} · {pool}"], ["Latest closing rank", f"{int(r.last):,} ({int(r.lastyear)}) · opening {int(r.lastopen):,}" if r.lastopen == r.lastopen else f"{int(r.last):,} ({int(r.lastyear)})"],
               ["Closing rank range", f"{int(r.lo):,} – {int(r.hi):,} over {int(r.n)} records"], ["Recent trend", r.trend]]
        if inf: det += [["State", inf[2]], ["Annual fee (approx.)", f"₹{inf[4]} lakh"], ["Placement security (approx.)", f"~{inf[5]}% placed"], ["Average package (approx.)", f"₹{inf[6]} LPA"]]
        out.append({"id": inf[0] if inf else norm(short_inst(r.inst)), "name": inf[1] if inf else short_inst(r.inst), "location": inf[2] if inf else "India", "branch": short_prog(r.prog),
                    "fee": f"₹{inf[4]} lakh / yr" if inf else "—", "dem": round(float(r.dem), 3), "prob": None if p is None else float(p), "score": float(r.sc) if p is not None else None,
                    "level": None if p is None else "High" if p >= .7 else "Moderate" if p >= .35 else "Low", "plc": inf[5] if inf else None, "pkg": inf[6] if inf else None, "details": det})
    return out, (warn + " " + note).strip()
