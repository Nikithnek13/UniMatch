import json, os, re, sqlite3, numpy as np, pandas as pd, joblib
from urllib.parse import quote
from flask import Flask, jsonify, request, session, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from common import features
import jee_data, jee_model

app = Flask(__name__, static_folder="static"); app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret")   # set SECRET_KEY in your hosting dashboard
BASE = os.path.dirname(os.path.abspath(__file__)); os.chdir(BASE)
if not os.path.exists("models/rf.joblib"): raise SystemExit("Model not found. Run:  python train.py")
RF = joblib.load("models/rf.joblib"); META = json.load(open("models/meta.json")); ENC = META["enc"]
MARKS = joblib.load("models/marks_rf.joblib") if os.path.exists("models/marks_rf.joblib") else None
GROUPS = pd.read_csv("models/groups.csv"); CLASSES = pd.read_csv("models/classes.csv")
CLASSES["dist"] = CLASSES["dist"].fillna(""); CLASSES["place"] = CLASSES["place"].fillna("")
INFO = {k.upper(): v for k, v in (json.load(open("static/college_info.json")) if os.path.exists("static/college_info.json") else {}).items()}
JEE_OK = jee_model.load()
K = ["college_code", "branch_code"]; SMED = float(CLASSES["strength"].median())
CLASSES = CLASSES.merge(GROUPS.groupby(K)["med"].median().rank(pct=True).rename("dem").reset_index(), on=K)
CLASSES["dem"] = 1 - CLASSES["dem"]; CLASSES["feesc"] = 1 - CLASSES["fee"].rank(pct=True)
HYD_DIST = {"HYD", "MDL", "RR"}      # "Hyderabad region" = these districts (+ keyword match for colleges without district data)
HYD_WORDS = ["HYDERABAD", "SECUNDERABAD", "GHATKESAR", "MEDCHAL", "KUKATPALLY", "GANDIPET", "BACHUPALLY", "BACHUPALLI", "SHAMSHABAD", "IBRAHIMPATNAM", "DUNDIGAL",
             "MAISAMMAGUDA", "KANDLAKOYA", "SHAMIRPET", "KEESARA", "NIZAMPET", "MOINABAD", "CHEVELLA", "SHANKARPALLY", "HAYATHNAGAR", "UPPAL", "PATANCHERU"]
DISTN = {d["code"]: d["name"] for d in META["districts"]}

# ---- user database ----
# Local: SQLite file next to app.py.  Vercel / Render etc.: set DATABASE_URL (Postgres, e.g. Neon or Supabase) so accounts persist.
# Without DATABASE_URL on a read-only server (Vercel) it falls back to /tmp, which works but is wiped on every cold start.
PG_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or ""
SQLITE_PATH = os.environ.get("DB_PATH") or (os.path.join(BASE, "users.db") if os.access(BASE, os.W_OK) else "/tmp/users.db")
def _pg():
    import pg8000.native
    from urllib.parse import urlparse, unquote
    u = urlparse(PG_URL)
    return pg8000.native.Connection(user=unquote(u.username or ""), password=unquote(u.password or ""), host=u.hostname, port=u.port or 5432, database=(u.path or "/").lstrip("/"), ssl_context=True)
def get_user(email):
    if PG_URL:
        c = _pg()
        try:
            c.run("create table if not exists users(email text primary key, name text, pw text, done text)")
            r = c.run("select email, name, pw, done from users where email = :e", e=email)
        finally: c.close()
        return dict(zip(("email", "name", "pw", "done"), r[0])) if r else None
    with sqlite3.connect(SQLITE_PATH) as c:
        c.execute("create table if not exists users(email text primary key, name text, pw text, done text)")
        r = c.execute("select email, name, pw, done from users where email=?", (email,)).fetchone()
    return dict(zip(("email", "name", "pw", "done"), r)) if r else None
def add_user(email, name, pw, done):
    if PG_URL:
        c = _pg()
        try: c.run("insert into users(email, name, pw, done) values (:e, :n, :p, :d)", e=email, n=name, p=pw, d=done)
        finally: c.close()
        return
    with sqlite3.connect(SQLITE_PATH) as c: c.execute("insert into users values(?,?,?,?)", (email, name, pw, done))
def need_login():
    if "u" not in session: return jsonify(error="Please log in"), 401

@app.get("/")
def home(): return send_from_directory("static", "index.html")
@app.get("/api/meta")
def meta(): return jsonify(**{k: META[k] for k in ("districts", "places", "branches", "metrics")}, castes=ENC["caste"], genders=ENC["gender"], regions=ENC["region"],
                           states=jee_data.STATES, marks=MARKS is not None, jee_branches=(jee_model.available() or {}).get("families") or [[b, jee_data.NAMES[b]] for b in jee_data.BR], jee_metrics=(jee_model.available() or {}).get("metrics"))
@app.get("/api/me")
def me(): return jsonify(user=session.get("u"))
@app.post("/api/register")
def register():
    d = request.json; e = d["email"].strip().lower(); name = (d.get("name") or "").strip() or e.split("@")[0]; done = d.get("done", "y")
    if "@" not in e or len(d["password"]) < 6: return jsonify(error="Enter a valid email and a password of 6+ characters"), 400
    if get_user(e): return jsonify(error="Account already exists - please log in"), 400
    add_user(e, name, generate_password_hash(d["password"], method="pbkdf2:sha256"), done)
    session["u"] = {"email": e, "name": name, "done": done}; return jsonify(user=session["u"])
@app.post("/api/login")
def login():
    d = request.json; r = get_user(d["email"].strip().lower())
    if not r or not check_password_hash(r["pw"], d["password"]): return jsonify(error="Invalid email or password"), 401
    session["u"] = {"email": r["email"], "name": r["name"], "done": r["done"]}; return jsonify(user=session["u"])
@app.get("/api/health")
def health():
    """Open /api/health in the browser to check the deployment."""
    out = {"models": True, "jee_model": bool(JEE_OK), "db": "postgres" if PG_URL else "sqlite:" + SQLITE_PATH}
    try: get_user("health@check.local"); out["db_ok"] = True
    except Exception as e: out["db_ok"] = False; out["db_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return jsonify(out)
@app.errorhandler(Exception)
def on_error(e):
    from werkzeug.exceptions import HTTPException
    if request.path.startswith("/api/"):
        if isinstance(e, HTTPException): return jsonify(error=e.description), e.code
        app.logger.exception(e); return jsonify(error=f"Server error: {type(e).__name__}: {str(e)[:200]}"), 500
    return e if isinstance(e, HTTPException) else ("Server error", 500)
@app.post("/api/logout")
def logout(): session.clear(); return jsonify(ok=1)

@app.post("/api/marks2rank")
def marks2rank():
    if (x := need_login()): return x
    if MARKS is None: return jsonify(error="marks_vs_rank.xlsx was not found when training"), 400
    m = float(request.json["marks"]); lo, hi = META["metrics"]["marks_range"]; mm = min(max(m, lo), hi)
    return jsonify(rank=int(round(float(np.exp(MARKS.predict([[mm]])[0])), -1)), clamped=(m != mm), range=[lo, hi])

IMG_DIR = os.path.join(BASE, "static", "img")
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif")
IMG_ALIAS = {"IITSM": "IITISM"}      # file name -> institute id (IIT (ISM) Dhanbad)
def img_map(kind):
    """college code -> photo URLs. Looks in static/img/<kind>/ (EAMCET or JEE), then static/img/.
    File names: CODE.jpg (thumbnail + first slide), CODE_2.png, CODE_3.webp ... (extra slides). Read per request: no restart needed."""
    m = {}
    for sub in (kind, ""):
        try: files = sorted(os.listdir(os.path.join(IMG_DIR, sub)))
        except OSError: continue
        for f in files:
            st, ext = os.path.splitext(f)
            if ext.lower() not in IMG_EXT: continue
            code = re.sub(r"[^A-Z0-9]", "", re.sub(r"_\d+$", "", st).upper()); code = IMG_ALIAS.get(code, code)
            m.setdefault(code, []).append("/static/img/" + (sub + "/" if sub else "") + quote(f))
    return m

def num(v):
    try: return int(v) if v == v else None
    except (TypeError, ValueError): return None

def eamcet_core(d, search=False, q=""):
    """returns (results, note)"""
    caste, gender, region = d["caste"], d["gender"], d["region"]; note = ""
    try: rank = float(d.get("rank") or 0)
    except ValueError: rank = 0
    g = GROUPS.copy() if search else GROUPS[GROUPS.caste.isin([caste, "OC"])].copy()   # students can also compete in open (OC) seats
    if gender == "M": g = g[g.gender == "M"]            # Male: no girls-only seats/colleges. Female / "Prefer not to say" (N): all seats are shown
    g["pri"] = (g.region != region) * 1 + (~g.caste.isin([caste, "OC"])) * 4
    g = g.sort_values(["pri", "med"], ascending=[True, False]).drop_duplicates(K).merge(CLASSES, on=K)
    if q:
        toks = q.upper().split(); txt = (g.college + " " + g.college_code + " " + g.place).str.upper()
        g = g[txt.apply(lambda s: all(t in s for t in toks))]
    if not search:
        if d.get("branch"): g = g[g.branch_code == d["branch"]]
        loc = d.get("location", "")
        if loc == "HYD": g = g[g.dist.isin(HYD_DIST) | g.place.str.upper().apply(lambda s: any(h in s for h in HYD_WORDS))]
        elif loc.startswith("d:"): g = g[g.dist == loc[2:]]
        elif loc.startswith("p:"): g = g[g.place == loc[2:]]
        if d.get("max_fee"): g = g[g.fee <= float(d["max_fee"])]
    if g.empty: return [], ("No colleges match these filters - try another branch, location or fee limit." if not search else "")
    g = g.copy(); g["strength_f"] = g.strength.fillna(SMED)
    if rank:
        X = features(np.full(len(g), rank), g.med, g.caste, g.gender, g.region, g.branch_code, g.fee, g.strength_f, ENC)
        g["p"] = RF.predict_proba(X)[:, 1]
        if not search:
            keep = g[g.p >= .05]
            if len(keep) >= 15: g = keep
            else:       # very high rank / narrow filters: still show the closest options instead of an empty page
                g = g.sort_values("p", ascending=False).head(25)
                note = "Your rank is beyond the last ranks of most colleges for these filters, so these are the closest options (low chances). Try other branches or locations, and watch for later counselling phases and spot admissions."
        w = {"b": (.3, .65, .35), "a": (.8, .6, .4), "f": (.3, .2, .8), "d": (.2, .9, .1)}.get(d.get("priority", "b"), (.3, .65, .35))   # (chance weight, demand, fee)
        g["sc"] = 100 * g.p * (w[0] + (1 - w[0]) * (w[1] * g.dem + w[2] * g.feesc) / (w[1] + w[2])); g = g.sort_values("sc", ascending=False)
    else: g = g.sort_values(["college", "branch"])
    out = []; IM = img_map("EAMCET")
    for r in g.head(60 if search else 40).itertuples():
        inf = INFO.get(r.college_code.upper(), {}); p = getattr(r, "p", None)
        det = [["Branch", r.branch], ["College code", r.college_code], ["Location", ", ".join(x for x in [r.place, DISTN.get(r.dist, "")] if x) or "Telangana"]]
        if isinstance(r.type, str) or isinstance(r.affil, str): det.append(["Type / affiliation", " · ".join(str(x) for x in [r.type, r.affil] if isinstance(x, str))])
        if num(r.estb): det.append(["Established", num(r.estb)])
        det += [["Annual fee", f"₹{int(r.fee):,}"]]
        if num(r.strength): det.append(["Seats in this class", num(r.strength)])
        det += [["Seat group matched", f"{r.caste} · {r.gender} · {r.region}"], ["Historical closing rank", f"{int(r.lo):,} – {int(r.hi):,} (median {int(r.med):,}, {int(r.n)} records)"]]
        if "plc" in inf: det.append(["Placement security", f"~{inf['plc']}% placed"])
        if "pkg" in inf: det.append(["Average package", f"₹{inf['pkg']} LPA"])
        out.append({"id": r.college_code, "name": r.college, "location": r.place or "Telangana", "branch": r.branch, "fee": f"₹{int(r.fee):,} / yr",
                    "dem": round(float(r.dem), 3), "prob": None if p is None else float(p), "score": float(r.sc) if p is not None else None,
                    "level": None if p is None else "High" if p >= .7 else "Moderate" if p >= .35 else "Low",
                    "plc": inf.get("plc"), "pkg": inf.get("pkg"), "site": inf.get("site", ""), "imgs": IM.get(r.college_code.upper(), []), "details": det})
    return out, note

def with_imgs(res):
    IM = img_map("JEE")
    for r in res: r["imgs"] = IM.get(r["id"].upper(), []); r["site"] = INFO.get(r["id"].upper(), {}).get("site", "")
    return res

@app.post("/api/predict/eamcet")
def eamcet():
    if (x := need_login()): return x
    res, note = eamcet_core(request.json); return jsonify(results=res, note=note)
@app.post("/api/search/eamcet")
def search_eamcet():
    if (x := need_login()): return x
    d = request.json; res, note = eamcet_core(d, True, d.get("q", "").strip()); return jsonify(results=res, note=note)
@app.post("/api/predict/jee")
def jee():
    if (x := need_login()): return x
    res, warn = jee_model.predict(request.json); return jsonify(results=with_imgs(res), warning=warn)
@app.post("/api/search/jee")
def search_jee():
    if (x := need_login()): return x
    d = request.json; res, warn = jee_model.predict(d, True, d.get("q", "").strip()); return jsonify(results=with_imgs(res), warning=warn)

if __name__ == "__main__": app.run(debug=True, port=5001)
