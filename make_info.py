"""Reads your college workbook (Website / optional Placement % / Avg package columns) and writes static/college_info.json.
Looks for college_list.xlsx or static/img/college_new.xlsx.     Usage: python make_info.py"""
import json, os, pandas as pd
OUT = "static/college_info.json"; info = json.load(open(OUT)) if os.path.exists(OUT) else {}
src = next((p for p in ["college_list.xlsx", "static/img/college_new.xlsx", "college_new.xlsx"] if os.path.exists(p)), None)
if not src: raise SystemExit("Workbook not found (college_list.xlsx / static/img/college_new.xlsx)")
sheets = pd.read_excel(src, sheet_name=None, header=0)
for name, key in [("EAMCET Colleges", "Code"), ("JEE Institutes", "ID")]:
    for r in sheets[name].to_dict("records"):
        e = info.setdefault(str(r[key]).strip().upper(), {})
        for col, k in [("Website", "site"), ("Placement % (optional)", "plc"), ("Avg package LPA (optional)", "pkg")]:
            v = r.get(col)
            if pd.notna(v) and str(v).strip(): e[k] = str(v).strip() if k == "site" else float(v)
info = {k: v for k, v in info.items() if v}
json.dump(info, open(OUT, "w"), indent=1); print("Wrote", OUT, "with", len(info), "colleges from", src)
