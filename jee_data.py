"""JEE Main (NIT/IIIT, CRL) and JEE Advanced (IIT, AIR) predictor.
Closing ranks below are APPROXIMATE (open category). Replace with official JoSAA data when you have it."""
import math
# id, name, state, type, fee(lakh/yr), placement %, avg package LPA, closing ranks [CSE, ECE, EEE, MECH, CIVIL] (0 = not offered)
R = [("iitb","IIT Bombay","Maharashtra","IIT",2.3,95,23,[70,600,900,1500,3500]),("iitd","IIT Delhi","Delhi","IIT",2.3,94,22,[130,800,1100,2000,3600]),
("iitm","IIT Madras","Tamil Nadu","IIT",2.3,94,21,[190,1100,1600,2800,5500]),("iitk","IIT Kanpur","Uttar Pradesh","IIT",2.3,92,21,[300,1600,2300,4000,6500]),
("iitkgp","IIT Kharagpur","West Bengal","IIT",2.3,90,19,[560,2500,3500,5500,9000]),("iitr","IIT Roorkee","Uttarakhand","IIT",2.3,90,19,[700,2800,3800,5800,9000]),
("iitg","IIT Guwahati","Assam","IIT",2.3,89,18,[900,3400,4600,7000,11000]),("iith","IIT Hyderabad","Telangana","IIT",2.3,90,20,[600,2400,3600,5500,8000]),
("iitbhu","IIT (BHU) Varanasi","Uttar Pradesh","IIT",2.3,88,17,[1500,3500,5000,7500,12000]),("iiti","IIT Indore","Madhya Pradesh","IIT",2.3,88,18,[1800,4000,5500,8500,13000]),
("iitism","IIT (ISM) Dhanbad","Jharkhand","IIT",2.3,86,16,[2200,4800,6500,9500,14000]),("iitrpr","IIT Ropar","Punjab","IIT",2.3,85,15,[2800,5500,7500,11000,16000]),
("iittp","IIT Tirupati","Andhra Pradesh","IIT",2.3,82,14,[4500,7500,9500,13000,18000]),
("nitw","NIT Warangal","Telangana","NIT",1.6,88,12,[2400,5000,9000,14000,24000]),("nitt","NIT Tiruchirappalli","Tamil Nadu","NIT",1.6,90,14,[1500,4000,7500,10500,18000]),
("nitk","NIT Karnataka (Surathkal)","Karnataka","NIT",1.6,88,14,[2300,4500,8000,13000,20000]),("nitc","NIT Calicut","Kerala","NIT",1.6,86,12,[3500,6500,10500,16000,26000]),
("nitr","NIT Rourkela","Odisha","NIT",1.6,84,11,[4500,8500,13000,19000,32000]),("mnnit","MNNIT Allahabad","Uttar Pradesh","NIT",1.6,84,11,[5500,9500,14000,20000,34000]),
("nitkkr","NIT Kurukshetra","Haryana","NIT",1.6,82,10,[8000,13000,19000,27000,45000]),("nitap","NIT Andhra Pradesh","Andhra Pradesh","NIT",1.6,76,8,[14000,22000,32000,50000,75000]),
("nitdgp","NIT Durgapur","West Bengal","NIT",1.6,80,9,[14000,22000,30000,42000,65000]),
("iiita","IIIT Allahabad","Uttar Pradesh","IIIT",1.9,88,15,[3800,6500,0,0,0]),("iiitm","IIITM Gwalior","Madhya Pradesh","IIIT",1.8,84,12,[5500,9000,0,0,0]),
("iiitsc","IIIT Sri City","Andhra Pradesh","IIIT",2.8,80,9,[15000,25000,0,0,0]),("iiitdm","IIITDM Kancheepuram","Tamil Nadu","IIIT",1.7,80,9,[9000,14000,0,30000,0]),
("iiitdmj","IIITDM Jabalpur","Madhya Pradesh","IIIT",1.7,78,8,[9000,14000,0,40000,0]),("iiitv","IIIT Vadodara","Gujarat","IIIT",1.8,76,8,[12000,20000,0,0,0])]
# ---- remaining IITs, NITs and IIITs (closing ranks are APPROXIMATE estimates built from a CSE base rank and typical branch ratios) ----
def _g(i, n, st, t, fee, plc, pkg, base, ratios, drop=()):
    cuts = [int(base)] + [int(base * r) for r in ratios]
    for k in drop: cuts[k] = 0
    R.append((i, n, st, t, fee, plc, pkg, cuts))
for i, n, st, base, plc, pkg, drop in [("iitbbs", "IIT Bhubaneswar", "Odisha", 3900, 82, 14, ()), ("iitgn", "IIT Gandhinagar", "Gujarat", 2200, 85, 16, (1,)),
        ("iitj", "IIT Jodhpur", "Rajasthan", 2600, 82, 14, ()), ("iitp", "IIT Patna", "Bihar", 4700, 80, 13, ()), ("iitmandi", "IIT Mandi", "Himachal Pradesh", 4500, 80, 13, ()),
        ("iitpkd", "IIT Palakkad", "Kerala", 6500, 75, 11, (1,)), ("iitjammu", "IIT Jammu", "Jammu & Kashmir", 6900, 72, 10, (1,)), ("iitdh", "IIT Dharwad", "Karnataka", 6500, 74, 11, (1,)),
        ("iitbhilai", "IIT Bhilai", "Chhattisgarh", 8500, 70, 10, (1, 4)), ("iitgoa", "IIT Goa", "Goa", 6500, 70, 10, (1, 4))]:
    _g(i, n, st, "IIT", 2.3, plc, pkg, base, (1.9, 2.5, 3.8, 5.5), drop)
for i, n, st, base in [("nitagartala", "NIT Agartala", "Tripura", 23000), ("nitarunachal", "NIT Arunachal Pradesh", "Arunachal Pradesh", 40000), ("manit", "MANIT Bhopal", "Madhya Pradesh", 8500),
        ("nitdelhi", "NIT Delhi", "Delhi", 12000), ("nitgoa", "NIT Goa", "Goa", 18000), ("nithamirpur", "NIT Hamirpur", "Himachal Pradesh", 12500), ("mnit", "MNIT Jaipur", "Rajasthan", 3800),
        ("nitjal", "NIT Jalandhar", "Punjab", 9500), ("nitjsr", "NIT Jamshedpur", "Jharkhand", 15000), ("nitmanipur", "NIT Manipur", "Manipur", 38000), ("nitmeghalaya", "NIT Meghalaya", "Meghalaya", 36000),
        ("nitmizoram", "NIT Mizoram", "Mizoram", 50000), ("nitnagaland", "NIT Nagaland", "Nagaland", 48000), ("vnit", "VNIT Nagpur", "Maharashtra", 5200), ("nitpatna", "NIT Patna", "Bihar", 12000),
        ("nitpy", "NIT Puducherry", "Puducherry", 24000), ("nitraipur", "NIT Raipur", "Chhattisgarh", 9500), ("nitsikkim", "NIT Sikkim", "Sikkim", 34000), ("nitsilchar", "NIT Silchar", "Assam", 14000),
        ("nitsrinagar", "NIT Srinagar", "Jammu & Kashmir", 19000), ("svnit", "SVNIT Surat", "Gujarat", 5800), ("nituk", "NIT Uttarakhand", "Uttarakhand", 28000)]:
    _g(i, n, st, "NIT", 1.6, 82 if base <= 9000 else 76 if base <= 20000 else 68, 11 if base <= 9000 else 8.5 if base <= 20000 else 7, base, (2.1, 3.8, 6, 10))
for i, n, st, base in [("iiitg", "IIIT Guwahati", "Assam", 12000), ("iiitkota", "IIIT Kota", "Rajasthan", 13000), ("iiitn", "IIIT Nagpur", "Maharashtra", 17000), ("iiitpune", "IIIT Pune", "Maharashtra", 15000),
        ("iiitu", "IIIT Una", "Himachal Pradesh", 22000), ("iiitsonepat", "IIIT Sonepat", "Haryana", 16000), ("iiitranchi", "IIIT Ranchi", "Jharkhand", 18000), ("iiitl", "IIIT Lucknow", "Uttar Pradesh", 14000),
        ("iiitdwd", "IIIT Dharwad", "Karnataka", 20000), ("iiitkalyani", "IIIT Kalyani", "West Bengal", 21000), ("iiitkottayam", "IIIT Kottayam", "Kerala", 17000), ("iiitmanipur", "IIIT Manipur", "Manipur", 45000),
        ("iiittrichy", "IIIT Tiruchirappalli", "Tamil Nadu", 24000), ("iiitbh", "IIIT Bhagalpur", "Bihar", 30000), ("iiitbhopal", "IIIT Bhopal", "Madhya Pradesh", 18000), ("iiitsurat", "IIIT Surat", "Gujarat", 20000),
        ("iiitagartala", "IIIT Agartala", "Tripura", 38000), ("iiitraichur", "IIIT Raichur", "Karnataka", 30000)]:
    _g(i, n, st, "IIIT", 2.0, 62, 6.5, base, (1.7, 0, 0, 0))
_g("iiitkurnool", "IIITDM Kurnool", "Andhra Pradesh", "IIIT", 2.0, 62, 6.5, 21000, (1.7, 0, 4, 0))

BR = ["CSE", "ECE", "EEE", "MECH", "CIVIL"]; NAMES = dict(zip(BR, ["Computer Science", "Electronics & Communication", "Electrical", "Mechanical", "Civil"]))
CAT = {"OPEN": 1, "EWS": 1.2, "OBC": 1.8, "SC": 4, "ST": 6}
STATES = sorted({r[2] for r in R})

def predict(d, search=False, q=""):
    """Prediction (search=False) or college look-up by name (search=True, rank optional)."""
    exam = d.get("exam", "main"); cat = d.get("category", "OPEN"); fem = d.get("gender") in ("F", "N"); pct = float(d.get("pct") or 100)
    try: rank = float(d.get("rank") or 0)
    except ValueError: rank = 0
    types = ("IIT",) if exam == "adv" else ("NIT", "IIIT"); w = {"b": (.4, .3, .2, .1), "p": (.3, .45, .15, .1), "f": (.3, .2, .05, .45), "a": (.7, .15, .1, .05)}[d.get("priority", "b")]
    maxf = float(d.get("max_fee") or 99); toks = q.upper().split(); out = []
    for i, n, st, t, fee, plc, pkg, cuts in R:
        if t not in types or (toks and not all(k in n.upper() + " " + i.upper() for k in toks)): continue
        if not search and fee > maxf: continue
        for b, c in zip(BR, cuts):
            if not c or (not search and d.get("branch") in BR and d["branch"] != b): continue
            p = sc = None
            if rank:
                e = c * CAT[cat] * (1.3 if fem and t != "IIIT" else 1) * (2.3 if t == "NIT" and d.get("state") == st else 1)
                p = 1 / (1 + math.exp(-math.log(e / rank) / 0.28))
                if p < .05 and not search: continue
                sc = 100 * (w[0] * p + w[1] * plc / 100 + w[2] * min(pkg / 20, 1) + w[3] * (1 - min(fee / 4, 1)))
            out.append({"id": i, "name": n, "location": st, "branch": NAMES[b], "fee": f"₹{fee} lakh / yr", "prob": p, "score": sc, "plc": plc, "pkg": pkg,
                        "level": None if p is None else "High" if p >= .7 else "Moderate" if p >= .35 else "Low",
                        "details": [["Institute type", t], ["Branch", NAMES[b]], ["State", st], ["Annual fee", f"₹{fee} lakh (approx.)"],
                                    ["Approx. closing rank (open)", f"{c:,}"], ["Placement security", f"~{plc}% placed"], ["Average package", f"₹{pkg} LPA (approx.)"]]})
    out.sort(key=lambda r: (-(r["score"] if r["score"] is not None else 0), r["name"], r["branch"]))
    warn = "" if pct >= (65 if cat in ("SC", "ST") else 75) else "Below the 75% (65% SC/ST) Class 12 criterion: eligible only if you are in your board's top-20 percentile."
    return out[:60 if search else 40], warn
