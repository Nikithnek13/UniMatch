# UniMatch
1. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
2. `python train.py`   (trains on everything in datasets/: tseamcet.csv, 2022 + 2021 statements, marks_vs_rank.xlsx)
3. `python app.py`  ->  open http://127.0.0.1:5001

Optional: fill `static/college_info.json` ({"CBIT": {"site": "https://...", "plc": 90, "pkg": 9}}) and put photos in `static/img/<college code>.jpg`.

## College photos
Put images in `static/img/` named by college code (the code shown on each card, e.g. `ANRH`):
`ANRH.jpg` is the thumbnail + first slide; extra slides: `ANRH_2.jpg`, `ANRH_3.png` ... (jpg/jpeg/png/webp/gif). No restart needed - just run the prediction again.
JEE institutes use their ids: `IITB`, `NITW`, `IIITA` ... (see jee_data.py).
