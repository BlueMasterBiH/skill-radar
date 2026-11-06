# 🛰️ Skill Radar

Svaki dan pretražuje GitHub za Claude / Agent skillovima i rangira one sa najvećim
rastom zvjezdica u zadnja 24h, podijeljeno na **marketing, biznis i dizajn**.

- Najnoviji izvještaj: [`reports/LATEST.md`](reports/LATEST.md)
- Ručno pokretanje: `python3 skill_radar.py` (treba `gh` CLI prijavljen)
- Automatski: GitHub Action `.github/workflows/daily.yml` (svako jutro)

Kategorije i pretrage se mijenjaju na vrhu `skill_radar.py` (`QUERIES`, `CATEGORIES`).
