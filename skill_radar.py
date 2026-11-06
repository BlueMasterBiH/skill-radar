#!/usr/bin/env python3
"""
Skill Radar — svaki dan pretražuje GitHub za Claude skillovima i pronalazi
one koji su "eksplodirali" u zadnja 24h (najveći rast zvjezdica).

Kako mjeri rast:
  * Svaki run sprema snapshot (data/snapshot-YYYY-MM-DD.json) sa brojem zvjezdica.
  * Sljedeći dan uspoređuje sa prethodnim snapshotom -> stvarni +zvjezdice/24h.
  * Prvi run (nema historije): koristi "brzinu" = zvjezdice / starost repoa
    za repoe kreirane u zadnjih 30 dana.

Kategorije: marketing, biznis, dizajn (+ "ostalo" ako je baš trending).
Potreban je samo `gh` CLI (prijavljen) ili GITHUB_TOKEN env varijabla.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

NOW = dt.datetime.now(dt.timezone.utc)
TODAY = NOW.date()
RECENT = (TODAY - dt.timedelta(days=30)).isoformat()

# --- Gdje tražimo kandidate -------------------------------------------------
BASE = f"pushed:>={RECENT} stars:>=5"
QUERIES = [
    f"topic:claude-skills {BASE}",
    f"topic:agent-skills {BASE}",
    f"topic:claude-code-skills {BASE}",
    f"topic:claude-skill {BASE}",
    f"topic:anthropic-skills {BASE}",
    f'"claude skills" in:name,description {BASE}',
    f'"agent skills" in:name,description {BASE}',
    f"skills claude in:name {BASE}",
    f'"SKILL.md" in:readme {BASE}',
    # ciljane kategorije
    f"claude skill marketing in:name,description,readme {BASE}",
    f"claude skill seo in:name,description,readme {BASE}",
    f"claude skill copywriting in:name,description,readme {BASE}",
    f"claude skill design in:name,description,readme {BASE}",
    f"claude skill ui ux in:name,description,readme {BASE}",
    f"claude skill business in:name,description,readme {BASE}",
    f"claude skill sales in:name,description,readme {BASE}",
    f"claude skill brand in:name,description,readme {BASE}",
]

CATEGORIES = {
    "marketing": ["marketing", "seo", "copywriting", "ads", "advert", "social media",
                  "content", "newsletter", "email", "growth", "campaign", "tiktok", "instagram",
                  "youtube", "brand voice", "geo", "aeo", "landing page", "cro", "funnel"],
    "biznis":    ["business", "sales", "startup", "founder", "finance", "pitch", "crm", "strategy",
                  "product manager", "pm ", "investor", "consult", "legal", "hr", "ecommerce",
                  "e-commerce", "saas", "revenue", "pricing", "okr", "operations", "analytics"],
    "dizajn":    ["design", "ui", "ux", "figma", "frontend", "tailwind", "animation", "motion",
                  "typography", "brand", "visual", "css", "interface", "component", "shadcn",
                  "illustration", "logo", "slides", "presentation"],
}
SKILL_HINT = re.compile(r"skill", re.I)


def token():
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    try:
        return subprocess.check_output(["gh", "auth", "token"], text=True,
                                       stderr=subprocess.DEVNULL).strip() or None
    except Exception:
        return None


def search(q, pages=2):
    """GitHub search preko gh CLI (radi i u Actions i lokalno)."""
    out = []
    for page in range(1, pages + 1):
        cmd = ["gh", "api", "-X", "GET", "search/repositories",
               "-f", f"q={q}", "-f", "sort=stars", "-f", "order=desc",
               "-f", "per_page=100", "-f", f"page={page}"]
        for attempt in range(3):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                break
            if "rate limit" in (r.stdout + r.stderr).lower():
                time.sleep(30)
                continue
            print(f"  ! greška za '{q}': {r.stderr.strip()[:150]}", file=sys.stderr)
            return out
        else:
            return out
        items = json.loads(r.stdout).get("items", [])
        out += items
        time.sleep(2.2)  # search limit = 30 req/min
        if len(items) < 100:
            break
    return out


def classify(repo):
    text = " ".join([repo["name"], repo.get("description") or "",
                     " ".join(repo.get("topics") or [])]).lower()
    cats = [c for c, kws in CATEGORIES.items()
            if any(re.search(r"\b" + re.escape(k.strip()) + r"s?\b", text) for k in kws)]
    return cats


def load_prev():
    """Najnoviji snapshot stariji od ~20h."""
    snaps = sorted(DATA.glob("snapshot-*.json"))
    for p in reversed(snaps):
        d = json.loads(p.read_text())
        taken = dt.datetime.fromisoformat(d["taken_at"])
        if (NOW - taken).total_seconds() > 20 * 3600:
            return d, (NOW - taken).total_seconds() / 3600
    return None, None


def main():
    DATA.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    repos = {}
    for q in QUERIES:
        print(f"> {q}")
        for it in search(q):
            text = (it["full_name"] + " " + (it.get("description") or "") +
                    " " + " ".join(it.get("topics") or []))
            if not SKILL_HINT.search(text):
                continue
            repos[it["full_name"]] = {
                "name": it["full_name"],
                "url": it["html_url"],
                "description": (it.get("description") or "").strip(),
                "stars": it["stargazers_count"],
                "forks": it["forks_count"],
                "created_at": it["created_at"],
                "pushed_at": it["pushed_at"],
                "topics": it.get("topics") or [],
            }
    print(f"\nPronađeno {len(repos)} skill repoa.")

    prev, hours = load_prev()
    prev_stars = {r["name"]: r["stars"] for r in prev["repos"]} if prev else {}

    for r in repos.values():
        r["categories"] = classify(r)
        age_days = max((NOW - dt.datetime.fromisoformat(r["created_at"].replace("Z", "+00:00"))
                        ).total_seconds() / 86400, 0.5)
        r["age_days"] = round(age_days, 1)
        if r["name"] in prev_stars:
            r["delta_24h"] = round((r["stars"] - prev_stars[r["name"]]) * 24 / hours)
            r["mode"] = "izmjereno"
        elif prev and age_days <= 2:
            r["delta_24h"] = r["stars"]          # potpuno nov repo
            r["mode"] = "nov"
        elif age_days <= 30:
            r["delta_24h"] = round(r["stars"] / age_days)  # procjena
            r["mode"] = "procjena"
        else:
            r["delta_24h"] = None
            r["mode"] = "n/a"
        base = max(r["stars"] - (r["delta_24h"] or 0), 1)
        r["growth_pct"] = round(100 * (r["delta_24h"] or 0) / base, 1)

    snap = {"taken_at": NOW.isoformat(), "repos": list(repos.values())}
    (DATA / f"snapshot-{TODAY}.json").write_text(json.dumps(snap, indent=1, ensure_ascii=False))

    write_report(list(repos.values()), prev is not None)


def fmt_row(i, r):
    d = r["delta_24h"]
    tag = {"izmjereno": "", "procjena": " ~", "nov": " 🆕"}.get(r["mode"], "")
    desc = r["description"][:140].replace("|", "/")
    return (f"| {i} | [{r['name']}]({r['url']}) | **+{d}**{tag} | {r['growth_pct']}% | "
            f"{r['stars']:,} | {r['age_days']}d | {desc} |")


def write_report(repos, measured):
    ranked = [r for r in repos if r["delta_24h"]]
    ranked.sort(key=lambda r: (r["delta_24h"], r["growth_pct"]), reverse=True)

    lines = [f"# 🛰️ Skill Radar — {TODAY}", ""]
    if measured:
        lines.append("Rast = stvarne nove zvjezdice u zadnja 24h (usporedba sa jučerašnjim snapshotom).")
    else:
        lines.append("⚠️ Prvi run — nema jučerašnjeg snapshota, pa je rast **procjena** "
                     "(zvjezdice / starost repoa, samo repoi mlađi od 30 dana). "
                     "Od sutra su brojke stvarne.")
    lines += ["", "`~` = procjena · 🆕 = repo nov u zadnja 2 dana", ""]

    header = ["| # | Repo | +⭐ 24h | rast | ⭐ ukupno | starost | opis |",
              "|---|---|---|---|---|---|---|"]
    for cat, title in [("marketing", "📣 Marketing"), ("biznis", "💼 Biznis"), ("dizajn", "🎨 Dizajn")]:
        rows = [r for r in ranked if cat in r["categories"]][:10]
        lines += [f"## {title}", ""]
        if rows:
            lines += header + [fmt_row(i, r) for i, r in enumerate(rows, 1)]
        else:
            lines.append("_Ništa novo danas._")
        lines.append("")

    lines += ["## 🔥 Top 15 općenito (sve kategorije)", ""]
    lines += header + [fmt_row(i, r) for i, r in enumerate(ranked[:15], 1)]
    lines.append("")

    out = REPORTS / f"{TODAY}.md"
    out.write_text("\n".join(lines))
    (REPORTS / "LATEST.md").write_text("\n".join(lines))
    print(f"Izvještaj: {out}")


if __name__ == "__main__":
    main()
