"""The edit plan: what every style needs to know about a clip, written once per clip.

usage: python plan.py <cache_dir>        writes <cache_dir>/plan.json from words.json if it does not exist

A plan is a list of beats (one idea each) plus a few texts. prep.py gives the words; this file guesses beats and
keywords; Claude then edits plan.json (labels, sub lines, title) before rendering.

plan.json
{
  "title": "Stack your prompts",            hook / headline
  "kicker": "PROMPT STACKING",              small label
  "name": "...", "role": "...", "handle": "...",
  "hot": ["stack", "camera", ...],          words that get the accent treatment in captions
  "beats": [
    {"t0": 0.0, "t1": 1.38, "text": "And you can stack them", "key": "stack", "key_t": 0.46,
     "label": "Stack", "sub": "TWO LOOKS"}
  ]
}
key_t is when the keyword starts: styles land their main move on it.
"""
import json
import sys
from pathlib import Path

STOP = set("a an and the of to in on at for with you your i we it is are be can so just this that then them they "
           "there here my me our us as or but if into out up down not do does did was were have has had will".split())


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


JOIN = {"and", "but", "so", "then", "in", "with", "because", "or", "to", "into", "on"}


def auto(words, max_beats=6):
    """Split at pauses (>0.18 s) and sentence punctuation, then split long beats before joining words until
    there is about one beat per 1.3 s (3-6 beats); pick a one- or two-word keyword per beat."""
    groups, cur = [], []
    for k, w in enumerate(words):
        cur.append(w)
        gap = words[k + 1]["s"] - w["e"] if k + 1 < len(words) else 9
        if gap > 0.18 or w["w"].rstrip()[-1:] in ".,!?;:":
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    dur = (words[-1]["e"] - words[0]["s"]) if words else 0
    target = int(min(max_beats, max(3, round(dur / 1.3))))
    while len(groups) < target:  # split the longest beat, preferring a cut before a joining word
        k = max(range(len(groups)), key=lambda j: groups[j][-1]["e"] - groups[j][0]["s"])
        g = groups[k]
        if len(g) < 4:
            break
        mid = (g[0]["s"] + g[-1]["e"]) / 2
        cuts = range(2, len(g) - 1)
        cut = min(cuts, key=lambda j: abs(g[j]["s"] - mid) - (0.6 if norm(g[j]["w"]) in JOIN else 0))
        groups[k:k + 1] = [g[:cut], g[cut:]]
    while len(groups) > max_beats:  # merge the shortest neighbour pair
        k = min(range(len(groups) - 1), key=lambda j: len(groups[j]) + len(groups[j + 1]))
        groups[k:k + 2] = [groups[k] + groups[k + 1]]
    beats = []
    for k, g in enumerate(groups):
        content = [w for w in g if norm(w["w"]) not in STOP] or g
        key = max(content, key=lambda w: len(norm(w["w"])))
        j = g.index(key)  # pair with a neighbouring content word for a two-word key ("camera angle")
        pair = [g[j - 1]] if j > 0 and norm(g[j - 1]["w"]) not in STOP else []
        if not pair and j + 1 < len(g) and norm(g[j + 1]["w"]) not in STOP:
            pair = [g[j + 1]]
        kws = sorted([key] + pair, key=lambda w: w["s"])
        key = dict(kws[0], w=" ".join(norm(w["w"]) for w in kws))
        t1 = groups[k + 1][0]["s"] if k + 1 < len(groups) else g[-1]["e"] + 0.4
        beats.append({"t0": round(g[0]["s"], 3), "t1": round(t1, 3), "text": " ".join(w["w"] for w in g).strip(" ,."),
                      "key": key["w"], "key_t": key["s"], "label": key["w"].capitalize(), "sub": ""})
    hot = sorted({x for b in beats for x in b["key"].split()})
    return {"title": beats[0]["text"].capitalize() if beats else "", "kicker": "", "name": "", "role": "",
            "handle": "", "hot": hot, "beats": beats}


class Plan:
    def __init__(self, clip, opts=None):
        p = clip.cache / "plan.json"
        data = json.load(open(p)) if p.exists() else auto(clip.words)
        data.update({k: v for k, v in (opts or {}).items() if k in ("title", "kicker", "name", "role", "handle")})
        self.d = data
        self.beats = data["beats"]
        self.hot = {norm(h) for h in data.get("hot", [])} | {norm(x) for b in self.beats for x in b["key"].split()}
        self.duration = clip.n / clip.fps
        self.beats[-1]["t1"] = max(self.beats[-1]["t1"], self.duration)
        for b in self.beats:  # the beat's own words, and when each keyword lands
            b["words"] = [w for w in clip.words if b["t0"] - 0.02 <= w["s"] < b["t1"] - 0.02]
            keys = set(b["key"].split())
            kw = [w for w in b["words"] if norm(w["w"]) in keys] or [w for w in b["words"] if w["s"] >= b["key_t"]][:1]
            b["kw"] = kw or b["words"][:1]
            b["last_t"] = b["kw"][-1]["s"] if b["kw"] else b["key_t"]

    def __getattr__(self, k):
        return self.d.get(k, "")

    def at(self, t):
        """(index, beat, progress 0..1 through the beat) at time t."""
        for k, b in enumerate(self.beats):
            if t < b["t1"] or k == len(self.beats) - 1:
                return k, b, min(1.0, max(0.0, (t - b["t0"]) / max(b["t1"] - b["t0"], 1e-3)))
        return 0, self.beats[0], 0.0


if __name__ == "__main__":
    cache = Path(sys.argv[1])
    out = cache / "plan.json"
    if not out.exists():
        json.dump(auto(json.load(open(cache / "words.json"))), open(out, "w"), indent=1)
    print(out.read_text())
