# Build transcript/cuts.json: speech regions between long pauses (>=0.6 s at -45 dB),
# padded 0.08 s before / 0.12 s after, snapped to the footage frame grid.
import json, re, sys
FPS = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
words = json.load(open("transcript/words.json"))["words"]
sil = [tuple(map(float, re.findall(r"[0-9.]+", l))) for l in open("transcript/silences.txt")]
keep_pauses = {(55.6609, 56.3211)}          # mid-sentence comma pause, keep it
sil = [s for s in sil if s not in keep_pauses]
regions = [(sil[i][1], sil[i+1][0]) for i in range(len(sil)-1)]
snap = lambda t: round(round(t * FPS) / FPS, 4)
segs = []
for i, (a, b) in enumerate(regions):
    ws = [w for w in words if "start" in w and a - 0.3 <= w["start"] < b]
    s = min([a] + [ws[0]["start"]] if ws else [a]) - 0.08
    e = b + 0.12
    if segs and s < segs[-1]["end"]:
        mid = (segs[-1]["end"] + s) / 2; segs[-1]["end"] = snap(mid); s = mid
    segs.append({"start": snap(max(s, 0)), "end": snap(e), "text": " ".join(w["word"] for w in ws)})
json.dump({"segments": segs}, open("transcript/cuts.json", "w"), ensure_ascii=False, indent=1)
tot = sum(x["end"] - x["start"] for x in segs)
for i, x in enumerate(segs, 1): print(f'{i:2}. {x["start"]:6.2f}–{x["end"]:6.2f}  {x["text"]}')
print(f"{len(segs)} pieces, {tot:.2f} s (raw 88.80 s)")
# Labels: aligner pushes sentence-final words into pauses, so relabel each piece by sentence order.
