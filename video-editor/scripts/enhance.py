# Edited render: cut + alternating punch-ins at each cut, word-by-word captions,
# CTA phone number on screen, cleaned and loudness-normalised audio. One encode.
# usage: enhance.py raw.mov cuts.json out.mp4 "<phone>"
import json, subprocess, sys
raw, cuts, out, phone = sys.argv[1:5]
segs = json.load(open(cuts))["segments"]
words = [w for w in json.load(open("transcript/words.json"))["words"] if "start" in w]
FIX = {"endodunca": "endodoncija", "Endodunca": "Endodoncija", "isprima": "ispunima", "kos": "kost",
       "osmije.": "osmijeh.", "Bolji": "Bol", "prijegled": "pregled", "prijegled,": "pregled,",
       "posle": "poslije", "pristaniš": "pristaneš", "misa": "misao", "krunjice": "krunice"}
ZOOM = 1.15
# sentence pieces: words assigned by sentence order (same as cuts.json labels)
sents, cur = [], []
for w in words:
    cur.append(w)
    if w["word"].endswith((".", "?")): sents.append(cur); cur = []
allsegs = json.load(open("transcript/cuts.json"))["segments"]
def ts(t): h=int(t//3600); m=int(t%3600//60); s=t%60; return f"{h}:{m:02d}:{s:05.2f}"
ev, off = [], 0.0
for seg in segs:
    sw = sents[[s["start"] for s in allsegs].index(seg["start"])]
    toks = []
    for w in sw:
        t = FIX.get(w["word"], w["word"])
        if w["word"] == "o" and toks and toks[-1][0] == "misao": continue
        toks.append([t, max(w["start"], seg["start"])])
    groups, g = [], []
    for t in toks:
        g.append(t)
        if len(g) == 3 or t[0][-1] in ",.?": groups.append(g); g = []
    if g: groups.append(g)
    flat = [t for gr in groups for t in gr]
    for gi, gr in enumerate(groups):
        for wi, (txt, st) in enumerate(gr):
            idx = flat.index(gr[wi])
            en = flat[idx+1][1] if idx+1 < len(flat) else seg["end"]
            en = min(en, seg["end"]); 
            if en - st < 0.05: continue
            line = " ".join(("{\\c&H0000E5FF&}" + x.rstrip(",.") + "{\\c&HFFFFFF&}") if j == wi else x.rstrip(",.")
                            for j, (x, _) in enumerate(gr))
            ev.append(f"Dialogue: 0,{ts(off+st-seg['start'])},{ts(off+en-seg['start'])},Cap,,0,0,0,,{line}")
    off += seg["end"] - seg["start"]
last = segs[-1]; cta_start = off - (last["end"] - last["start"])
ev.append(f"Dialogue: 1,{ts(cta_start)},{ts(off)},Cta,,0,0,0,,{{\\fad(200,0)}}📞 {phone}".replace("📞 ", ""))
ass = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Inter Display Black,94,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,7,3,2,60,60,640,1
Style: Cta,Inter Display Black,76,&H00FFFFFF,&H00FFFFFF,&H005A7A1E,&H005A7A1E,0,0,0,0,100,100,0,0,3,22,0,2,60,60,380,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(ev) + "\n"
assf = out.rsplit(".", 1)[0] + ".ass"; open(assf, "w").write(ass)
args = ["ffmpeg", "-v", "error", "-y"]
for s in segs: args += ["-ss", f"{s['start']:.2f}", "-to", f"{s['end']:.2f}", "-i", raw]
n = len(segs); g = ""
for i, s in enumerate(segs):
    d = s["end"] - s["start"]
    z = f"crop=iw/{ZOOM}:ih/{ZOOM}:(iw-iw/{ZOOM})/2:(ih-ih/{ZOOM})*0.15,scale=1080:1920," if i % 2 else ""
    g += f"[{i}:v]{z}setsar=1,setpts=PTS-STARTPTS[v{i}];"
    g += f"[{i}:a]asetpts=PTS-STARTPTS,afade=t=in:d=0.015,afade=t=out:st={d-0.02:.3f}:d=0.02[a{i}];"
g += "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[vc][ac];"
g += f"[vc]ass={assf}[v];[ac]highpass=f=80,afftdn=nf=-30,acompressor=threshold=-24dB:ratio=3:attack=10:release=150,loudnorm=I=-14:TP=-1.5:LRA=9,aresample=48000[a]"
args += ["-filter_complex", g, "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-fps_mode", "passthrough", "-movflags", "+faststart", out]
subprocess.run(args, check=True)
