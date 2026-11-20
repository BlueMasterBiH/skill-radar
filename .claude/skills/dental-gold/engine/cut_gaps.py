"""Tighten long pauses in a prepared hook folder, keeping everything in sync.

usage: cut_gaps.py <workdir> [--first N] [--min 0.9] [--keep 0.2] [--ranges "a-b,c-d"]

Finds silences of at least --min seconds (ffmpeg silencedetect) and removes all but --keep seconds at each side,
so a 1.8 s pause becomes 0.4 s. --first N only touches the first N such pauses (e.g. hook->lead and lead->body).
--ranges cuts exact ranges instead. It cuts clip.mp4 (video and audio), words.json, and, when present, the
arcads matte frames (.arc/matte) and face track (.arc/face.json). Originals go to <workdir>/_before_cut/ once.
Storyboard anchors are phrases, so they follow the new timing; re-render afterwards.
"""
import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

FPS = 30


def silences(path, noise="-35dB", d=0.4):
    r = subprocess.run(["ffmpeg", "-i", str(path), "-af", f"silencedetect=noise={noise}:d={d}", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r)]
    return list(zip(st, en))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--first", type=int)
    ap.add_argument("--min", type=float, default=0.9)
    ap.add_argument("--keep", type=float, default=0.2)
    ap.add_argument("--ranges")
    ap.add_argument("--noise", default="-35dB", help="silence threshold; use -45dB for a soft-spoken voice")
    a = ap.parse_args()
    wd = Path(a.workdir)
    bak = wd / "_before_cut"
    if not bak.exists():                      # always cut from the untouched originals
        bak.mkdir()
        shutil.copy(wd / "clip.mp4", bak / "clip.mp4")
        shutil.copy(wd / "words.json", bak / "words.json")
        if (wd / ".arc" / "face.json").exists():
            shutil.copy(wd / ".arc" / "face.json", bak / "arc_face.json")
        if (wd / ".arc" / "matte").exists():
            shutil.copytree(wd / ".arc" / "matte", bak / "matte")
    src = bak / "clip.mp4"
    if a.ranges:
        cuts = [tuple(map(float, r.split("-"))) for r in a.ranges.split(",")]
    else:
        sil = [(s, e) for s, e in silences(src, a.noise) if s > 0.05 and e - s >= a.min]
        if a.first:
            sil = sil[:a.first]
        cuts = [(s + a.keep, e - a.keep) for s, e in sil]
    # snap to whole frames: frame i is removed when fa <= i < fb
    fcuts = [(int(round(s * FPS)), int(round(e * FPS))) for s, e in cuts]
    fcuts = [(fa, fb) for fa, fb in fcuts if fb > fa]
    cuts = [(fa / FPS, fb / FPS) for fa, fb in fcuts]
    for s, e in cuts:
        print(f"cut {s:.2f}-{e:.2f} ({e - s:.2f}s)", flush=True)
    vexpr = "+".join(f"between(n,{fa},{fb - 1})" for fa, fb in fcuts) or "0"
    aexpr = "+".join(f"between(t,{s:.4f},{e - 0.0001:.4f})" for s, e in cuts) or "0"
    tmp = wd / "clip_cut.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-vf", f"select='not({vexpr})',setpts=N/{FPS}/TB",
                    "-af", f"aselect='not({aexpr})',asetpts=N/SR/TB",
                    "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-r", str(FPS), "-c:a", "aac", "-b:a", "192k",
                    str(tmp)], check=True)
    tmp.replace(wd / "clip.mp4")

    def removed_before(t):
        return sum(min(e, t) - s for s, e in cuts if t > s)

    words = json.load(open(bak / "words.json"))
    for w in words:
        w["s"] = round(w["s"] - removed_before(w["s"]), 3)
        w["e"] = round(w["e"] - removed_before(w["e"]), 3)
    json.dump(words, open(wd / "words.json", "w"), ensure_ascii=False, indent=0)
    if (wd / ".arc").exists():
        json.dump(words, open(wd / ".arc" / "words.json", "w"), ensure_ascii=False, indent=0)

    def kept(i):
        return not any(fa <= i < fb for fa, fb in fcuts)

    if (bak / "arc_face.json").exists():
        faces = json.load(open(bak / "arc_face.json"))
        json.dump([f for i, f in enumerate(faces) if kept(i)], open(wd / ".arc" / "face.json", "w"))
    if (bak / "matte").exists():
        md = wd / ".arc" / "matte"
        shutil.rmtree(md)
        md.mkdir()
        files = sorted((bak / "matte").glob("*.png"))
        j = 0
        for i, f in enumerate(files):
            if kept(i):
                shutil.copy(f, md / f"{j:04d}.png")
                j += 1
        meta_p = wd / ".arc" / "meta.json"
        if meta_p.exists():
            meta = json.load(open(meta_p))
            meta["n"] = j
            json.dump(meta, open(meta_p, "w"))
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_frames", "-of", "csv=p=0", str(wd / "clip.mp4")],
                           capture_output=True, text=True).stdout.strip() or 0)
    print(f"CUT_DONE {wd}  removed {sum(e - s for s, e in cuts):.2f}s  frames {n}"
          + (f"  matte {j}" if (bak / 'matte').exists() else ""), flush=True)


if __name__ == "__main__":
    main()
