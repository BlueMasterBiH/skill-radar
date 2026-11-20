"""One entry point for the commands: analyse if needed, make a plan if missing, render a style (or all).

usage:
  python run.py list                                   styles and what they do
  python run.py <style> <clip> [--check]               render (or a contact sheet with --check)
  python run.py all <clip>                             every style + a lineup grid
options: --out DIR (default: <clip folder>/<clip name>_arcads-cut)  --cache DIR  --audio  --opt key=value

The cache (matte, face track, words, plan.json, assets/) lives in <out>/.cache unless --cache is given.
Polish <cache>/plan.json (labels, sub lines, title) before the final render for the best result.
"""
import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
STYLES = ENGINE / "styles"


def styles():
    out = {}
    for p in sorted(STYLES.glob("*.py")):
        doc = ast.get_docstring(ast.parse(p.read_text())) or ""
        out[p.stem.replace("_", "-")] = " ".join(doc.split())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("style")
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--out")
    ap.add_argument("--cache")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--opt", action="append", default=[])
    a = ap.parse_args()
    all_styles = styles()
    if a.style == "list":
        for name, doc in all_styles.items():
            print(f"/{name}\n    {doc}\n")
        return
    if not a.clip:
        sys.exit("give a clip path")
    clip = Path(a.clip).expanduser().resolve()
    out = Path(a.out).expanduser() if a.out else clip.parent / f"{clip.stem}_arcads-cut"
    cache = Path(a.cache).expanduser() if a.cache else out / ".cache"
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    if not (cache / "plate.png").exists():
        print("analysing the clip (one time, about a minute)...", flush=True)
        subprocess.run([py, str(ENGINE / "prep.py"), str(clip), str(cache)], check=True)
    if not (cache / "plan.json").exists():
        subprocess.run([py, str(ENGINE / "plan.py"), str(cache)], check=True, capture_output=True)
        print(f"PLAN_CREATED {cache / 'plan.json'}  (polish labels, sub lines and title, then render again)", flush=True)
    names = list(all_styles) if a.style == "all" else [a.style.lstrip("/").replace("_", "-")]
    for name in names:
        if name not in all_styles:
            sys.exit(f"unknown style {name}. Try: python run.py list")
        target = out / (f"{name}_check.jpg" if a.check else f"{name}.mp4")
        cmd = [py, str(ENGINE / "render.py"), name, str(clip), str(cache), str(target)]
        cmd += ["--sheet", "--every", "11"] if a.check else []
        cmd += ["--audio"] if a.audio else []
        for o in a.opt:
            cmd += ["--opt", o]
        subprocess.run(cmd, check=True)
        print(f"OUTPUT {target}", flush=True)
    if a.style == "all" and not a.check:
        vids = [str(out / f"{n}.mp4") for n in names]
        cols = 5
        rows = (len(vids) + cols - 1) // cols
        inputs = sum([["-i", v] for v in vids], [])
        fc = "".join(f"[{k}:v]scale=360:640,setsar=1[v{k}];" for k in range(len(vids)))
        fc += "".join(f"[v{k}]" for k in range(len(vids)))
        layout = "|".join(f"{(k % cols) * 360}_{(k // cols) * 640}" for k in range(len(vids)))
        fc += f"xstack=inputs={len(vids)}:layout={layout}:fill=black[g]"
        subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", fc, "-map", "[g]", "-an",
                        "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p", str(out / "00_lineup.mp4")],
                       check=True)
        print(f"OUTPUT {out / '00_lineup.mp4'}  ({rows} rows)")


if __name__ == "__main__":
    main()
