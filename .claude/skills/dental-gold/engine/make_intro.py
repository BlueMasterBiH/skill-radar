"""Gold explainer intro (first seconds): the video springs into a card under a numbered node.

Needs the arcads-cut skill (it provides the explainer-panel-reel style); it is recoloured to the brand gold here.
usage: make_intro.py <workdir>   (reads storyboard.json "intro", writes <workdir>/intro.mp4)
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

wd = Path(sys.argv[1]).resolve()
ARC = Path(os.environ.get("ARCADS_CUT", Path.home() / ".claude/skills/arcads-cut"))
if not (ARC / "engine/styles/explainer_panel_reel.py").exists():
    sys.exit(f"NEED_ARCADS_CUT: install the arcads-cut skill (looked in {ARC}) or drop \"intro\" from the storyboard")
sb = json.load(open(wd / "storyboard.json"))
intro = sb["intro"]
words = json.load(open(wd / "words.json"))
brand = sb.get("brand", {})
gold = brand.get("gold", [214, 178, 94])

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elements as E  # noqa: E402
import render as R  # noqa: E402
E.words[:] = words
ev, lock, until = R.build({"events": [], "intro": intro}, words)
key_t = R.build({"events": [], "intro": {"until": intro["key"]}}, words)[2]
dur = until + 1.0

tmp = wd / ".intro"
shutil.rmtree(tmp, ignore_errors=True)
cache = tmp / "cache"
cache.mkdir(parents=True)
clip = tmp / "intro_src.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wd / "clip.mp4"), "-t", f"{dur:.2f}", "-c:v", "libx264",
                "-crf", "16", "-preset", "fast", "-c:a", "aac", str(clip)], check=True)
json.dump([w for w in words if w["s"] < dur], open(cache / "words.json", "w"), ensure_ascii=False)
steps = int(intro.get("steps", 4))
beats = [{"t0": 0.0, "t1": dur, "text": "", "key": intro["key"], "key_t": key_t,
          "label": intro["label"], "sub": intro.get("sub", "")}]
for k in range(1, steps):  # placeholders after the end: only their dots show in the progress rail
    beats.append({"t0": dur + k, "t1": dur + k + 1, "text": "", "key": "", "key_t": dur + k + 0.5,
                  "label": "·", "sub": ""})
json.dump({"title": intro.get("label", ""), "kicker": intro.get("kicker", ""), "name": "", "role": "", "handle": "",
           "hot": [], "beats": beats}, open(cache / "plan.json", "w"), ensure_ascii=False)

py = ARC / ".venv/bin/python"
py = str(py if py.exists() else sys.executable)
subprocess.run([py, str(ARC / "engine/prep.py"), str(clip), str(cache)], check=True)
wrapper = tmp / "gold_render.py"
wrapper.write_text(f'''import sys, numpy as np
sys.path.insert(0, {str(ARC / "engine")!r})
import importlib
m = importlib.import_module("styles.explainer_panel_reel")
m.C1 = np.float32({list(gold)})
m.C2 = np.float32({[int(v * 0.7) for v in gold]})
import render
sys.argv = ["render.py"] + sys.argv[1:]
render.main()
''')
subprocess.run([py, str(wrapper), "explainer-panel-reel", str(clip), str(cache), str(wd / "intro.mp4")], check=True)
print("OUTPUT", wd / "intro.mp4")
