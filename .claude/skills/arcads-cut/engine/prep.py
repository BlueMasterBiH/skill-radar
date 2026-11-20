"""Analyse a raw talking-head clip once, so every style can reuse the results.

usage: python prep.py <clip> <cache_dir>

Writes into <cache_dir>:
  meta.json     fps, frame count, source size
  matte/NNNN.png  person alpha at source resolution (MediaPipe + guided filter, 8-bit)
  face.json     smoothed face box per frame, in source pixels [cx, cy, w, h]
  words.json    word timings [{w, s, e}] (Whisper CLI, small.en)
Run it with a Python that has mediapipe and opencv (the effects repo's .venv works).
  plate.png     clean background plate at half resolution (person removed)
Each step is skipped when its output already exists.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def probe(src):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets:stream_side_data=rotation",
                          "-of", "json", src], capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    num, den = map(int, s["r_frame_rate"].split("/"))
    rot = next((int(d.get("rotation", 0)) for d in s.get("side_data_list", []) if "rotation" in d), 0)
    w, h = s["width"], s["height"]
    if abs(rot) == 90:
        w, h = h, w
    return {"fps": num / den, "n": int(s["nb_read_packets"]), "w": w, "h": h, "rotation": rot}


def frames(src, w, h, scale=1.0):
    """Yield RGB uint8 frames, optionally downscaled."""
    ow, oh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    vf = [] if scale == 1.0 else ["-vf", f"scale={ow}:{oh}:flags=area"]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", src, *vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, bufsize=ow * oh * 3 * 2)
    size = ow * oh * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
    p.wait()


def guided(I, p, r, eps=1e-3):
    box = lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1))
    mI, mp_ = box(I), box(p)
    a = (box(I * p) - mI * mp_) / (box(I * I) - mI * mI + eps)
    b = mp_ - a * mI
    return np.clip(box(a) * I + box(b), 0, 1)


def run_matte(src, meta, out):
    """MediaPipe selfie segmenter, refined against the full-resolution frame with a guided filter."""
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    out.mkdir(parents=True, exist_ok=True)
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(HERE / "selfie_segmenter.tflite")),
        running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True, output_category_mask=False))
    r = max(4, meta["w"] // 240)
    prev = None
    for i, rgb in enumerate(frames(src, meta["w"], meta["h"])):
        small = cv2.resize(rgb, (meta["w"] // 4, meta["h"] // 4), interpolation=cv2.INTER_AREA)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(small))
        m = seg.segment_for_video(img, int(i * 1000 / meta["fps"])).confidence_masks[-1].numpy_view()
        m = cv2.resize(m.astype(np.float32).reshape(m.shape[0], m.shape[1]), (meta["w"], meta["h"]),
                       interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        m = guided(gray, m, r)
        m = np.clip((m - 0.5) * 1.8 + 0.5, 0, 1)
        if prev is not None:
            m = 0.7 * m + 0.3 * prev
        prev = m
        cv2.imwrite(str(out / f"{i:04d}.png"), (m * 255 + 0.5).astype(np.uint8))
        if i % 25 == 0:
            print(f"  matte {i}/{meta['n']}", flush=True)


def run_face(src, meta, out):
    s = 960 / meta["w"]
    fw, fh = int(round(meta["w"] * s / 2) * 2), int(round(meta["h"] * s / 2) * 2)
    fd = cv2.FaceDetectorYN.create(str(HERE / "yunet.onnx"), "", (fw, fh), 0.6, 0.3, 20)
    boxes = []
    for rgb in frames(src, meta["w"], meta["h"], s):
        _, det = fd.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if det is None or len(det) == 0:
            boxes.append(None)
            continue
        x, y, w, h = max(det, key=lambda d: d[2] * d[3])[:4]
        boxes.append([(x + w / 2) / s, (y + h / 2) / s, w / s, h / s])
    last = next(b for b in boxes if b)
    filled = []
    for b in boxes:
        last = b or last
        filled.append(last)
    arr = np.array(filled, float)
    k = 5  # light temporal smoothing, keeps the head moves
    pad = np.pad(arr, ((k, k), (0, 0)), mode="edge")
    smooth = np.stack([pad[i:i + 2 * k + 1].mean(0) for i in range(len(arr))])
    json.dump([[round(v, 1) for v in b] for b in smooth], open(out, "w"))


def run_words(src, out):
    """Word timestamps: faster-whisper (installed by setup.sh), else the openai-whisper CLI if present."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(os.environ.get("WHISPER_MODEL", "small"), device="cpu", compute_type="int8")
        segs, _ = model.transcribe(src, word_timestamps=True, vad_filter=True)
        words = [{"w": w.word.strip(), "s": round(w.start, 3), "e": round(w.end, 3)} for s in segs for w in (s.words or [])]
    except ImportError:
        import shutil
        if not shutil.which("whisper"):
            sys.exit("NEED_TRANSCRIBER: run scripts/setup.sh (installs faster-whisper), or write words.json from "
                     "arcads_transcribe (format srt) as [{w, s, e}] and run again")
        tmp = out.parent / "whisper"
        tmp.mkdir(exist_ok=True)
        subprocess.run(["whisper", src, "--model", os.environ.get("WHISPER_MODEL", "small.en"), "--language", "en",
                        "--word_timestamps", "True", "--output_format", "json", "--output_dir", str(tmp),
                        "--fp16", "False"], check=True, capture_output=True)
        r = json.load(open(tmp / (Path(src).stem + ".json")))
        words = [{"w": w["word"].strip(), "s": round(w["start"], 3), "e": round(w["end"], 3)}
                 for seg in r["segments"] for w in seg.get("words", [])]
    json.dump(words, open(out, "w"), indent=0)


def run_plate(src, meta, cache, out):
    s = 0.5
    acc, cnt = None, None
    for i, rgb in enumerate(frames(src, meta["w"], meta["h"], s)):
        if i % 3:
            continue
        a = cv2.imread(str(cache / "matte" / f"{i:04d}.png"), cv2.IMREAD_GRAYSCALE)
        a = cv2.resize(a, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        bg = (cv2.dilate(a, np.ones((25, 25), np.uint8)) < 8).astype(np.float32)[..., None]
        if acc is None:
            acc, cnt = np.zeros(rgb.shape, np.float64), np.zeros(bg.shape, np.float64)
        acc += rgb * bg
        cnt += bg
    plate = (acc / np.maximum(cnt, 1)).astype(np.uint8)
    hole = (cnt[..., 0] < 1).astype(np.uint8) * 255
    plate = cv2.inpaint(plate, hole, 15, cv2.INPAINT_TELEA)
    plate = cv2.GaussianBlur(plate, (0, 0), 6)  # the wall is soft anyway; hides inpaint seams
    cv2.imwrite(str(out), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))


def main():
    src, cache = sys.argv[1], Path(sys.argv[2])
    cache.mkdir(parents=True, exist_ok=True)
    meta_p = cache / "meta.json"
    if not meta_p.exists():
        json.dump(probe(src), open(meta_p, "w"))
    meta = json.load(open(meta_p))
    print("meta", meta, flush=True)
    if not (cache / "matte" / f"{meta['n'] - 1:04d}.png").exists():
        run_matte(src, meta, cache / "matte")
        n = len(list((cache / "matte").glob("*.png")))  # packet counts can overstate the decoded frames
        if n and n != meta["n"]:
            meta["n"] = n
            json.dump(meta, open(meta_p, "w"))
    if not (cache / "face.json").exists():
        run_face(src, meta, cache / "face.json")
    if not (cache / "words.json").exists():
        run_words(src, cache / "words.json")
    if not (cache / "plate.png").exists():
        run_plate(src, meta, cache, cache / "plate.png")
    print("PREP_DONE", cache, flush=True)


if __name__ == "__main__":
    main()
