# Render a cut list to MP4: trim/atrim per piece, concat, one H.264/AAC encode, no forced rate.
import json, subprocess, sys
raw, cuts, out = sys.argv[1], sys.argv[2], sys.argv[3]
segs = json.load(open(cuts))["segments"]
f = []
for i, s in enumerate(segs):
    f.append(f"[0:v]trim=start={s['start']}:end={s['end']},setpts=PTS-STARTPTS[v{i}];"
             f"[0:a]atrim=start={s['start']}:end={s['end']},asetpts=PTS-STARTPTS[a{i}];")
f.append("".join(f"[v{i}][a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[v][a]")
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", raw, "-filter_complex", "".join(f),
    "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "192k", "-fps_mode", "passthrough", "-movflags", "+faststart", out], check=True)
