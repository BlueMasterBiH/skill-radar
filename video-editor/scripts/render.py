# Render a cut list to MP4. Each piece is its own frame-accurate seeked input (-ss/-to before -i),
# then one concat filter and a single H.264/AAC encode with no forced rate (-fps_mode passthrough).
# (Trimming 11 branches of one 1080p100 input in a single graph ran out of memory.)
import json, subprocess, sys
raw, cuts, out = sys.argv[1], sys.argv[2], sys.argv[3]
segs = json.load(open(cuts))["segments"]
args = ["ffmpeg", "-v", "error", "-y"]
for s in segs: args += ["-ss", f"{s['start']:.2f}", "-to", f"{s['end']:.2f}", "-i", raw]
n = len(segs)
graph = "".join(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];[{i}:a]asetpts=PTS-STARTPTS[a{i}];" for i in range(n))
graph += "".join(f"[v{i}][a{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=1[v][a]"
args += ["-filter_complex", graph, "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-fps_mode", "passthrough", "-movflags", "+faststart", out]
subprocess.run(args, check=True)
