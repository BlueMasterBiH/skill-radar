import json, os, sys, whisperx
wav, out = sys.argv[1], sys.argv[2]
if os.path.exists(out): sys.exit(f"{out} exists, not re-transcribing")
audio = whisperx.load_audio(wav)
raw = out.replace(".json", "_raw.json")
if os.path.exists(raw): res = json.load(open(raw))
else:
    model = whisperx.load_model("large-v3", "cpu", compute_type="int8")
    res = model.transcribe(audio, batch_size=8)
    json.dump(res, open(raw, "w"), ensure_ascii=False)
lang = res["language"]; print("language:", lang)
# Bosnian/Serbian have no default align model; Croatian one fits the same language
align_lang = "hr" if lang in ("bs", "sr", "hr") else lang
am, meta = whisperx.load_align_model(language_code=align_lang, device="cpu")
al = whisperx.align(res["segments"], am, meta, audio, "cpu", return_char_alignments=False)
json.dump({"language": lang, "segments": al["segments"], "words": al.get("word_segments", [])}, open(out, "w"), ensure_ascii=False, indent=1)
print("saved", out)
