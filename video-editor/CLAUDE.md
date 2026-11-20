# Video editor notes
- Runs in a cloud Linux container (no GPU): WhisperX large-v3 on CPU int8, venv ~/.cache/video-editor/whisperx-venv.
- Raw clips come from Google Drive via the share link: curl "https://drive.usercontent.google.com/download?id=<ID>&export=download&confirm=t".
- Model downloads need HF_HUB_DISABLE_XET=1; NLTK punkt_tab needs NLTK_ALLOW_PROXIED_URLOPEN=1.
- Language is Bosnian (bs); no default align model, so use the Croatian one ("hr").
- Footage: iPhone-style .mov, 1920x1080 stored with rotation 90 => vertical 1080x1920, 100 fps, PCM audio, no timecode.
- Recording is quiet (speech around -40 to -45 dBFS). Aligner stretches sentence-final words into the pauses after them,
  so cut points come from silencedetect=noise=-45dB:d=0.6 (transcript/silences.txt), not from word end times.
- Delivery: no editing app; render cut.mp4 and upload it to the user's Google Drive.
