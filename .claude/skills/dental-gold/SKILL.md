---
name: dental-gold
description: Black-and-gold motion edit for dental clinic talking-head reels in BHS (Bosnian/Croatian/Serbian). From one raw recording (several hooks back to back, then lead, body and CTA) and its script, it cuts one version per hook and edits it in the clinic's house style - small chest captions, gold type lockups on key words, black UI cards, a comparison card, a checklist, drawn dental animations (veneer onto the front of a tooth, minimal-preparation profile) and a comment-box CTA, with a gold explainer-panel intro. Use when someone drops a dental talking-head clip with a script and asks for the gold / dental edit, the "hook 3 gold dental" look, or this template.
---

# dental-gold: the clinic's reel template

The person is a marketer, not an engineer. Reply in their language (usually BHS), keep it short, run every command yourself.
`DG` = this skill's folder. `PY` = `DG/.venv/bin/python` (numpy, opencv-python<5, pillow, faster-whisper). If it is missing, run `bash DG/scripts/setup.sh` once.

## Inputs
- **The recording**: one person facing the camera; all hooks said one after another, then lead, body and CTA once each, with a pause of 2 s or more between parts. Vertical is best; landscape gets centre-cropped.
- **The script** as a .md/.txt file with lines `**Hook 1:** ...`, `**Hook 2:** ...`, `**Lead:** ...`, `**Body:** ...`, `**CTA:** ...`. Save what they paste into a file first.

## Workflow
1. **Prepare**: `PY DG/engine/prepare.py "<clip>" "<script.md>" --out "<clip folder>/<name>_dental-gold" [--hooks 3]`
   - It prints each part with its time range. Check that they look sensible: a hook is 2-5 s, the body is the longest part.
   - Caption timing comes from Whisper (local, no API key). The first run downloads the model (~500 MB for `small`) and transcribes in about 1x real time on a laptop CPU. The transcript is cached in `<out>/whisper_<model>.json`, so re-runs are instant.
   - Each part prints `timing: whisper (NN% of words heard)`. `WHISPER_WEAK <part>` means that part fell back to estimated timing. Usually the script does not match what was said (an ad-lib or a changed sentence). Fix the script to match the recording, or rerun with `--whisper-model medium`, which is more exact but about 3x slower. `WHISPER_MISSING` means faster-whisper is not installed: run `PY -m pip install faster-whisper`.
   - `BLOCKS_MISMATCH` means the pauses did not split cleanly. It lists the speech blocks it found; pick one range per part and run again with `--blocks "a-b,c-d,..."` (order: hooks, lead, body, cta).
   - Each `<out>/hook<N>/` gets `clip.mp4`, `words.json`, `face.json` and `script.txt`.
2. **Write the storyboard**: `<out>/hook<N>/storyboard.json`. This is where your effort goes. Start from `DG/examples/viniri_hook3.storyboard.json` and follow the rules below.
3. **Intro** (optional, needs the arcads-cut skill): `PY DG/engine/make_intro.py <out>/hook<N>`. It writes `intro.mp4`, the gold explainer opener, in about 1-2 min. If arcads-cut is missing, drop `"intro"` from the storyboard.
4. **Preview**: `PY DG/engine/render.py <out>/hook<N> --preview 2.5,5,9,14,20,27,35,44,55`. Pick times that land on your events, then read `preview.jpg`. Fix the storyboard if:
   - text covers the face
   - a card is still empty when it shows
   - a lockup collides with a card
5. **Render**: `PY DG/engine/render.py <out>/hook<N>`. This writes `dental-gold.mp4` (1080x1920, 30 fps, with the voice) in about 2-3 min per minute of video.
6. **Music** (when asked): `PY DG/engine/music.py <out>/hook<N> [--track song.mp3] [--volume 0.2]`. It writes `dental-gold_music.mp4`, with the music ducked under the voice. Without `--track` it synthesises an original calm track, so there is nothing to license, which matters for paid ads. A song the person sends must be licensed for ads; Instagram's in-app music is not.
7. **Hand over** the file in one line. Remind them it is a draft: the doctor checks the dental animations and every claim before posting.

## Dynamic cut (when they want it livelier, or "not like last time")
The same storyboard plus camera cuts, giant words behind the speaker, drawn b-roll and sound effects. Needs the arcads-cut skill.
1. Person matte and clean wall: `<arcads>/.venv/bin/python <arcads>/engine/prep.py <out>/hook<N>/clip.mp4 <out>/hook<N>/.arc`. Copy `words.json` into `.arc/` first so it skips its own transcription. On Linux, MediaPipe needs `libegl1 libgles2` (apt). If it crashes after the matte is written, run it again; it resumes.
2. Add the dynamic events to `storyboard.json` (below), use fewer cards, and let b-roll carry the procedure.
3. Preview: `PY DG/engine/dynamic.py <out>/hook<N> --preview 1.5,4,11,...` and read `preview_dyn.jpg`.
4. Render: `PY DG/engine/dynamic.py <out>/hook<N>` writes `dental-gold.mp4` and `sfx.json`. Then `music.py` mixes music, sound effects and voice.

| type | fields | use it for |
|---|---|---|
| `depth` | at, to, text | Giant gold word behind the speaker. She slides right over the clean wall and the word sits left, partly behind her. 1-2 words, about one every 8-12 s, on the idea of the sentence (ENDODONCIJA, PREGLED, BOL). The camera goes wide for it. |
| `broll` | at, to, kind, marks, title | Full-screen drawn b-roll with the speaker in a window top right. Kinds: `xray` (marks zub, korijen, kost), `root` (remove, clean, fill, keep), `restore` (implant, krunica, most, osmijeh), `pain`, `versus` (left, right), `extract` (out). 2-9 s each, where she describes something you can show. |
| `cam` | at, shot (W/M/C), mode (cut/push) | Force a camera change. Otherwise wide, medium and close cut on sentence ends about every 2-3 s. |

Hooks: give each hook its own b-roll (`pain`, `versus`, `extract`...) and depth word, so the three versions open differently.

## Storyboard
Times are seconds or phrases from the script: `"dvije vrste"` (when the phrase starts), `"kod nas+0.6"`, `"pitaj-0.1"`, `"brušenje#2"` (the second time it is said), `"end"`. Phrases match the script words ignoring case and punctuation, so copy them from `script.txt`.

| type | fields | use it for |
|---|---|---|
| `badge` | at, to, text | Name tag with an arrow to the face, right after the hook. Text is what the person gives (for now "Doktorica"). |
| `strip` | at, to, text | Gold topic strip under the badge: the video's promise in caps, max ~26 characters. |
| `lockup` | at, to, big, above, below | Big gold word(s) with small white words around. One idea word per sentence (ESTETIKA, PITAJ). Captions hide while it shows, and the camera punches in. |
| `card` | at, to, title, sub, icon (`check`, `calendar`, `tooth`) | One fact from the script. |
| `compare` | at, to, left{title, at, bullets[[text, at]], active[from, to]}, right{...} | Two options side by side (two types, before/after). `\n` breaks a bullet line. |
| `checklist` | at, to, title, items[[text, at]] | Steps or questions said in a row; each item ticks as it is said. |
| `veneer_front` | at, to, title | Drawn animation: veneers slide onto two incisors, the smile brightens. Needs about 2 s. |
| `veneer_side` | at, to, title | Drawn animation: a thin front layer marked in gold is removed and the veneer takes its place. Needs about 2 s. |
| `root_canal` | at, to, title, remove, clean, fill, keep | Drawn animation in 5 steps: inflamed nerve, the nerve is removed, a gold file cleans the canal, the canal is filled and the crown closed, the tooth stays. Anchor `remove`/`clean`/`fill`/`keep` to the words that describe each step; give the cleaning at least ~1.5 s. Needs about 5 s in total. |
| `comment` | at, to, text | CTA comment box that types the text and pulses the send button. Use it on the CTA, until `end`. |
| `brand` | gold, gold_hi, badge | Colours (RGB) and the default badge text. |
| `intro` | until, key, label, sub, kicker, steps | Gold explainer opener. `until` is about when the hook ends (seconds); `key` is the hook's keyword phrase. |

### Rules of the house style
- **Something new every 2-4 s** after the hook. Alternate a lockup, a card, an animation, a lockup… Never leave more than ~5 s with only captions.
- **One element at a time** in the card zone (y≈1100-1650). Lockups sit higher, on the chest, so a lockup may overlap a card's lifetime. Two cards may not.
- **Text only from the script.** Shorten it, never add facts, prices, durations, guarantees or medical claims the doctor did not say.
- **Lockups**: 1-2 words, caps, the word that carries the sentence. About one every 6-10 s.
- **Dental animations** go where the doctor describes the procedure. Veneers have `veneer_front`/`veneer_side`, root canal treatment (endodoncija) has `root_canal`. If a topic has no matching animation (implants, whitening, braces...), use a card with the `tooth` icon instead of inventing anatomy.
- **Hook**: intro until the hook ends, then badge + strip for 2-3 s.
- **CTA**: `comment` with the request in the viewer's words (e.g. "Šta te zanima o vinirima?"). Not a giant "COMMENT X" word, the clinic prefers it subtle.
- **Colours**: black cards, gold accents, white text. Do not bring back teal or red.

## Troubleshooting
- **A long pause between parts** (hook -> lead -> body): `python3 DG/engine/cut_gaps.py <out>/hook<N> --first 2` shortens the first two pauses of 0.9 s or more to 0.4 s (drop `--first` for all of them). It keeps clip, words, matte and face track in sync and keeps the originals in `_before_cut/`. Then render again.
- **Captions drift a little.** Check the prepare output for `WHISPER_WEAK` on that part (see step 1). Delete `<out>/whisper_*.json` before rerunning with another `--whisper-model` or `--lang` (default `hr`; `bs`/`sr` also work).
- **Anchor not found.** The phrase is not in `words.json` exactly. Check the spelling in `script.txt`.
- **Face badge/arrow off.** Edit `face.json` (`[centre x, centre y, width, height]` at 1080x1920).
