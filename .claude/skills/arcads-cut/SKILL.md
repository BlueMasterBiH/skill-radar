---
name: arcads-cut
description: Edit one raw talking-head clip in 15 premium motion-design editing styles, one slash command each (/liquid-glass, /depth-title, /kinetic-captions, /cinematic-trailer, /neon-outline, /glitch-hud, /card-flip-3d, /infinite-zoom, /ui-depth-reel, /news-broadcast, /explainer-panel-reel, /swiss-grid, /scrapbook-collage, /cutout-grid, /ranking-format). Each style is a whole editing language (smooth eased camera with motion blur, liquid glass, 3D cards, kinetic type, light) timed to the spoken words, made only from the footage itself. Use when someone drops a talking-head video and names one of these styles, asks for a premium or "epic" edit of a clip, or wants the same clip edited several ways.
---

# arcads-cut: one raw clip, fifteen edits

The person is a creator, not an engineer. Keep replies short, do the work yourself, and never ask them to run commands.
All paths are relative to this skill's folder. Use its absolute path in commands. `PY` means `.venv/bin/python` inside this folder (`.venv\Scripts\python.exe` on Windows).

Every edit is made from the clip itself: reframes, punch-ins, alternate crops used as a second camera, freeze-frame layouts, captions, typography and drawn graphics. No stock images and no generated pictures get added.

## 0. First run (once per machine)
1. Run `bash scripts/setup.sh`. It prints `READY` when done (1 to 3 minutes the first time). It can also print one of these:
   - `NEED_FFMPEG`: install ffmpeg for them (`brew install ffmpeg` on Mac, `winget install ffmpeg` on Windows), then run setup again.
   - `NEED_PYTHON`: install Python 3.12, then run setup again.
2. When they ask to install the slash commands, run `bash scripts/install.sh`. It copies the skill to `~/.claude/skills/arcads-cut` and the commands to `~/.claude/commands`. Then tell them to restart Claude Code.
3. **Arcads connector** (tools named `arcads_*`) is optional for the editing itself. It helps in two places:
   - No camera? Generate the raw talking-head clip with Arcads from a script, then edit that.
   - If local transcription fails, `arcads_transcribe` gives the words (see Troubleshooting).

## 1. The workflow for every clip
The clip is whatever they dragged into the chat, or a path. One person facing the camera works best. Vertical phone footage with the head and shoulders in frame works best; landscape is cropped to 9:16.

### 1. Analyse and preview
Run `PY engine/run.py <style> "<clip>" --check`.

- The first time, this analyses the clip, about a minute for a short clip. It writes the following into `<clip folder>/<clip name>_arcads-cut/.cache/`:
  - a person matte
  - a face track
  - word timings
  - a clean wall plate
- It also writes a first `plan.json` (it prints `PLAN_CREATED`) and a contact sheet `<style>_check.jpg`.

### 2. Polish the plan
Edit `.cache/plan.json`. This is the editor's brain, so spend your effort here (see "The plan" below):
- Check the beats.
- Write short `label`s and `sub` lines.
- Set the `title` and `kicker`.
- Fill in `name`, `role` and `handle` if you know them.

### 3. Check
Run `PY engine/run.py <style> "<clip>" --check` again and look at the sheet yourself. Fix the plan, or pass `--opt`, when:
- a label is too long for its slot
- text sits on the eyes or mouth
- a beat lands on the wrong word

### 4. Render
- **One style:** `PY engine/run.py <style> "<clip>"` writes `<clip folder>/<clip name>_arcads-cut/<style>.mp4`.
  - Output is 1080x1920 at the clip's frame rate.
  - It is **silent by default**. Add `--audio` to keep their voice.
- **All styles:** `PY engine/run.py all "<clip>"` renders all 15, plus `00_lineup.mp4`, a 5-across grid of every style playing in sync.

### 5. Hand it over
Tell them where the file is, in one line. Remind them it's a draft they approve before posting.

## The plan (`.cache/plan.json`)
```json
{
 "title": "Stack shots in one prompt",
 "kicker": "PROMPT STACKING",
 "name": "Simone Ferretti", "role": "AI creator", "handle": "@sferro.ai",
 "hot": ["stack", "camera", "angle", "hard", "flash", "same", "prompt"],
 "beats": [
  {"t0": 0.0,  "t1": 1.38, "text": "And you can stack them", "key": "stack",        "key_t": 0.46, "label": "Stack",        "sub": "TWO LOOKS"},
  {"t0": 1.38, "t1": 2.6,  "text": "put a camera angle",     "key": "camera angle", "key_t": 1.8,  "label": "Camera angle", "sub": "LOW · 24MM"},
  {"t0": 2.6,  "t1": 3.84, "text": "and hard flash",         "key": "hard flash",   "key_t": 3.08, "label": "Hard flash",   "sub": "ON-CAMERA · DIRECT"},
  {"t0": 3.84, "t1": 5.28, "text": "in the same prompt",     "key": "same prompt",  "key_t": 4.22, "label": "One prompt",   "sub": "LOW ANGLE + HARD FLASH"}
 ]
}
```

### Beat times
- The auto plan splits the script at pauses and before words like "and", "in" and "but". It aims for about one beat per 1.3 s (3 to 6 beats), and picks a one- or two-word keyword per beat.
- `t0` and `t1` are the beat's span. Beats touch each other, and the last one runs to the end.
- `key` is the keyword and `key_t` is when it starts (from `.cache/words.json`). Styles land their main move on `key_t`, and a second move on the last keyword word: "flash" in "hard flash".

### What the styles expect from the beats
- **Beat 1 is the hook.** It gets the title move: masthead, slam, ransom letters, splash panel.
- **Middle beats are the ideas.** They get callouts, app-window notes, chips, tiers, evidence and panels.
- **The last beat is the payoff.** It gets the prompt box, stamp, S tier, headline or closing card.
- **How many beats:** four is the sweet spot. 3 to 6 work.

### Text fields
| Field | Length | Notes |
|---|---|---|
| `label` | 1–3 words | It appears in tiers, callouts, headlines, notes and bubbles. |
| `sub` | Under 22 characters | Caps, short. Shown on cards, slabs and lower thirds. |
| `title` | Up to 5 words | |
| `kicker` | 1–3 words | |

`hot` lists the words that get the accent colour in captions.

## The 15 styles
Every style opens with a big move in the first half-second and lands a distinct move on every keyword, so any 2-second slice reads as that style.

| Command | The edit | Signature moves |
|---|---|---|
| `/liquid-glass` | Liquid glass | A glass sphere melts into a refracting caption capsule; on each keyword a droplet splits off, rises and blooms into a glass label card; a glass lens sweeps over the speaker and magnifies him; light sweep on the payoff |
| `/depth-title` | Cinematic depth headline | Giant beat words slam in behind the speaker with 3D tilt and motion blur, parallax second type layer, slow push |
| `/kinetic-captions` | Premium kinetic captions | Words spring in in 3D with motion blur, the active word glows, keywords punch the camera with zoom blur and a chromatic hit |
| `/cinematic-trailer` | Movie trailer | Letterbox bars, teal-orange grade, anamorphic flares on the beats, black title card with tracking type, light leak, grain, gate weave |
| `/neon-outline` | Neon | A neon tube races around the speaker's outline, flickering neon words behind him, dark grade, bloom |
| `/glitch-hud` | Cyber HUD | Face-tracking brackets lock on, labels decode from random characters, scanlines, RGB-split slice glitches on the beats |
| `/card-flip-3d` | 3D card | The video floats on a glossy 3D card in a studio: flies in from deep space, flips to show each beat label, reflection, light sweep |
| `/infinite-zoom` | Infinite zoom | The frame holds a framed copy of itself, recursively; the camera dives seamlessly into the next frame on each beat |
| `/ui-depth-reel` | Glass app UI in 3D | Liquid-glass windows in depth around and behind the speaker: a prompt types, a cursor clicks Generate, a freeze-frame window flies at camera |
| `/news-broadcast` | Premium news package | Wireframe globe behind the speaker, swoosh-band reveal, BREAKING slab on a 3D hinge, glass lower third, ticker, live bug |
| `/explainer-panel-reel` | Split-screen explainer | The video springs into a card; glowing numbered nodes and animated connectors build a diagram above; progress rail |
| `/swiss-grid` | Swiss poster in motion | Grid lines draw in, the video cell reflows per beat with springs, giant grotesk words through mask reveals, split-flap numerals, red accent |
| `/scrapbook-collage` | Stop-motion zine | Torn-paper reveal, die-cut speaker with sticker outline, ransom letters flipping in, tape, 12 fps boil on graphics |
| `/cutout-grid` | Pop-art grid | The cut-out speaker multiplies into bold duotone tiles that flip in a wave and rearrange 1 → 4 → 9 → 1 per beat |
| `/ranking-format` | Countdown ranking | Giant 3D-extruded numerals swing in on a hinge, labels fly into a glass tier board, gold sweep, confetti on the payoff |

`PY engine/run.py list` prints the same list from the code.

## Options (`--opt key=value`)
Every style reads `title`, `kicker`, `name`, `role` and `handle` (they override plan.json).

## Rules
- **Each command is one editing language.** Don't mix styles in one render. Every beat gets its own move, so the style reads in a few seconds.
- **Edit only with the footage, type and drawn graphics.** Don't add stock or generated images.
- **Never cover the eyes or mouth.** If a check frame shows a clash, shorten the label or adjust the beat.
- **Keep text phone-legible**: at least 22 px at 1080 wide for labels.
- **Silent unless `--audio` is passed.** Sound design is a separate pass.
- **These are drafts.** Be honest about that if they compare them with a hand edit.

## Troubleshooting
- **Graphics misplaced.** The clip needs one person facing the camera most of the time.
- **Framing.** Vertical phone video works best: the styles show the whole upper body and keep the camera wide (zoom 1.15 at most). Landscape clips are cropped to 9:16 around the face.
- **Text too long.** Shorten `label`/`sub`/`title`.
- **`NEED_TRANSCRIBER`.** Run setup again. Or:
  - call `arcads_transcribe` with `format: "srt"`
  - convert the cues into `.cache/words.json` as `[{"w": word, "s": start, "e": end}]`, spreading each cue's words evenly over its time
  - run again
- **Slow renders.** About 1 to 2 minutes per style for a 5-second 4K clip. `--check` takes seconds.
- **Credits and licences.**
  - Fonts are SIL OFL / Apache, in `fonts/`.
  - Face detection uses OpenCV YuNet (MIT); segmentation uses MediaPipe Selfie Segmenter (Apache 2.0). Notices are in `engine/NOTICE.md`.
