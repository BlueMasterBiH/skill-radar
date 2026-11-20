---
description: Swiss poster in motion: grid lines draw in, the video cell reflows per beat, giant grotesk words, red accent
argument-hint: [video path] [optional notes, e.g. title=..., keep audio]
---
Edit the talking-head clip in $ARGUMENTS (or the clip just dropped in the chat) in the **swiss-grid** style, using the arcads-cut skill.

Follow the skill's SKILL.md: set it up if this is the first run, analyse the clip, polish plan.json (beats, labels, sub lines, hot words), check the contact sheet, then render with `engine/run.py swiss-grid "<clip>"` and tell the user where the video is. Keep it silent unless they asked for audio.
