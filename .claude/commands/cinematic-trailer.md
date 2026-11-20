---
description: Movie trailer: letterbox bars, teal-orange grade, anamorphic flares, black title cards with tracking type
argument-hint: [video path] [optional notes, e.g. title=..., keep audio]
---
Edit the talking-head clip in $ARGUMENTS (or the clip just dropped in the chat) in the **cinematic-trailer** style, using the arcads-cut skill.

Follow the skill's SKILL.md: set it up if this is the first run, analyse the clip, polish plan.json (beats, labels, sub lines, hot words), check the contact sheet, then render with `engine/run.py cinematic-trailer "<clip>"` and tell the user where the video is. Keep it silent unless they asked for audio.
