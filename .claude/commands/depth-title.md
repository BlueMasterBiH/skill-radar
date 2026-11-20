---
description: Cinematic depth headline: giant beat words slam in behind the speaker in 3D, parallax type, slow push
argument-hint: [video path] [optional notes, e.g. title=..., keep audio]
---
Edit the talking-head clip in $ARGUMENTS (or the clip just dropped in the chat) in the **depth-title** style, using the arcads-cut skill.

Follow the skill's SKILL.md: set it up if this is the first run, analyse the clip, polish plan.json (beats, labels, sub lines, hot words), check the contact sheet, then render with `engine/run.py depth-title "<clip>"` and tell the user where the video is. Keep it silent unless they asked for audio.
