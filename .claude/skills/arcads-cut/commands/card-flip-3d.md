---
description: 3D card: the video floats on a glossy card in a studio, flies in and flips to show each beat label
argument-hint: [video path] [optional notes, e.g. title=..., keep audio]
---
Edit the talking-head clip in $ARGUMENTS (or the clip just dropped in the chat) in the **card-flip-3d** style, using the arcads-cut skill.

Follow the skill's SKILL.md: set it up if this is the first run, analyse the clip, polish plan.json (beats, labels, sub lines, hot words), check the contact sheet, then render with `engine/run.py card-flip-3d "<clip>"` and tell the user where the video is. Keep it silent unless they asked for audio.
