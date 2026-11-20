---
description: Floating liquid-glass app windows in 3D depth around the speaker: a prompt types, a cursor clicks Generate
argument-hint: [video path] [optional notes, e.g. title=..., keep audio]
---
Edit the talking-head clip in $ARGUMENTS (or the clip just dropped in the chat) in the **ui-depth-reel** style, using the arcads-cut skill.

Follow the skill's SKILL.md: set it up if this is the first run, analyse the clip, polish plan.json (beats, labels, sub lines, hot words), check the contact sheet, then render with `engine/run.py ui-depth-reel "<clip>"` and tell the user where the video is. Keep it silent unless they asked for audio.
