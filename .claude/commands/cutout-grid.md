---
description: Pop-art grid: the cut-out speaker multiplies into bold colour tiles that flip and rearrange per beat
argument-hint: [video path] [optional notes, e.g. title=..., keep audio]
---
Edit the talking-head clip in $ARGUMENTS (or the clip just dropped in the chat) in the **cutout-grid** style, using the arcads-cut skill.

Follow the skill's SKILL.md: set it up if this is the first run, analyse the clip, polish plan.json (beats, labels, sub lines, hot words), check the contact sheet, then render with `engine/run.py cutout-grid "<clip>"` and tell the user where the video is. Keep it silent unless they asked for audio.
