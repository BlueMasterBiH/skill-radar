---
description: Liquid glass: a refracting glass caption capsule, droplets that bloom into glass label cards, a lens that magnifies the speaker
argument-hint: [video path] [optional notes, e.g. title=..., keep audio]
---
Edit the talking-head clip in $ARGUMENTS (or the clip just dropped in the chat) in the **liquid-glass** style, using the arcads-cut skill.

Follow the skill's SKILL.md: set it up if this is the first run, analyse the clip, polish plan.json (beats, labels, sub lines, hot words), check the contact sheet, then render with `engine/run.py liquid-glass "<clip>"` and tell the user where the video is. Keep it silent unless they asked for audio.
