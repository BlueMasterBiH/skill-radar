# dental-gold

The clinic's black-and-gold reel template, packaged as a Claude Code skill.

**Input:** one raw recording (all hooks back to back, then lead, body and CTA) and its script.
**Output:** one finished 1080x1920 video per hook, with:
- a gold explainer intro and a "Doktorica" badge
- small captions
- gold key words
- black cards, a comparison card and a checklist
- drawn veneer animations
- a comment-box CTA

## Install
1. Unzip, drag the `dental-gold` folder into Claude Code and say **"install the dental-gold skill"**. Claude runs `scripts/install.sh`.
2. Caption timing uses Whisper (faster-whisper), which runs locally with no API key. `scripts/setup.sh` installs it, and the model downloads on first use.
3. For the gold intro, the **arcads-cut** skill must be installed as well. Its Python environment is reused.
4. Restart Claude Code.

## Use
Drop in the clip and the script, then type `/dental-gold`. Add "samo hook 3" to edit one version only.

## How it works
| Part | File |
|---|---|
| Cut the parts and time the script words with Whisper (local, free) | `engine/prepare.py` |
| What appears on screen and when, anchored to script phrases | `storyboard.json` |
| Gold opener (uses arcads-cut) | `engine/make_intro.py` |
| Draw everything and encode | `engine/render.py` |
| Element library (cards, lockups, dental animations) | `engine/elements.py` |

`examples/` holds the veneers video script and storyboard, the reference edit.

The font is Montserrat (SIL OFL, `fonts/`). All graphics are drawn in code; no stock images.
