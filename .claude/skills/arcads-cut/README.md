# arcads-cut

One raw talking-head clip in, 15 finished edit styles out. Each style is a slash command in Claude Code. Every edit is built from your footage, captions, typography and graphics only: no stock images.

## Install (2 minutes)
1. Put this `arcads-cut` folder anywhere on your computer.
2. Open Claude Code. Drag the folder into the chat and say **"install the arcads-cut skill"**. Claude runs `scripts/install.sh` and `scripts/setup.sh` for you. Or run them yourself:
   ```bash
   bash scripts/install.sh
   ```
3. Restart Claude Code. Optional: connect the **Arcads** connector in Claude's settings to make a talking-head clip from a script when you don't want to film.

## Use
Drop a talking-head clip into the chat and type a command:

| Command | Style |
|---|---|
| `/liquid-glass` | Refracting liquid glass captions, cards and lens |
| `/depth-title` | Giant 3D words behind the speaker |
| `/kinetic-captions` | Premium 3D word-by-word captions |
| `/cinematic-trailer` | Movie trailer with flares and title cards |
| `/neon-outline` | Neon tube tracing the speaker |
| `/glitch-hud` | Cyber HUD with glitches and decoding labels |
| `/card-flip-3d` | Video on a flipping 3D card |
| `/infinite-zoom` | Recursive frame-in-frame dive |
| `/ui-depth-reel` | Glass app windows in 3D depth |
| `/news-broadcast` | Premium 3D news package |
| `/explainer-panel-reel` | Split-screen diagram explainer |
| `/swiss-grid` | Swiss poster in motion |
| `/scrapbook-collage` | Stop-motion paper zine |
| `/cutout-grid` | Pop-art cut-out tiles |
| `/ranking-format` | 3D countdown ranking |

`/arcads-cut <clip> all` renders every style plus a side-by-side lineup.

Videos land next to your clip in `<clip name>_arcads-cut/`. They're 1080x1920 and silent by default. Say "keep the audio" to include your voice.

## What it needs
- ffmpeg and Python 3.10–3.13. Setup installs the rest into a private `.venv`: numpy, OpenCV, Pillow, MediaPipe and faster-whisper.
- Works best with one person facing the camera, 4K landscape or vertical, under 2 minutes.

## Credits
- Engine and styles are original. They were inspired by talking-head editing workflows such as claude-video-effects (MIT).
- Fonts are SIL OFL / Apache. Face detection uses OpenCV YuNet (MIT), and segmentation uses MediaPipe (Apache 2.0).
