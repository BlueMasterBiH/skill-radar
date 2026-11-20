#!/usr/bin/env bash
# Install dental-gold into Claude Code: the skill into ~/.claude/skills, the /dental-gold command into ~/.claude/commands.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CLAUDE_HOME:-$HOME/.claude}"
mkdir -p "$DEST/skills" "$DEST/commands"
if [ "$SRC" != "$DEST/skills/dental-gold" ]; then
  rm -rf "$DEST/skills/dental-gold"
  cp -R "$SRC" "$DEST/skills/dental-gold"
  rm -rf "$DEST/skills/dental-gold/.venv"
fi
cp "$SRC"/commands/*.md "$DEST/commands/"
echo "INSTALLED skill: $DEST/skills/dental-gold   command: /dental-gold"
echo "For the gold intro also install the arcads-cut skill."
