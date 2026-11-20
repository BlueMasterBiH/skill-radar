#!/usr/bin/env bash
# Install arcads-cut into Claude Code: the skill into ~/.claude/skills, the slash commands into ~/.claude/commands.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CLAUDE_HOME:-$HOME/.claude}"
mkdir -p "$DEST/skills" "$DEST/commands"
if [ "$SRC" != "$DEST/skills/arcads-cut" ]; then
  rm -rf "$DEST/skills/arcads-cut"
  cp -R "$SRC" "$DEST/skills/arcads-cut"
  rm -rf "$DEST/skills/arcads-cut/.venv"
fi
cp "$SRC"/commands/*.md "$DEST/commands/"
echo "INSTALLED skill: $DEST/skills/arcads-cut   commands: $(ls "$SRC"/commands | wc -l | tr -d ' ') in $DEST/commands"
echo "Restart Claude Code, connect the Arcads connector, then drop a clip in and type a command, e.g. /podcast-clip"
