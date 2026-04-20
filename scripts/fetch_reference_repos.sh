#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$ROOT/references/vendor"
mkdir -p "$VENDOR"

clone_or_update() {
  local name="$1"
  local url="$2"
  local target="$VENDOR/$name"
  if [[ -d "$target/.git" ]]; then
    git -C "$target" fetch --depth 1 origin
    git -C "$target" checkout -q FETCH_HEAD
  else
    git clone --depth 1 "$url" "$target"
  fi
  git -C "$target" rev-parse HEAD
}

clone_or_update rank_bm25 https://github.com/dorianbrown/rank_bm25.git
clone_or_update mem0 https://github.com/mem0ai/mem0.git
clone_or_update MemGPT https://github.com/cpacker/MemGPT.git
clone_or_update MemoryBank-SiliconFriend https://github.com/zhongwanjun/MemoryBank-SiliconFriend.git
