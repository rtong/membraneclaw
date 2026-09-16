# Fast-forward the worktree, clearing untracked files the pull would overwrite
# ONLY where they are byte-identical to what is being pulled. Artefacts get
# rsynced back and committed from the laptop, so the same bytes arrive here as
# untracked files and block the next pull; anything that actually differs is
# left alone and reported, because that would be an artefact this box has and
# the commit does not.
set -euo pipefail
cd ~/grpo-seed
git fetch -q origin
BR=$(git rev-parse --abbrev-ref --symbolic-full-name @{u})
kept=0
for f in $(git diff --name-only HEAD "$BR"); do
  if [ -f "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    a=$(sha256sum "$f" | cut -d" " -f1)
    b=$(git show "$BR:$f" | sha256sum | cut -d" " -f1)
    if [ "$a" = "$b" ]; then rm "$f"; else echo "DIFFERS, kept: $f"; kept=$((kept+1)); fi
  fi
done
[ "$kept" -eq 0 ] || { echo "refusing to pull: $kept file(s) differ"; exit 1; }
git pull -q --ff-only
git log --oneline -1
