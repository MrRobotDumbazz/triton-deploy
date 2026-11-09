#!/usr/bin/env bash
# sterile-branch.sh — стерильная ветка с сохранением истории (ZTACP I53)
set -euo pipefail
BRANCH=${1:-submit}; BASE=${2:-main}
ROOT=$(git rev-parse --show-toplevel)
TMP=$(mktemp -d /tmp/ztacp-sterile.XXXXXX); trap 'rm -rf "$TMP"' EXIT
git clone -q "$ROOT" "$TMP/clone"; cd "$TMP/clone"
git checkout -q -B "$BRANCH" "origin/$BASE"
GOV=".ztacp .agents .plan .spec AGENTS.md docs/ztacp-pattern.md docs/ztacp-pattern.en.md"
MSG='sed -E "s/[Zz][Tt][Aa][Cc][Pp]//g; s/\((I[0-9]+(, ?I[0-9]+)*)\)//g; s/\(6\.[0-9.]+\)//g; s/\bI[0-9]{1,2}\b//g; s/  +/ /g"'
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --prune-empty \
  --index-filter "git rm -rq --cached --ignore-unmatch $GOV" \
  --msg-filter "eval $MSG" -- HEAD >/dev/null
HITS=$(git grep -Iil -e 'ZTACP' -e '\.ztacp' . 2>/dev/null || true)
[ -n "$HITS" ] && { echo "FAIL: $HITS"; exit 1; }
cd "$ROOT"; git fetch -q "$TMP/clone" "+refs/heads/$BRANCH:refs/heads/$BRANCH"
echo "OK: $BRANCH — $(git rev-list --count $BRANCH) commits, clean"
echo "Push manually: git push origin $BRANCH --force"
