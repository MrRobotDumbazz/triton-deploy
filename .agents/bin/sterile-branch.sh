#!/usr/bin/env bash
# sterile-branch.sh — стерильная ветка с сохранением истории (I53)
# GOV-файлы удаляются из всех коммитов ветки; продуктовые коммиты/авторы/последовательность
# сохраняются; чисто governance-коммиты отсекаются (prune-empty); msg-фильтр снимает
# маркеры ZTACP/инвариантов. Пуш — только человек (I48/I52).
set -euo pipefail
BRANCH=${1:-submit}; BASE=${2:-main}
ROOT=$(git rev-parse --show-toplevel)
TMP=$(mktemp -d /tmp/ztacp-sterile.XXXXXX); trap 'rm -rf "$TMP"' EXIT
git clone -q "$ROOT" "$TMP/clone"; cd "$TMP/clone"
git checkout -q -B "$BRANCH" "origin/$BASE"
GOV=".ztacp .agents .plan .spec AGENTS.md docs/ztacp-pattern.md docs/ztacp-pattern.en.md docs/ztacp-agents-school.md"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f --prune-empty \
  --index-filter "git rm -rq --cached --ignore-unmatch $GOV" \
  --msg-filter 'sed -E -e "s/[Zz][Tt][Aa][Cc][Pp]//g" -e "s/\((I[0-9]+(, ?I[0-9]+)*)\)//g" -e "s/\(6\.[0-9.]+\)//g" -e "s/(^|[^A-Za-z0-9])I[0-9]{1,2}([^0-9]|$)/\1\2/g" -e "s/  +/ /g"' \
  -- HEAD >/dev/null
HITS=$(git grep -Iil -e 'ZTACP' -e '\.ztacp' . 2>/dev/null || true)
[ -n "$HITS" ] && { echo "FAIL: $HITS"; exit 1; }
cd "$ROOT"; git fetch -q "$TMP/clone" "+refs/heads/$BRANCH:refs/heads/$BRANCH"
echo "OK: $BRANCH — $(git rev-list --count $BRANCH) commits, clean"
echo "Push manually: git push origin $BRANCH --force"
