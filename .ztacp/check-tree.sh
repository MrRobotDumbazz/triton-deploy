#!/usr/bin/env bash
# check-tree.sh — машинная проверка паритета дерева (I51): REQUIRED присутствует, extras санкционированы.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
REF=.ztacp/reference-tree.txt

section() { sed -n "/^## $1/,/^## /p" "$REF" | grep -v '^## ' | grep -v '^#' | sed 's/#.*//' | awk '{$1=$1};1' | grep -v '^$'; }
REQUIRED=$(section REQUIRED)
SANCTIONED=$(section SANCTIONED)
ALLOWED=$(printf '%s\n%s\n' "$REQUIRED" "$SANCTIONED")

fail=0
# 1) все REQUIRED существуют в рабочем дереве
while IFS= read -r p; do
  [ -e "$p" ] || { echo "MISSING: $p"; fail=1; }
done <<<"$REQUIRED"

# 2) всё закоммиченное в продуктовых каталогах — из REQUIRED/SANCTIONED (или glob-исключение samples)
while IFS= read -r p; do
  case "$p" in client/samples/img_*.jpg) continue;; deployment/grafana/*) continue;; esac
  grep -qxF -- "$p" <<<"$ALLOWED" || { echo "UNSANCTIONED EXTRA: $p"; fail=1; }
done < <(git ls-files models deployment monitoring client)

if [ "$fail" -eq 0 ]; then echo "TREE PARITY: OK"; else echo "TREE PARITY: FAIL (Дрейф П3 / перестройка — override P-1, I6)"; exit 1; fi
