#!/usr/bin/env bash
# check-tree.sh — машинная проверка паритета дерева (I51): REQUIRED присутствует, extras санкционированы.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
fail=0
while read -r p; do
  [ -z "$p" ] && continue
  case "$p" in \#*) continue;; esac
  if [ ! -e "$p" ]; then echo "MISSING: $p"; fail=1; fi
done < <(grep -A100 '^## REQUIRED' .ztacp/reference-tree.txt | grep -v '^## ' | grep -v '^#' | sed 's/#.*//' | awk '{$1=$1};1' | grep -v '^$')
# extras: всё закоммиченное в продуктовых каталогах обязано быть в REQUIRED или SANCTIONED
SANCTIONED=$(cat .ztacp/reference-tree.txt | sed -n '/## SANCTIONED/,/## RUNTIME/p')
while read -r p; do
  [ -z "$p" ] && continue
  case "$p" in models/*|deployment/*|monitoring/*|client/*) ;; *) continue;; esac
  case "$p" in client/samples/img_*.jpg) continue;; esac
  if ! grep -qF -- "$p" <<<"$SANCTIONED"; then echo "UNSANCTIONED EXTRA: $p"; fail=1; fi
done < <(git ls-files models deployment monitoring client)
[ "$fail" -eq 0 ] && echo "TREE PARITY: OK" || { echo "TREE PARITY: FAIL (Дрейф П3 / перестройка — override P-1, I6)"; exit 1; }
