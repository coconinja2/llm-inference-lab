#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

patterns='([[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,})|([0-9]{3}[-. ][0-9]{3}[-. ][0-9]{4})|(AKIA[0-9A-Z]{16})|(-----BEGIN [A-Z ]*PRIVATE KEY-----)|(/Users/[^/ ]+)|(C:\\Users\\[^\\ ]+)'

if rg -n --hidden --glob '!.git/**' --glob '!build/**' \
  --glob '!scripts/privacy-audit.sh' "$patterns" .; then
  echo "privacy audit failed: review the matches above" >&2
  exit 1
fi

echo "privacy audit passed"
