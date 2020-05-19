#!/bin/sh

set -euo pipefail

echo 'module "crosstool" [system] {'

for dir in $@; do
  find -L "${dir}" -type f | sort | uniq | while read header; do
    echo "  textual header \"${header}\""
  done
done

echo "}"
