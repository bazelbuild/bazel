#!/bin/bash
# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
declare -r OUT="$1"
declare -r INPUT="$2"
declare -r CONTENT="$3"
declare -r REGEXP="$4"
declare -r MATCHES="$5"

if [[ ( -n "${CONTENT:-}" && -n "${REGEXP:-}" ) || \
      ( -z "${CONTENT:-}" && -z "${REGEXP:-}" ) ]]; then
  echo >&2 "ERROR: expected either 'content' or 'regexp'"
  exit 1
elif [[ -n "${CONTENT:-}" && \
        ( -n "${MATCHES:-}" && "$MATCHES" != "-1" ) ]]; then
  echo >&2 "ERROR: cannot specify 'matches' together with 'content'"
  exit 1
elif [[ ! ( -z "${MATCHES:-}" || "$MATCHES" = 0 || \
            "$MATCHES" =~ ^-?[1-9][0-9]*$ ) ]]; then
  echo >&2 "ERROR: 'matches' must be an integer"
  exit 1
elif [[ ! -e "${INPUT:-/dev/null/does-not-exist}" ]]; then
  echo >&2 "ERROR: input file must exist"
  exit 1
else
  if [[ -n "${CONTENT:-}" ]]; then
    declare -r GOLDEN_FILE="$(mktemp)"
    declare -r ACTUAL_FILE="$(mktemp)"
    echo -e -n "$CONTENT" | sed 's,\r\n,\n,g' > "$GOLDEN_FILE"
    sed 's,\r\n,\n,g' "$INPUT" > "$ACTUAL_FILE"
    if ! diff -u "$GOLDEN_FILE" "$ACTUAL_FILE" ; then
      echo >&2 "DEBUG[cica] ----"
      hexdump -C $ACTUAL_FILE | sed 's,^, (,;s,$,),' >&2
      echo >&2 "DEBUG[cica] ----"
      hexdump -C $GOLDEN_FILE | sed 's,^, (,;s,$,),' >&2
      echo >&2 "DEBUG[cica] ----"
      echo >&2 "ERROR: file did not have expected content"
      exit 1
    fi
  else
    if [[ -n "${MATCHES:-}" && $MATCHES -gt -1 ]]; then
      if [[ "$MATCHES" != $(grep -c "$REGEXP" "$INPUT") ]]; then
        echo >&2 "ERROR: file did not contain expected regexp $MATCHES times"
        exit 1
      fi
    else
      if ! grep "$REGEXP" "$INPUT"; then
        echo >&2 "ERROR: file did not contain expected regexp"
        exit 1
      fi
    fi
  fi
  date +"%%s.%%N" > "$OUT"
fi
