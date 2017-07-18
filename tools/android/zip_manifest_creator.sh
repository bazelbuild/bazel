#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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

# This script takes in a regular expression and a zip file and writes a file
# containing the names of all files in the zip file that match the regular
# expression with one per line. Names of directories are not included.

if [ "$#" -ne 3 ]; then
  echo "Usage: zip_manifest_creator.sh <regexp> <input zip> <output manifest>"
  exit 1
fi

REGEX="$1"
INPUT_ZIP="$2"
OUTPUT_MANIFEST="$3"

RUNFILES="${RUNFILES:-$0.runfiles}"
RUNFILES_MANIFEST_FILE="${RUNFILES_MANIFEST_FILE:-$RUNFILES/MANIFEST}"

IS_WINDOWS=false
case "$(uname | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  IS_WINDOWS=true
esac

if "$IS_WINDOWS" && ! type rlocation &> /dev/null; then
  function rlocation() {
    # Use 'sed' instead of 'awk', so if the absolute path ($2) has spaces, it
    # will be printed completely.
    local result="$(grep "$1" "${RUNFILES_MANIFEST_FILE}" | head -1)"
    # If the entry has a space, it is a mapping from a runfiles-path to absolute
    # path, otherwise it resolves to itself.
    echo "$result" | grep -q " " \
        && echo "$result" | sed 's/^[^ ]* //' \
        || echo "$result"
  }
fi

# For @bazel_tools//tools/android:zip_manifest_creator in BUILD.tools, zipper is here:
#   Windows (in MANIFEST):  <repository_name>/tools/zip/zipper/zipper.exe
#   Linux/MacOS (symlink):  ${RUNFILES}/<repository_name>/tools/zip/zipper/zipper
if "$IS_WINDOWS"; then
  ZIPPER="$(rlocation "[^/]*/tools/zip/zipper/zipper.exe")"
else
  ZIPPER="$(find "$RUNFILES" -path "*/tools/zip/zipper/zipper" | head -1)"
fi
if [ ! -x "$ZIPPER" ]; then
  # For //tools/android:zip_manifest_creator_test, zipper is here:
  #   Windows (in MANIFEST):  <workspace_name>/third_party/ijar/zipper.exe
  #   Linux/MacOS (symlink):  ${RUNFILES}/<workspace_name>/third_party/ijar/zipper
  if "$IS_WINDOWS"; then
    ZIPPER="$(rlocation "[^/]*/third_party/ijar/zipper.exe")"
  else
    ZIPPER="$(find "${RUNFILES}" -path "*/third_party/ijar/zipper" | head -1)"
  fi
fi
if [ ! -x "$ZIPPER" ]; then
  echo >&2 "ERROR: $(basename $0): could not find zipper executable. Additional info:"
  echo >&2 "  \$0=($0)"
  echo >&2 "  RUNFILES=($RUNFILES)"
  echo >&2 "  RUNFILES_MANIFEST_FILE=($RUNFILES_MANIFEST_FILE)"
  echo >&2 "  IS_WINDOWS=($IS_WINDOWS)"
  if "$IS_WINDOWS"; then
    echo >&2 "  grep=($(grep zipper "$RUNFILES_MANIFEST_FILE"))"
  else
    echo >&2 "  find=($(find "$RUNFILES" -name "zipper" | head -1))"
  fi
  exit 1
fi

"$ZIPPER" v "$INPUT_ZIP" \
  | cut -d ' ' -f3 \
  | grep -v \/$ \
  | grep -x "$REGEX" \
  > "$OUTPUT_MANIFEST"
exit 0
