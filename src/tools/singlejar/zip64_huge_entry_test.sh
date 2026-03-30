#!/usr/bin/env bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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
# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

# Load test environment
source "$(rlocation "io_bazel/src/test/shell/unittest.bash")" \
  || { echo "unittest.bash not found!" >&2; exit 1; }

IS_WINDOWS=false
case "$(uname | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  IS_WINDOWS=true
esac

if "$IS_WINDOWS"; then
  EXE_EXT=".exe"
else
  EXE_EXT=""
fi

singlejar="$(rlocation "io_bazel/src/tools/singlejar/singlejar${EXE_EXT}")"
javabase="$1"
if [[ $javabase = external/* ]]; then
  javabase=${javabase#external/}
fi
jartool="$(rlocation "${javabase}/bin/jar${EXE_EXT}")"

# Test that a huge zip entry can be successfully added to Zip64 archive
function test_hugezipentry() {
  local -r top="$TEST_TMPDIR/hugezipentry"
  date
  mkdir -p "$top"
  # Create a single 4GB file
  dd if=/dev/zero of="$top/file"  bs=4M  count=1024
  local -r inzip="$TEST_TMPDIR/inhugezipentry.zip"
  local -r outzip="$TEST_TMPDIR/outhugezipentry.zip"
  rm -f "$inzip" "$outzip"
  "$jartool" -cf "$inzip" -C "$top" "file"
  rm -rf "$top"
  "$singlejar" --output "$outzip" --sources "$inzip"
  # Verify jar can read it, and the filesize is correct.
  expected_size=$((4*1024*1024*1024)) # 4 GB
  local actual_size=$("$jartool" tvf "$outzip" "file" | awk '{print $1}')
  [[ "$actual_size" -eq "$expected_size" ]] || \
    { echo "Expected size $expected_size, got $actual_size" >&2; exit 1; }
}

run_suite "singlejar Zip64 handling"
