#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
#
#
# Arguments:
#   unittest.bash script
#   singlejar path
#   jar tool path

(($# >= 3)) || \
  { echo "Usage: $0 <unittest.bash dir> <singlejar> <jartool>" >&2; exit 1; }

# Load test environment
source $1/unittest.bash \
  || { echo "unittest.bash not found!" >&2; exit 1; }

set -e
declare -r singlejar="$2"
declare -r jartool="$3"


# Test that an archive with >64K entries can be created.
function test_65Kentries() {
  local -r top="$TEST_TMPDIR/65Kentries"
  date
  mkdir -p "$top"
  dd if=/dev/zero of="$top/file" bs=256 count=1
  for dir in {1..256}; do
    # Create 256 tiny files in $dirpath
    local dirpath="$top/dir$dir"
    mkdir -p "$dirpath"
    split -b 1 "$top/file" "$dirpath/x."
  done
  # Now we have 256 directories with 256 files in each. Zipping them together
  # yields an archive with >64K entries.
  local -r inzip="$TEST_TMPDIR/in65K.zip"
  local -r outzip="$TEST_TMPDIR/out65K.zip"
  rm -f "$inzip" "$outzip"
  "$jartool" -cf "$inzip" "$top"

  "$singlejar" --output "$outzip" --sources "$inzip"
  # Verify jar can read it.
  local -ir n_entries=$("$jartool" -tf "$outzip" | wc -l)
  ((${n_entries:-0} > 65536)) || \
    { echo Expected 65536 entries, got "$n_entries" >&2; exit 1; }
}

run_suite "singlejar Zip64 handling"
