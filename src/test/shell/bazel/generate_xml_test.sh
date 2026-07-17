#!/usr/bin/env bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
# Tests the UTF-8 fixing script in test-setup.sh / generate-xml.sh.
#

# Bootstrap runfiles lookup. We can't use a central script for that because we'd
# need to be able to lookup runfiles in order to use it.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
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

# Load the unit test framework
source "$(rlocation io_bazel/src/test/shell/unittest.bash)" \
  || (echo "unittest.bash not found!" && exit 1)
GENERATE_XML="$(rlocation io_bazel/tools/test/generate-xml.sh)"

# Encode the passed parameters using the encode_utf8 routing in generate-xml.sh.
function encode {
  echo -e "$@" | "$GENERATE_XML" "-" "-" "-" "-"
}

function test_simple_ascii() {
  assert_equals 'Simple ascii' "$(encode "Simple ascii")"
}

function test_low_control_chars() {
  # Need echo to turn \t into a tab.
  assert_equals "$(echo -e '????????\t?')" \
      "$(encode '\x1\x2\x3\x4\x5\x6\x7\x8\x9\xb')"
}

function test_high_control_chars() {
  assert_equals '?????????????' \
      "$(encode '\xc\xe\xf\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19')"
}

function test_valid_two_byte_seq() {
  assert_equals "$(echo -e '\xc0\x80')" "$(encode '\xc0\x80')"
}

function test_valid_three_byte_seq() {
  assert_equals "$(echo -e '\xea\xa0\xb0')" "$(encode '\xea\xa0\xb0')"
}

function test_invalid_two_byte_seq() {
  assert_equals '??' "$(encode '\xc0\xc0')"
}

run_suite "generate-xml.sh tests"
