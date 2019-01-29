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


# Test that the entries single jar creates can be extracted (that is, they do
# not have some funny Unix access more settings making them unreadable).
function test_new_entries() {
  local -r out_jar="${TEST_TMPDIR}/out.jar"
  "$singlejar" --output "$out_jar"
  cd "${TEST_TMPDIR}"
  unzip "$out_jar" build-data.properties
  [[ -r build-data.properties ]] || \
    { echo "build-data.properties is not readable" >&2; exit 1; }
}

# Regression test https://github.com/bazelbuild/bazel/issues/6820
function test_empty_resource_file() {
  local -r out_jar="${TEST_TMPDIR}/out.jar"
  local -r empty="${TEST_TMPDIR}/empty"
  echo -n > "$empty"
  "$singlejar" --output "$out_jar" --resources $empty
}

run_suite "Misc shell tests"
#!/bin/bash

