#!/bin/bash
#
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
#
# nonincremental_builds_test.sh: tests for the --keep_state_after_build flag.

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

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2018-08-14, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

if "$is_windows"; then
  EXE_EXT=".exe"
else
  EXE_EXT=""
fi

javabase="$1"
if [[ $javabase = external/* ]]; then
  javabase=${javabase#external/}
fi
jmaptool="$(rlocation "${javabase}/bin/jmap${EXE_EXT}")"

if ! type try_with_timeout >&/dev/null; then
  # Bazel's testenv.sh defines try_with_timeout but the Google-internal version
  # uses a different testenv.sh.
  function try_with_timeout() { $* ; }
fi

function tear_down() {
  try_with_timeout bazel shutdown || fail "Failed to shut down bazel"
}

#### TESTS #############################################################
function create_minimal_target() {
    local -r pkg=$1
    mkdir $pkg
    cat > $pkg/BUILD <<EOF || fail "Couldn't make BUILD file"
genrule(
    name = 'top',
    outs = ['final.out'],
    local = 1,
    cmd = 'touch \$@'
)
EOF
    INCREMENTAL_ANALYSIS_LOGLINE="Analy[sz]ed target //$pkg:top (0 packages loaded)"
    NONINCREMENTAL_ANALYSIS_LOGLINE="Analy[sz]ed target //$pkg:top ([1-9][0-9]* packages loaded)"
}

# Test that the execution is not repeated, test to validate the test case
# for the nonincremental test below.
function test_build_is_incremental_with_keep_state() {
    local -r pkg=$FUNCNAME
    create_minimal_target $pkg
    bazel build $pkg:top &> "$TEST_log"  \
        || fail "Couldn't build $pkg"
    expect_log_once $NONINCREMENTAL_ANALYSIS_LOGLINE \
        "First build expected to execute the target."

    bazel build $pkg:top &> "$TEST_log"  \
        || fail "Couldn't build $pkg"
    expect_log_once $INCREMENTAL_ANALYSIS_LOGLINE \
        "Second build not expected to reexecute."
}

# Test that the execution is actually repeated, indirect test that the state
# was not reused.
function test_build_is_nonincremental_with_nokeep_state() {
    local -r pkg=$FUNCNAME
    create_minimal_target $pkg
    bazel build --nokeep_state_after_build $pkg:top &> "$TEST_log"  \
        || fail "Couldn't build $pkg"
    expect_log_once $NONINCREMENTAL_ANALYSIS_LOGLINE \
        "First build expected to execute the target."

    bazel build $pkg:top &> "$TEST_log"  \
        || fail "Couldn't build $pkg"
    expect_log_once $NONINCREMENTAL_ANALYSIS_LOGLINE \
        "Second build should not use the cached state."
}

# Test directly that the inmemory state does persist after the build by default.
function test_inmemory_state_present_after_build() {
    local -r pkg=$FUNCNAME
    create_minimal_target $pkg
    bazel build $pkg:top &> "$TEST_log"  \
        || fail "Couldn't build $pkg"
    local server_pid="$(bazel info server_pid 2>> "$TEST_log")"
    "$jmaptool" -histo:live "$server_pid" > histo.txt

    cat histo.txt >> "$TEST_log"
    assert_contains "GenRuleAction" histo.txt
    assert_contains "InMemoryNodeEntry" histo.txt
}

# Test directly that the inmemory state does not persist after the build.
function test_inmemory_state_absent_after_build_with_nokeep_state() {
    local -r pkg=$FUNCNAME
    create_minimal_target $pkg
    bazel build --nokeep_state_after_build $pkg:top &> "$TEST_log"  \
        || fail "Couldn't build $pkg"
    local server_pid="$(bazel info server_pid 2>> "$TEST_log")"
    "$jmaptool" -histo:live "$server_pid" > histo.txt

    cat histo.txt >> "$TEST_log"
    assert_not_contains "GenRuleAction$" histo.txt
    assert_not_contains "InMemoryNodeEntry" histo.txt
}

run_suite "test for --keep_state_after_build"
