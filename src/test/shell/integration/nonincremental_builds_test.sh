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
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/discard_graph_edges_lib.sh" \
  || { echo "${CURRENT_DIR}/discard_graph_edges_lib.sh not found!" >&2; exit 1; }

#### SETUP #############################################################
set -e

function tear_down() {
    bazel shutdown || fail "Failed to shut down bazel"
}

#### TESTS #############################################################
function create_minimal_target() {
    rm -rf simpletarget
    mkdir simpletarget
    cat > simpletarget/BUILD <<EOF || fail "Couldn't make BUILD file"
genrule(
    name = 'top',
    outs = ['final.out'],
    local = 1,
    cmd = 'touch \$@'
)
EOF
    INCREMENTAL_ANALYSIS_LOGLINE="Analysed target //simpletarget:top (0 packages loaded)"
    NONINCREMENTAL_ANALYSIS_LOGLINE="Analysed target //simpletarget:top ([1-9][0-9]* packages loaded)"
}

# Test that the execution is not repeated, test to validate the test case
# for the nonincremental test below.
function test_build_is_incremental_with_keep_state() {
    create_minimal_target
    bazel build simpletarget:top &> "$TEST_log"  \
        || fail "Couldn't build simpletarget"
    expect_log_once $NONINCREMENTAL_ANALYSIS_LOGLINE \
        "First build expected to execute the target."

    bazel build simpletarget:top &> "$TEST_log"  \
        || fail "Couldn't build simpletarget"
    expect_log_once $INCREMENTAL_ANALYSIS_LOGLINE \
        "Second build not expected to reexecute."
}

# Test that the execution is actually repeated, indirect test that the state
# was not reused.
function test_build_is_nonincremental_with_nokeep_state() {
    create_minimal_target
    bazel build --nokeep_state_after_build simpletarget:top &> "$TEST_log"  \
        || fail "Couldn't build simpletarget"
    expect_log_once $NONINCREMENTAL_ANALYSIS_LOGLINE \
        "First build expected to execute the target."

    bazel build simpletarget:top &> "$TEST_log"  \
        || fail "Couldn't build simpletarget"
    expect_log_once $NONINCREMENTAL_ANALYSIS_LOGLINE \
        "Second build should not use the cached state."
}

# Test directly that the inmemory state does persist after the build by default.
function test_inmemory_state_present_after_build() {
    create_minimal_target
    bazel build simpletarget:top &> "$TEST_log"  \
        || fail "Couldn't build simpletarget"
    local server_pid="$(bazel info server_pid 2>> "$TEST_log")"
    "$bazel_javabase"/bin/jmap -histo:live "$server_pid" > histo.txt

    cat histo.txt >> "$TEST_log"
    assert_contains "GenRuleAction" histo.txt
    assert_contains "InMemoryNodeEntry" histo.txt
}

# Test directly that the inmemory state does not persist after the build.
function test_inmemory_state_absent_after_build_with_nokeep_state() {
    create_minimal_target
    bazel build --nokeep_state_after_build simpletarget:top &> "$TEST_log"  \
        || fail "Couldn't build simpletarget"
    local server_pid="$(bazel info server_pid 2>> "$TEST_log")"
    "$bazel_javabase"/bin/jmap -histo:live "$server_pid" > histo.txt

    cat histo.txt >> "$TEST_log"
    assert_not_contains "GenRuleAction$" histo.txt
    assert_not_contains "InMemoryNodeEntry" histo.txt
}

run_suite "test for --keep_state_after_build"
