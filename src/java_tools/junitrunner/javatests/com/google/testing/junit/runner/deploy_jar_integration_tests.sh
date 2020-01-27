#!/bin/bash
#
# Copyright 2012 The Bazel Authors. All Rights Reserved.
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
# Integration testing of _deploy.jar behavior. Not using the runfiles wrapper
# can create a few puzzling hazards. This verifies that we warn the user for
# the most obvious ones.
#
# These tests operate by executing test methods in a testbed deploy.jar program
# and checking the output.
#

[ -z "$TEST_SRCDIR" ] && { echo "TEST_SRCDIR not set!" >&2; exit 1; }

# Load the unit-testing framework
source "$1" || \
  { echo "Failed to load unit-testing framework $1" >&2; exit 1; }

set +o errexit

unset TEST_PREMATURE_EXIT_FILE
JAVA_HOME="$2"
TESTBED_JAR="${PWD}/$3"

shift 3

#######################

# Test that we see a warning about missing the test suite Java system property
function test_Warning() {
  test_pid=""
  ${JAVA_HOME}/bin/java -jar $TESTBED_JAR >& $TEST_log && test_pid=$!

  expect_log "The test suite Java system property .* is required but missing"

  wait $test_pid || fail "Expected process to finish successfully"
}

run_suite "deploy_jar"
