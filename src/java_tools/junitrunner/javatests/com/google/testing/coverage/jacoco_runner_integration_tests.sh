#!/usr/bin/env bash
#
# Copyright 2023 The Bazel Authors. All Rights Reserved.
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
# Integration testing of the JacocoCoverageRunner.
#

[ -z "$TEST_SRCDIR" ] && { echo "TEST_SRCDIR not set!" >&2; exit 1; }

# Load the unit-testing framework
source "$1" || \
  { echo "Failed to load unit-testing framework $1" >&2; exit 1; }

set +o errexit

JACOCO_RUNNER="${PWD}/$2"
JAR_WITH_PLUS="${PWD}/$3"
shift 3

JACOCO_CMD="${JACOCO_RUNNER} --main_advice=com.google.testing.coverage.JacocoCoverageRunner"
#######################

function test_jar_with_plus_in_the_filename() {
  export JACOCO_MAIN_CLASS="com.google.testing.coverage.TestClass"
  ${JACOCO_CMD} --main_advice_classpath="${JAR_WITH_PLUS}" >& $TEST_log || fail "expected success"
  expect_not_log "java.nio.file.NoSuchFileException"
  expect_log "Coverage run success!"
}

run_suite "deploy_jar"
