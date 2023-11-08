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
# Integration testing of the JacocoCoverageRunner.
#

[ -z "$TEST_SRCDIR" ] && { echo "TEST_SRCDIR not set!" >&2; exit 1; }

# Load the unit-testing framework
source "$1" || \
  { echo "Failed to load unit-testing framework $1" >&2; exit 1; }

set +o errexit

JACOCO_RUNNER="${PWD}/$2"
shift 2

JACOCO_CMD="${JACOCO_RUNNER} --main_advice=com.google.testing.coverage.JacocoCoverageRunner"
#######################

function test_jar_with_plus_in_the_filename() {
  touch "Test.class"
  JAR_WITH_PLUS='file+with+plus.jar'
  zip -qo ${JAR_WITH_PLUS} "Test.class"
  export JACOCO_MAIN_CLASS="Test"
  ${JACOCO_CMD} \
  --main_advice_classpath="${JAR_WITH_PLUS}" >& $TEST_log && fail "expected build failure"
  expect_not_log "java.nio.file.NoSuchFileException"
}

run_suite "deploy_jar"
