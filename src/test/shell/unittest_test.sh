#!/bin/bash
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
# not a proper test suite, but
# - a sanity check that unittest.bash is syntactically valid
# - and a means to run some quick experiments

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${DIR}/unittest.bash || { echo "Could not source unittest.sh" >&2; exit 1; }

function test_1() {
  echo "Everything is okay in test_1"
}

function test_2() {
  echo "Everything is okay in test_2"
}

function test_timestamp() {
  local ts=$(timestamp)
  [[ $ts =~ ^[0-9]{13}$ ]] || fail "timestamp wan't valid: $ts"

  local time_diff=$(get_run_time 100000 223456)
  assert_equals $time_diff 123.456
}

run_suite "unittests Tests"
