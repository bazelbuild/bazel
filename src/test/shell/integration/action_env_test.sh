#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# An end-to-end test that Bazel's provides the correct environment variables
# to actions.

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/testenv.sh \
  || { echo "testenv.sh not found!" >&2; exit 1; }

create_and_cd_client
put_bazel_on_path
write_default_bazelrc


#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p pkg
  cat > pkg/BUILD <<EOF
genrule(
  name = "showenv",
  outs = ["env.txt"],
  cmd = "env | sort > \"\$@\""
)
EOF
}

#### TESTS #############################################################

function test_simple() {
  export FOO=baz
  bazel build --action_env=FOO=bar pkg:showenv \
      || fail "bazel build showenv failed"
  cat `bazel info bazel-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=bar"
}

function test_simple_latest_wins() {
  export FOO=environmentfoo
  export BAR=environmentbar
  bazel build --action_env=FOO=foo \
      --action_env=BAR=willbeoverridden --action_env=BAR=bar pkg:showenv \
      || fail "bazel build showenv failed"
  cat `bazel info bazel-genfiles`/pkg/env.txt > $TEST_log
  expect_log "FOO=foo"
  expect_log "BAR=bar"
}

run_suite "Tests for bazel's handling of environment variables in actions"
