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
# A test for --discard_analysis_cache.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function write_hello_world_files() {
  mkdir -p hello || fail "mkdir hello failed"
  cat >hello/BUILD <<EOF
java_binary(name = 'hello',
  srcs = ['Hello.java'],
  main_class = 'Hello')
EOF

  cat >hello/Hello.java <<EOF
public class Hello {
  public static void main(String[] args) {
    System.out.println("hello!");
  }
}
EOF
}

#### TESTS #############################################################

function test_compile_helloworld() {
  write_hello_world_files
  bazel run --discard_analysis_cache hello:hello >&$TEST_log \
      || fail "Build failed"
  expect_log 'hello!'

  bazel run --discard_analysis_cache hello:hello >&$TEST_log \
      || fail "Build failed"
  expect_log 'hello!'

  # Check that further incremental builds work fine.
  bazel run hello:hello >&$TEST_log \
      || fail "Build failed"
  expect_log 'hello!'
}

run_suite "test for --discard_analysis_cache"
