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
#
# Test top-level package
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_empty_package() {
  cat > BUILD <<EOF
java_binary(
    name = "noise",
    main_class = "Noise",
    srcs = ["Noise.java"],
)
EOF

  cat > Noise.java <<EOF
public class Noise {
  public static void main(String args[]) {
    System.out.println("SCREEEECH");
  }
}
EOF

  bazel run -s //:noise &> $TEST_log || fail "Failed to run //:noise"
  cat $TEST_log
  expect_log "SCREEEECH"
}

function test_empty_external() {
  mkdir foo
  cd foo
  # Create a dummy BUILD file, otherwise `bazel build` will complain that there
  # were no targets to build.
  cat > BUILD <<EOF
exports_files(["BUILD"])
EOF
  mkdir external
  bazel build ... &> $TEST_log || fail "Failed to build ..."
  bazel build --experimental_sibling_repository_layout ... &> $TEST_log \
      || fail "Failed to build ..."
}

run_suite "empty package tests"
