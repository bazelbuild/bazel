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
# Tests the examples provided in Bazel
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function basic_glob_scenario_test_template() {
  local chars="$1"
  local pkg="pkg${chars}"
  echo "chars = ${chars}, pkg = ${pkg}"
  mkdir -p "${pkg}/resources"
  cat >"${pkg}/BUILD" <<EOF
java_library(name = 'main',
    resources = glob(["resources/**"]),
    srcs = ['Main.java'])
EOF

  for i in $(seq 1 10); do
    cat >"${pkg}/resources/file${chars}$i" <<EOF
file${chars}$i
EOF
  done

  cat >"$pkg/Main.java" <<'EOF'
public class Main {
  public static void main(String[] args) {
    System.out.println("Hello, World!");
  }
}
EOF

  bazel build "//${pkg}:main" &>"${TEST_log}" \
      || fail "Failed to build java target"

  nb_files="$(unzip -l "bazel-bin/${pkg}/libmain.jar" \
      | grep -F "file${chars}" | tee "${TEST_log}" | wc -l | xargs echo)"
  [ "10" = "${nb_files}" ] || fail "Expected 10 files, got ${nb_files}"
}

function test_space_dollar_and_parentheses() {
  basic_glob_scenario_test_template '$( )'
}

run_suite "Integration tests for handling of special characters"

