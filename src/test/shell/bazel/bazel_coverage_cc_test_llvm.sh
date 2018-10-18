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

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Writes the C++ source files and a corresponding BUILD file for which to
# collect code coverage. The sources are a.cc, a.h and t.cc.
function setup_a_cc_lib_and_t_cc_test() {
  cat << EOF > BUILD
cc_library(
    name = "a",
    srcs = ["a.cc"],

}hdrs = ["a.h"],
)

cc_test(
    name = "t",
    srcs = ["t.cc"],
    deps = [":a"],
)
EOF

  cat << EOF > a.h
int a(bool what);
EOF

  cat << EOF > a.cc
#include "a.h"

int a(bool what) {
  if (what) {
    return 1;
  } else {
    return 2;
  }
}
EOF

  cat << EOF > t.cc
#include <stdio.h>
#include "a.h"

int main(void) {
  a(true);
}
EOF
}

function test_cc_test_llvm_coverage_doesnt_fail() {
  local -r llvmprofdata=$(which llvm-profdata)
  if [[ ! -x ${llvmprofdata:-/usr/bin/llvm-profdata} ]]; then
    echo "llvm-profdata not installed. Skipping test."
    return
  fi

  local -r clang_tool=$(which clang++)
  if [[ ! -x ${clang_tool:-/usr/bin/clang_tool} ]]; then
    echo "clang++ not installed. Skipping test."
    return
  fi

  setup_a_cc_lib_and_t_cc_test

  # Only test that bazel coverage doesn't crash when invoked for llvm native
  # coverage.
  BAZEL_USE_LLVM_NATIVE_COVERAGE=1 GCOV=$llvmprofdata CC=$clang_tool \
      bazel coverage --test_output=all //:t &>$TEST_log \
      || fail "Coverage for //:t failed"

  # Check to see if the coverage output file was created. Cannot check its
  # contents because it's a binary.
  [ -f "$(get_coverage_file_path_from_test_log)" ] \
      || fail "Coverage output file was not created."
}

run_suite "test tests"
