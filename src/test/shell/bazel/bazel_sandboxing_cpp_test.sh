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
# Test C++ builds with the sandboxing spawn strategy.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source ${CURRENT_DIR}/../sandboxing_test_utils.sh \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }

function set_up {
  mkdir -p examples/cpp/{bin,lib}
  cat << 'EOF' > examples/cpp/BUILD
cc_library(
    name = "hello-lib",
    srcs = ["lib/hello-lib.c"],
    hdrs = ["lib/hello-lib.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["bin/hello-world.c"],
    deps = [":hello-lib"],
)
EOF
  cat << 'EOF' > examples/cpp/lib/hello-lib.c
#include "examples/cpp/lib/hello-lib.h"

void greet(char *greeting) {
  printf("hello-lib says: %s\n", greeting);
}
EOF
  cat << 'EOF' > examples/cpp/lib/hello-lib.h
#ifndef EXAMPLES_CPP_LIB_HELLO_LIB_H_
#define EXAMPLES_CPP_LIB_HELLO_LIB_H_

#include <stdio.h>

void greet(char *greeting);

static inline void greet_from_header() {
  printf("greetings from the header");
}

#endif  // EXAMPLES_CPP_LIB_HELLO_LIB_H_
EOF
  cat << 'EOF' > examples/cpp/bin/hello-world.c
#include "examples/cpp/lib/hello-lib.h"

int main(int argc, char** argv) {
  if (argc > 1) {
    greet(argv[1]);
  }
  greet_from_header();
  return 0;
}
EOF
}

# Tests for #473: Sandboxing for C++ compilation was accidentally disabled.
function test_sandboxed_cpp_build_rebuilds_on_change() {
  bazel build --spawn_strategy=sandboxed //examples/cpp:hello-world &> $TEST_log \
    || fail "Building hello-world failed"

  bazel-bin/examples/cpp/hello-world | fgrep "greetings from the header" \
    || fail "Did not print expected string 'greetings from the header'"

  sed "s/from the header/from the modified header/g" < examples/cpp/lib/hello-lib.h > tmp \
    || fail "modifying hello-lib.h failed"

  mv tmp examples/cpp/lib/hello-lib.h \
    || fail "moving modified header file back to examples/cpp/lib/hello-lib.h failed"

  bazel build --spawn_strategy=sandboxed //examples/cpp:hello-world &> $TEST_log \
    || fail "Building modified hello-world failed"

  bazel-bin/examples/cpp/hello-world | fgrep "greetings from the modified header" \
    || fail "Did not print expected string 'greetings from the modified header'"
}

function test_sandboxed_cpp_build_catches_missing_header_via_sandbox() {
  cat << 'EOF' > examples/cpp/BUILD
cc_library(
    name = "hello-lib",
    srcs = ["lib/hello-lib.c"],
)
EOF

  bazel build --spawn_strategy=sandboxed //examples/cpp:hello-lib &> $TEST_log \
    && fail "build should not have succeeded with missing header file"

  fgrep "No such file or directory" $TEST_log \
    || fgrep "file not found" $TEST_log \
    || fgrep "this rule is missing dependency declarations" $TEST_log \
    || fail "could not find an indication of a missing file in bazel output"
  fgrep "examples/cpp/lib/hello-lib.h" $TEST_log \
    || fail "could not find 'examples/cpp/lib/hello-lib.h' bazel output"
}

# TODO(philwo) turns out, we have this special "hdrs" attribute and in theory you can only include
# header files from libraries that are specified in "hdrs" and not "srcs", but we never check that,
# so the test fails. :(
function DISABLED_test_sandboxed_cpp_build_catches_header_only_in_srcs() {
  cat << 'EOF' > examples/cpp/BUILD
cc_library(
    name = "hello-lib",
    srcs = ["hello-lib.c", "hello-lib.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.c"],
    deps = [":hello-lib"],
)
EOF

  bazel build --spawn_strategy=sandboxed //examples/cpp:hello-lib &> $TEST_log \
    || fail "building hello-lib should have succeeded with header file in srcs"

  bazel build --spawn_strategy=sandboxed //examples/cpp:hello-world &> $TEST_log \
    && fail "building hello-world should not have succeeded with library header file in srcs"

  fgrep "undeclared inclusion(s) in rule '//examples/cpp:hello-world'" $TEST_log \
    || fail "could not find 'undeclared inclusion' error message in bazel output"
}

function test_standalone_cpp_build_rebuilds_on_change() {
  bazel build --spawn_strategy=standalone //examples/cpp:hello-world &> $TEST_log \
    || fail "Building hello-world failed"

  bazel-bin/examples/cpp/hello-world | fgrep "greetings from the header" \
    || fail "Did not print expected string 'greetings from the header'"

  sed "s/from the header/from the modified header/g" < examples/cpp/lib/hello-lib.h > tmp \
    || fail "modifying hello-lib.h failed"

  mv tmp examples/cpp/lib/hello-lib.h \
    || fail "moving modified header file back to examples/cpp/lib/hello-lib.h failed"

  bazel build --spawn_strategy=standalone //examples/cpp:hello-world &> $TEST_log \
    || fail "Building modified hello-world failed"

  bazel-bin/examples/cpp/hello-world | fgrep "greetings from the modified header" \
    || fail "Did not print expected string 'greetings from the modified header'"
}

function test_standalone_cpp_build_catches_missing_header() {
  cat << 'EOF' > examples/cpp/BUILD
cc_library(
    name = "hello-lib",
    srcs = ["lib/hello-lib.c"],
)
EOF

  bazel build --spawn_strategy=standalone //examples/cpp:hello-lib &> $TEST_log \
    && fail "build should not have succeeded with missing header file"

  fgrep "undeclared inclusion(s) in rule '//examples/cpp:hello-lib'" $TEST_log \
    || fail "could not find 'undeclared inclusion' error message in bazel output"
}

# TODO(philwo) disabled for the same reason as test_sandboxed_cpp_build_catches_header_only_in_srcs
# above.
function DISABLED_test_standalone_cpp_build_catches_header_only_in_srcs() {
  cat << 'EOF' > examples/cpp/BUILD
cc_library(
    name = "hello-lib",
    srcs = ["hello-lib.c", "hello-lib.h"],
)

cc_binary(
    name = "hello-world",
    srcs = ["hello-world.c"],
    deps = [":hello-lib"],
)
EOF

  bazel build --spawn_strategy=standalone //examples/cpp:hello-lib &> $TEST_log \
    || fail "building hello-lib should have succeeded with header file in srcs"

  bazel build --spawn_strategy=standalone //examples/cpp:hello-world &> $TEST_log \
    && fail "building hello-world should not have succeeded with library header file in srcs"

  fgrep "undeclared inclusion(s) in rule '//examples/cpp:hello-world'" $TEST_log \
    || fail "could not find 'undeclared inclusion' error message in bazel output"
}

# The test shouldn't fail if the environment doesn't support running it.
check_supported_platform || exit 0
check_sandbox_allowed || exit 0

run_suite "sandbox"
