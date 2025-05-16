#!/usr/bin/env bash
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

set -euo pipefail
# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "$(rlocation "io_bazel/src/test/shell/sandboxing_test_utils.sh")" \
  || { echo "sandboxing_test_utils.sh not found!" >&2; exit 1; }

IS_WINDOWS=false
case "$(uname | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  IS_WINDOWS=true
esac

# TODO(rongjiecomputer): Individual marking external tools as readable with
# --sandbox_writable_path flag is only a temporary solution. Eventually Bazel
# should add those external tools it needs as readable automatically.
sandbox_flags=""
if "${IS_WINDOWS}"; then
  sandbox_flags="--experimental_use_windows_sandbox=yes"
  if [[ -n "${WINDOWS_SANDBOX+x}" ]]; then
    sandbox_flags="${sandbox_flags} --experimental_windows_sandbox_path=${WINDOWS_SANDBOX}"
  fi
  if [[ -n "${BAZEL_VC+x}" ]]; then
    sandbox_flags="${sandbox_flags} --sandbox_writable_path=${BAZEL_VC}"
  fi
  if [[ -n "${WIN10_SDK+x}" ]]; then
    sandbox_flags="${sandbox_flags} --sandbox_writable_path=${WIN10_SDK}"
  fi
fi

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
  bazel build $sandbox_flags --spawn_strategy=sandboxed //examples/cpp:hello-world &> $TEST_log \
    || fail "Building hello-world failed"

  bazel-bin/examples/cpp/hello-world | fgrep "greetings from the header" \
    || fail "Did not print expected string 'greetings from the header'"

  sed "s/from the header/from the modified header/g" < examples/cpp/lib/hello-lib.h > tmp \
    || fail "modifying hello-lib.h failed"

  mv tmp examples/cpp/lib/hello-lib.h \
    || fail "moving modified header file back to examples/cpp/lib/hello-lib.h failed"

  bazel build $sandbox_flags --spawn_strategy=sandboxed //examples/cpp:hello-world &> $TEST_log \
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

  bazel build $sandbox_flags --spawn_strategy=sandboxed //examples/cpp:hello-lib &> $TEST_log \
    && fail "build should not have succeeded with missing header file"

  fgrep "fatal error: examples/cpp/lib/hello-lib.h: No such file or directory" $TEST_log \
    || fgrep "fatal error: 'examples/cpp/lib/hello-lib.h' file not found" $TEST_log \
    || fgrep "examples/cpp/lib/hello-lib.c(1): fatal error C1083: Cannot open include file: 'examples/cpp/lib/hello-lib.h': No such file or directory" $TEST_log \
    || fail "could not find 'No such file or directory' error message in bazel output"
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

  bazel build $sandbox_flags --spawn_strategy=sandboxed //examples/cpp:hello-lib &> $TEST_log \
    || fail "building hello-lib should have succeeded with header file in srcs"

  bazel build $sandbox_flags --spawn_strategy=sandboxed //examples/cpp:hello-world &> $TEST_log \
    && fail "building hello-world should not have succeeded with library header file in srcs"

  fgrep "undeclared inclusion(s) in rule '//examples/cpp:hello-world'" $TEST_log \
    || fail "could not find 'undeclared inclusion' error message in bazel output"
}

function test_standalone_cpp_build_rebuilds_on_change() {
  bazel build $sandbox_flags --spawn_strategy=standalone //examples/cpp:hello-world &> $TEST_log \
    || fail "Building hello-world failed"

  bazel-bin/examples/cpp/hello-world | fgrep "greetings from the header" \
    || fail "Did not print expected string 'greetings from the header'"

  sed "s/from the header/from the modified header/g" < examples/cpp/lib/hello-lib.h > tmp \
    || fail "modifying hello-lib.h failed"

  mv tmp examples/cpp/lib/hello-lib.h \
    || fail "moving modified header file back to examples/cpp/lib/hello-lib.h failed"

  bazel build $sandbox_flags --spawn_strategy=standalone //examples/cpp:hello-world &> $TEST_log \
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

  bazel build $sandbox_flags --spawn_strategy=standalone //examples/cpp:hello-lib &> $TEST_log \
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

  bazel build $sandbox_flags --spawn_strategy=standalone //examples/cpp:hello-lib &> $TEST_log \
    || fail "building hello-lib should have succeeded with header file in srcs"

  bazel build $sandbox_flags --spawn_strategy=standalone //examples/cpp:hello-world &> $TEST_log \
    && fail "building hello-world should not have succeeded with library header file in srcs"

  fgrep "undeclared inclusion(s) in rule '//examples/cpp:hello-world'" $TEST_log \
    || fail "could not find 'undeclared inclusion' error message in bazel output"
}

# The test shouldn't fail if the environment doesn't support running it.
check_sandbox_allowed || "${IS_WINDOWS}" || exit 0

run_suite "sandbox"
