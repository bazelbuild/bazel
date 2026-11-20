#!/usr/bin/env bash
#
# Copyright 2011 The Bazel Authors. All rights reserved.
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
# warnings_as_errors_test.sh: A couple of end to end tests for the warnings
# as errors functionality.

# --- begin runfiles.bash initialization ---
set -euo pipefail
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

function set_up() {
  add_rules_java MODULE.bazel
}

#### HELPER FUNCTIONS ##################################################

# Scenario description:
#   //java/main
#     Contains a single class Main with the following warnings:
#       A fall through in a switch.
#       An unchecked class conversion when creating a list.
#       A call to a deprecated method in package //java/hello_library
#   //java/hello_library
#     Contains a single class HelloLibrary with no warnings.
function setup_scenario() {
  local werroropts=$1

  mkdir -p java/main
  # Deprecation and fallthrough warnings are silenced by default.
  cat >java/main/BUILD <<EOF
load("@rules_java//java:java_binary.bzl", "java_binary")
java_binary(
    name = 'main',
    main_class = 'main.Main',
    srcs = ['Main.java'],
    javacopts = [
        # TODO(cushon): remove flag once fallthrough is an error by default
        '-Xep:FallThrough:OFF',
        '-Xlint:unchecked,deprecation,fallthrough',
        $werroropts
    ],
    deps = ['//java/hello_library'],
)
EOF

  cat >java/main/Main.java <<EOF
      package main;
      import hello_library.HelloLibrary;
      import java.util.ArrayList;
      public class Main {
        public static void main(String[] args) {
          HelloLibrary.funcHelloLibrary();
          System.out.println("Hello, World!");
          ArrayList<String> list = new ArrayList();
          int num = 10;
          switch (num) {
            case 1:
              num = 3;
            case 2:
              num = 4;
          }
        }
      }
EOF

  mkdir -p java/hello_library
  cat >java/hello_library/BUILD <<EOF
load("@rules_java//java:java_library.bzl", "java_library")
package(default_visibility=['//visibility:public'])
java_library(name = 'hello_library',
             srcs = ['HelloLibrary.java'])
EOF

  cat >java/hello_library/HelloLibrary.java <<EOF
      package hello_library;
      public class HelloLibrary {
        @Deprecated
        public static void funcHelloLibrary() {
          System.out.print("Hello, Library!;");
        }
      }
EOF
}

function buildExpectSuccess() {
  local target="$1"
  bazel clean
  bazel build "$target" 2>stderr.txt ||
      { cat stderr.txt >&2; fail "build failed"; }
}

function buildExpectFailure() {
  local target="$1"
  bazel clean
  bazel build "$target" 2>stderr.txt &&
      { cat stderr.txt >&2; fail "build succeeded"; }
}

# Helper function for checking presence of a warning or error in stderr.txt.
# First parameter is warning/error name, second parameter is the type of
# exception, e.g. warning or error, and third parameter is the message's
# associated line number.
function expectOutput() {
  local name=$1
  local e_type=$2
  local line_number=$3
  local displayed_e_type=${e_type/error/}
  grep -q ":$line_number:.*$displayed_e_type.*\[$name\]" stderr.txt ||
      { cat stderr.txt >&2; fail "Expected $name $e_type on line $line_number"; }
}

# Helper function for checking absence of a warning or error in stderr.txt.
function expectNotOutput() {
  local name=$1
  grep -q "$name" stderr.txt &&
    { cat stderr.txt >&2; fail "Was not expecting $name"; }
}

#### TESTS #############################################################

# Check that we get all the warnings.
function test_warnings_without_flag() {
  # unchecked and rawtypes diagnostics were promoted to errors after
  # this test was written, demote them to warnings to match its assumptions
  setup_scenario "'-Werror:-unchecked,-rawtypes'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt || fail "build failed"
  expectOutput "deprecation" "warning" 6
  expectOutput "unchecked" "warning" 8
  expectOutput "fallthrough" "warning" 13
}

# Check that if an error is shown, no warnings are shown after the error.
function test_no_warnings_after_errors() {
  setup_scenario "'-Werror:+unchecked'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "warning" 6
  expectOutput "unchecked" "error" 8
  expectOutput "fallthrough" "warning" 13
}

# Check that only specifically listed warnings becomes errors.
function test_specific_warning_as_errors() {
  setup_scenario "'-Werror:+fallthrough'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "warning" 6
  expectOutput "unchecked" "error" 8
  expectOutput "fallthrough" "error" 13
}

# Check that if we specify Werror:+all, we get all errors except deprecation,
# which should never be treated as an error.
function test_all_warnings_as_errors() {
  setup_scenario "'-Werror:+all'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "error" 6
  expectOutput "unchecked" "error" 8
  expectOutput "fallthrough" "error" 13
}

# Check that all warnings are treated as errors except for specifically
# excluded warnings.
function test_all_warnings_as_errors_with_specific_exclusions {
  setup_scenario "'-Werror:+all,-deprecation,-unchecked,-rawtypes'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "warning" 6
  expectOutput "unchecked" "warning" 8
  expectOutput "fallthrough" "error" 13
}

# Check that last specified warning takes precedence.
# -Werror should get reduced to +unchecked.
function test_warnings_as_errors_precedence() {
  setup_scenario "'-Werror:-unchecked,+unchecked'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "warning" 6
  expectOutput "unchecked" "error" 8
  expectOutput "fallthrough" "warning" 13
}

# Check that +all overrides all preceding excluded warnings.
function test_plus_all_overrides_preceding_excluded_warnings() {
  setup_scenario "'-Werror:-unchecked,+all'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "error" 6
  expectOutput "unchecked" "error" 8
  expectOutput "fallthrough" "error" 13
}

# Check that -all overrides all preceding included warnings.
function test_minus_all_overrides_preceding_included_warnings() {
  setup_scenario "'-Werror:+unchecked,-all,+fallthrough'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "warning" 6
  expectOutput "unchecked" "warning" 8
  expectOutput "fallthrough" "error" 13
}

# Check that only the last -Werror is used.
function test_only_last_werror_used() {
  setup_scenario "'-Werror:+all', '-Werror:-all', '-Werror:+fallthrough'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "warning" 6
  expectOutput "unchecked" "warning" 8
  expectOutput "fallthrough" "error" 13
}

# Check that -Werror without a colon causes the build to fail and an error
# message to be printed.
function test_werror_no_colon_fail() {
  setup_scenario "'-Werror'"
  bazel clean
  bazel build //java/main:main 2>stderr.txt && fail "build succeeded"
  expectOutput "deprecation" "error" 6
  expectOutput "unchecked" "error" 8
  expectOutput "fallthrough" "error" 13
  local expected="error: warnings found and -Werror specified"
  grep "$expected" stderr.txt >/dev/null || fail "Error not found"
}

run_suite "Warning as errors tests"
