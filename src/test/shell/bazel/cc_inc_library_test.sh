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
# Tests the behavior of cc_inc_library.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
cc_binary(
  name = "bin",
  srcs = ["a.cc"],
  deps = [":inc"],
)

cc_inc_library(
  name = "inc",
  hdrs = ["hdr.h"],
)
EOF

  cat > package/a.cc <<EOF
#include <string.h>
#include "hdr.h"
int main() {
  return 0;
}
EOF

  cat > package/hdr.h <<EOF
int some_function();
EOF
}

function test_cc_library_include_prefix_external_repository() {
  r="$TEST_TMPDIR/r"
  mkdir -p "$TEST_TMPDIR/r/foo/v1"
  touch "$TEST_TMPDIR/r/WORKSPACE"
  echo "#define FOO 42" > "$TEST_TMPDIR/r/foo/v1/foo.h"
  cat > "$TEST_TMPDIR/r/foo/BUILD" <<EOF
cc_library(
  name = "foo",
  hdrs = ["v1/foo.h"],
  include_prefix = "foolib",
  strip_include_prefix = "v1",
  visibility = ["//visibility:public"],
)
EOF

  cat > WORKSPACE <<EOF
local_repository(
  name = "foo",
  path = "$TEST_TMPDIR/r",
)
EOF

  cat > BUILD <<EOF
cc_binary(
  name = "ok",
  srcs = ["ok.cc"],
  deps = ["@foo//foo"],
)

cc_binary(
  name = "bad",
  srcs = ["bad.cc"],
  deps = ["@foo//foo"],
)
EOF

  cat > ok.cc <<EOF
#include <stdio.h>
#include "foolib/foo.h"
int main() {
  printf("FOO is %d\n", FOO);
}
EOF

  cat > bad.cc <<EOF
#include <stdio.h>
#include "foo/v1/foo.h"
int main() {
  printf("FOO is %d\n", FOO);
}
EOF

  bazel build :bad && fail "Should not have found include at repository-relative path"
  bazel build :ok || fail "Should have found include at synthetic path"
}

function test_cc_inc_library_propagates_includes() {
  bazel build --verbose_failures //package:inc >& $TEST_log \
    || fail "Should build"
}

function test_cc_inc_library_stale_outputs() {
  rm -rf package
  mkdir -p package
  cat > package/BUILD <<EOF
cc_library(
  name = "a",
  srcs = ["a.cc"],
  deps = [":inc"],
)

cc_inc_library(
  name = "inc",
  hdrs = ["foo.h", "bar.h"],
)
EOF

  cat > package/a.cc <<EOF
#include <package/foo.h>
#include <package/bar.h>
EOF

  touch package/foo.h
  touch package/bar.h

  bazel build //package:a --spawn_strategy=standalone >& "$TEST_log" \
      || fail "Should have succeeded"
  cat > package/BUILD <<EOF
cc_library(
  name = "a",
  srcs = ["a.cc"],
  deps = [":inc", ":inc2"],
)

cc_inc_library(
  name = "inc",
  hdrs = ["foo.h"],
)

cc_inc_library(
  name = "inc2",
  hdrs = ["bar.h"],
)
EOF
  bazel build //package:a --spawn_strategy=standalone >& "$TEST_log" \
      || fail "Should have succeeded"
}

run_suite "cc_inc_library"
