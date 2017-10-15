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
# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

#### TESTS #############################################################

function test_no_rebuild_on_irrelevant_header_change() {
  mkdir -p a
  cat > a/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=["b"])
cc_library(name="b", srcs=["b1.h", "b2.h"])
EOF

  cat > a/a.cc <<EOF
#include "a/b1.h"

int main(void) {
  return B_RETURN_VALUE;
}
EOF

  cat > a/b1.h <<EOF
#define B_RETURN_VALUE 31
EOF

  cat > a/b2.h <<EOF
=== BANANA ===
EOF

  bazel build //a || fail "build failed"
  echo "CHERRY" > a/b2.h
  bazel build //a >& $TEST_log || fail "build failed"
  expect_not_log "Compiling a/a.cc"
}

function test_new_header_is_required() {
  mkdir -p a
  cat > a/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=[":b"])
cc_library(name="b", srcs=["b1.h", "b2.h"])
EOF

  cat > a/a.cc << EOF
#include "a/b1.h"

int main(void) {
    return B1;
}
EOF

  cat > a/b1.h <<EOF
#define B1 3
EOF

  cat > a/b2.h <<EOF
#define B2 4
EOF

  bazel build //a || fail "build failed"
  cat > a/a.cc << EOF
#include "a/b1.h"
#include "a/b2.h"

int main(void) {
    return B1 + B2;
}
EOF

  bazel build //a || fail "build failled"
}

function test_no_recompile_on_shutdown() {
  mkdir -p a
  cat > a/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=["b"])
cc_library(name="b", includes=["."], hdrs=["b.h"])
EOF

  cat > a/a.cc <<EOF
#include "b.h"

int main(void) {
  return B_RETURN_VALUE;
}
EOF

  cat > a/b.h <<EOF
#define B_RETURN_VALUE 31
EOF

  bazel build -s //a >& $TEST_log || fail "build failed"
  expect_log "Compiling a/a.cc"
  bazel shutdown >& /dev/null || fail "query failed"
  bazel build -s //a >& $TEST_log || fail "build failed"
  expect_not_log "Compiling a/a.cc"
}

run_suite "Tests for Bazel's C++ rules"
