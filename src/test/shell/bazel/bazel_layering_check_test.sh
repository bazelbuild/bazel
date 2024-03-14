#!/bin/bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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

set -euo pipefail

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function write_files {
  mkdir -p hello || fail "mkdir hello failed"
  cat >hello/BUILD <<EOF
cc_binary(
  name = 'hello',
  srcs = ['hello.cc'],
  deps = [":hello_lib"],
)

cc_library(
  name = "hello_lib",
  srcs = ["hello_private.h", "hellolib.cc"],
  hdrs = ["hello.h"],
  deps = [":base"],
  implementation_deps = [":implementation"],
)

cc_library(
  name = "base",
  srcs = ["base.cc"],
  hdrs = ["base.h"],
)

cc_library(
  name = "implementation",
  srcs = ["implementation.cc"],
  hdrs = ["implementation.h"],
)
EOF
  cat >hello/hello.h <<EOF
#include "hello_private.h"
int hello();
EOF

  cat >hello/hello_private.h <<EOF
int helloPrivate();
EOF

  cat >hello/base.h <<EOF
int base();
EOF

  cat >hello/base.cc <<EOF
#include "base.h"

int base() {
  return 42;
}
EOF

  cat >hello/implementation.h <<EOF
int implementation();
EOF

  cat >hello/implementation.cc <<EOF
#include "implementation.h"

int implementation() {
  return 42;
}
EOF

  cat >hello/hellolib.cc <<EOF
#include "hello.h"
#include "base.h"
#include "implementation.h"

int helloPrivate() {
  implementation();
  return base();
}

int hello() {
  return helloPrivate();
}
EOF

  cat >hello/hello.cc <<EOF
#ifdef private_header
#include "hello_private.h"
int main() {
  return helloPrivate() - 42;
}
#elif defined layering_violation
#include "base.h"
int main() {
  return base() - 42;
}
#else
#include "hello.h"
int main() {
  return hello() - 42;
}
#endif
EOF
}

# TODO(hlopko): Add a test for a "toplevel" header-only library
#   once we have parse_headers support in cc_configure.
function test_bazel_layering_check() {
  if is_darwin; then
    echo "This test doesn't run on Darwin. Skipping."
    return
  fi

  local -r clang_tool=$(which clang)
  if [[ ! -x ${clang_tool:-/usr/bin/clang_tool} ]]; then
    echo "clang not installed. Skipping test."
    return
  fi

  write_files

  CC="${clang_tool}" bazel build \
    //hello:hello --linkopt=-fuse-ld=gold --features=layering_check \
    &> "${TEST_log}" || fail "Build with layering_check failed"

  bazel-bin/hello/hello || fail "the built binary failed to run"

  if [[ ! -e bazel-bin/hello/hello.cppmap ]]; then
    fail "module map files were not generated"
  fi

  if [[ ! -e bazel-bin/hello/hello_lib.cppmap ]]; then
    fail "module map files were not generated"
  fi

  # Specifying -fuse-ld=gold explicitly to override -fuse-ld=/usr/bin/ld.gold
  # passed in by cc_configure because Ubuntu-16.04 ships with an old
  # clang version that doesn't accept that.
  CC="${clang_tool}" bazel build \
    --copt=-D=private_header \
    //hello:hello --linkopt=-fuse-ld=gold --features=layering_check \
    &> "${TEST_log}" && fail "Build of private header violation with "\
    "layering_check should have failed"
  expect_log "use of private header from outside its module: 'hello_private.h'"

  CC="${clang_tool}" bazel build \
    --copt=-D=layering_violation \
    //hello:hello --linkopt=-fuse-ld=gold --features=layering_check \
    &> "${TEST_log}" && fail "Build of private header violation with "\
    "layering_check should have failed"
  expect_log "module //hello:hello does not depend on a module exporting "\
    "'base.h'"
}

run_suite "test layering_check"
