#!/usr/bin/env bash
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
  add_rules_cc MODULE.bazel
  mkdir -p hello || fail "mkdir hello failed"
  cat >hello/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_binary(
  name = 'hello',
  srcs = ['hello.cc'],
  deps = [":hello_lib"],
)

cc_library(
  name = 'hello_header_only',
  hdrs = ['hello_header_only.h'],
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

  cat >hello/hello_header_only.h <<EOF
#ifdef private_header
#include "hello_private.h"
int func() {
  return helloPrivate() - 42;
}
#elif defined layering_violation
#include "base.h"
int func() {
  return base() - 42;
}
#else
#include "hello.h"
int func() {
  return hello() - 42;
}
#endif
EOF
}

function test_bazel_layering_check() {
  local -r clang_tool=$(which clang)
  if [[ ! -x ${clang_tool:-/usr/bin/clang_tool} ]]; then
    echo "clang not installed. Skipping test."
    return
  fi

  write_files

  CC="${clang_tool}" bazel build \
    //hello:hello --features=layering_check \
    &> "${TEST_log}" || fail "Build with layering_check failed"

  bazel-bin/hello/hello || fail "the built binary failed to run"

  if [[ ! -e bazel-bin/hello/hello.cppmap ]]; then
    fail "module map files were not generated"
  fi

  if [[ ! -e bazel-bin/hello/hello_lib.cppmap ]]; then
    fail "module map files were not generated"
  fi

  CC="${clang_tool}" bazel build \
    //hello:hello --copt=-DFORCE_REBUILD=1 \
    --spawn_strategy=local --features=layering_check \
    &> "${TEST_log}" || fail "Build with layering_check failed without sandboxing"

  CC="${clang_tool}" bazel build \
    --copt=-D=private_header \
    //hello:hello --features=layering_check \
    &> "${TEST_log}" && fail "Build of private header violation with "\
    "layering_check should have failed"
  expect_log "use of private header from outside its module: 'hello_private.h'"

  CC="${clang_tool}" bazel build \
    --copt=-D=layering_violation \
    //hello:hello --features=layering_check \
    &> "${TEST_log}" && fail "Build of private header violation with "\
    "layering_check should have failed"
  expect_log "module //hello:hello does not depend on a module exporting "\
    "'base.h'"
}

function test_bazel_layering_check_header_only() {
  local -r clang_tool=$(which clang)
  if [[ ! -x ${clang_tool:-/usr/bin/clang_tool} ]]; then
    echo "clang not installed. Skipping test."
    return
  fi

  write_files

  CC="${clang_tool}" bazel build \
    //hello:hello_header_only --features=layering_check --features=parse_headers \
    -s --process_headers_in_dependencies \
    &> "${TEST_log}" || fail "Build with layering_check + parse_headers failed"

  if [[ ! -e bazel-bin/hello/hello_header_only.cppmap ]]; then
    fail "module map file for hello_header_only was not generated"
  fi

  if [[ ! -e bazel-bin/hello/hello_lib.cppmap ]]; then
    fail "module map file for hello_lib was not generated"
  fi

  CC="${clang_tool}" bazel build \
    --copt=-D=private_header \
    //hello:hello_header_only --features=layering_check --features=parse_headers \
    --process_headers_in_dependencies \
    &> "${TEST_log}" && fail "Build of private header violation with "\
    "layering_check + parse_headers should have failed"
  expect_log "use of private header from outside its module: 'hello_private.h'"

  CC="${clang_tool}" bazel build \
    --copt=-D=layering_violation \
    //hello:hello_header_only --features=layering_check --features=parse_headers \
    --process_headers_in_dependencies \
    &> "${TEST_log}" && fail "Build of private header violation with "\
    "layering_check + parse_headers should have failed"
  expect_log "module //hello:hello_header_only does not depend on a module exporting "\
    "'base.h'"
}

run_suite "test layering_check"
