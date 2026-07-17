#!/usr/bin/env bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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

function write_hello_world_files {
  mkdir -p hello_shared || fail "mkdir hello failed"
  cat >hello_shared/BUILD <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_shared_library.bzl", "cc_shared_library")

cc_library(
    name = "a",
    srcs = ["a.cc"],
    deps = [":b"],
)

cc_library(
    name = "b",
    srcs = ["b.cc"],
)

cc_shared_library(
    name = "hello_so",
    deps = [":a"],
)
EOF
  touch hello_shared/a.cc
  touch hello_shared/b.cc
}

function test_bazel_thinlto() {

  if is_darwin; then
    echo "This test doesn't run on Darwin. Skipping."
    return
  fi

  local -r clang_tool=$(which clang)
  if [[ ! -x ${clang_tool:-/usr/bin/clang_tool} ]]; then
    echo "clang not installed. Skipping test."
    return
  fi

  local major_version=$($clang_tool --version | \
    grep -oP 'version.*' | cut -d' ' -f 2 | cut -d '.' -f 1)

  if [[ $major_version < 6 ]]; then
    echo "clang version is smaller than 6.0. Skipping test."
    return
  fi

  write_hello_world_files

  CC=$clang_tool bazel run \
    //hello_shared:hello_so -c opt -s --features=thin_lto &>$TEST_log \
    || fail "Build with ThinLTO failed"

  grep -q "action 'LTO Backend Compile" $TEST_log \
    || fail "LTO Actions missing"

  # Find thinlto.bc files in subdirectories
  if [[ -z $(find "bazel-bin/hello_shared/libhello_so.so.lto" -path "*hello_shared/_objs/a/a.o.thinlto.bc" -print -quit) ]]; then
    fail "bitcode files were not generated"
  fi
  if [[ -z $(find "bazel-bin/hello_shared/libhello_so.so.lto" -path "*hello_shared/_objs/b/b.o.thinlto.bc" -print -quit) ]]; then
    fail "bitcode files were not generated"
  fi
}

run_suite "test ThinLTO cc_shared_library"
