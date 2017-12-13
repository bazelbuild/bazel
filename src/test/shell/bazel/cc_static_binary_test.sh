#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# Tests for Bazel's static linking of C++ binaries.

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if ! is_linux; then
  echo "Tests for cc_binary static linking currently only run in Linux (this platform is ${PLATFORM})" >&2
  exit 0
fi

function set_up() {
  mkdir -p cpp/static
  cat > cpp/static/main.cc <<EOF
#include <iostream>
int main() {
  std::cout << "hello world!\n";
  return 0;
}
EOF
}

function test_linux_somewhat_static_cc_binary() {
  cat > cpp/static/BUILD <<EOF
cc_binary(
  name = "somewhat-static",
  srcs = ["main.cc"],
  linkstatic = 1,
)
EOF
  assert_build //cpp/static:somewhat-static
}

function test_linux_mostly_static_cc_binary() {
  cat > cpp/static/BUILD <<EOF
cc_binary(
  name = "mostly-static",
  srcs = ["main.cc"],
  linkstatic = 1,
  linkopts = [
    "-static-libstdc++",
    "-static-libgcc",
  ],
)
EOF
  assert_build //cpp/static:mostly-static
  LDD_STDOUT=$(ldd bazel-bin/cpp/static/mostly-static)
  if [[ "${LDD_STDOUT}" =~ "libstdc++" ]]; then
    log_fatal "A mostly-static binary shouldn't dynamically link libstdc++"
  fi
}

function test_linux_fully_static_cc_binary() {
  cat > cpp/static/BUILD <<EOF
cc_binary(
  name = "fully-static",
  srcs = ["main.cc"],
  linkstatic = 1,
  linkopts = [
    "-static",
    "-static-libstdc++",
    "-static-libgcc",
  ],
)
EOF
  assert_build //cpp/static:fully-static
  LDD_STDOUT=$(ldd bazel-bin/cpp/static/fully-static || true)
  if [[ ! "${LDD_STDOUT}" =~ "not a dynamic executable" ]]; then
    log_fatal "Expected a fully-static binary"
  fi
}

run_suite "static cc_binary"
