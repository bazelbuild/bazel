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
# Tests the examples provided in Bazel
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function set_up() {
  copy_examples
}

function check_has_rustc() {
  PATH=/usr/bin:/usr/local/bin:$PATH
  if [ ! -x "$(which rustc)" ]; then
    echo "No rustc found. Skipping..."
    return false
  fi
}

function test_rust() {
  local hello_lib_pkg=examples/rust/hello_lib
  assert_build_output ./bazel-bin/${hello_lib_pkg}/libhello_lib.rlib ${hello_lib_pkg}:hello_lib

  local hello_world_pkg=examples/rust/hello_world
  assert_build_output ./bazel-bin/${hello_world_pkg}/hello_world ${hello_world_pkg}:hello_world
  assert_binary_run_from_subdir "bazel-bin/${hello_world_pkg}/hello_data" "Hello world"
}

function test_rust_test() {
  hello_lib_test=examples/rust/hello_lib
  assert_build //${hello_lib_test}:greeting
  assert_test_ok //${hello_lib_test}:greeting
}

check_has_rustc || exit 0
run_suite "rust_examples"
