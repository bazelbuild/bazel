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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

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

function test_cc_inc_library_propagates_includes() {
  bazel build --verbose_failures //package:inc >& $TEST_log \
    || fail "Should build"
}

run_suite "cc_inc_library"
