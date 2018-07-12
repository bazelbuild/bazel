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

function test_flatten_virtual_headers() {
  mkdir -p a
  cat > a/BUILD <<EOF
objc_library(name="a", srcs=["main.m"], deps=[":b"])

objc_library(name="b",
    hdrs=["header.h", "one/deep/header/deep.h"],
    include_prefix='foo',
    flatten_virtual_headers=True
)
EOF

  cat > a/main.m << EOF
#include <foo/header.h>
#include <foo/deep.h>

int main(void) {
    return B1 + B2;
}
EOF

  cat > a/header.h <<EOF
#define B1 3
EOF

  mkdir -p a/one/deep/header
  cat > a/one/deep/header/deep.h <<EOF
#define B2 4
EOF

  bazel build -s --verbose_failures --experimental_enable_implicit_headermaps //a || {
    find bazel-out -type f -name \*.hmap
    fail "build failed"
  }
}

run_suite "Tests for Bazel's Objc rules"
