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
# Tests for Bazel's C++ rules on Darwin

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if ! is_darwin; then
  echo "This test suite requires running on Darwin. But now is ${PLATFORM}" >&2
  exit 0
fi

function test_osx_cc_wrapper_rpaths_handling() {
  mkdir -p cpp/rpaths
  cat > cpp/rpaths/BUILD <<EOF
cc_library(
  name = "foo",
  srcs = ["foo.cc"],
)
cc_binary(
  name = "libbar.so",
  srcs = [ "bar.cc" ],
  linkshared = 1,
)
cc_binary(
  name = "libbaz.dylib",
  srcs = [ "baz.cc" ],
  linkshared = 1,
)
cc_test(
  name = "test",
  srcs = [ "test.cc", ":libbar.so", ":libbaz.dylib"  ],
  deps = [":foo"],
)
EOF
  cat > cpp/rpaths/foo.cc <<EOF
  int foo() { return 2; }
EOF
  cat > cpp/rpaths/bar.cc <<EOF
  int bar() { return 12; }
EOF
  cat > cpp/rpaths/baz.cc <<EOF
  int baz() { return 42; }
EOF
  cat > cpp/rpaths/test.cc <<EOF
  int foo();
  int bar();
  int baz();
  int main() {
    int result = foo() + bar() + baz();
    if (result == 56) {
      return 0;
    } else {
      return result;
    }
  }
EOF
  assert_build //cpp/rpaths:test
  # Paths originally hardcoded in the binary assume workspace directory. Let's change the
  # directory and execute the binary to test whether the paths in the binary have been
  # updated to use @loader_path.
  cd bazel-bin
  ./cpp/rpaths/test || \
      fail "//cpp/rpaths:test execution failed, expected to return 0, but got $?"
}

function test_osx_binary_strip() {
  mkdir -p cpp/osx_binary_strip
  cat > cpp/osx_binary_strip/BUILD <<EOF
cc_binary(
  name = "main",
  srcs = ["main.cc"],
)
EOF
  cat > cpp/osx_binary_strip/main.cc <<EOF
int main() { return 0; }
EOF
  assert_build //cpp/osx_binary_strip:main.stripped
  ! dsymutil -s bazel-bin/cpp/osx_binary_strip/main | grep N_FUN || \
      fail "Stripping failed, debug symbols still found in the stripped binary"
}

run_suite "Tests for Bazel's C++ rules on Darwin"

