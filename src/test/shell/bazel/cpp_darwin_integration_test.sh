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

function test_osx_test_strip() {
  mkdir -p cpp/osx_test_strip
  cat > cpp/osx_test_strip/BUILD <<EOF
cc_test(
  name = "main",
  srcs = ["main.cc"],
)
EOF
  cat > cpp/osx_test_strip/main.cc <<EOF
int main() { return 0; }
EOF
  assert_build //cpp/osx_test_strip:main.stripped
  ! dsymutil -s bazel-bin/cpp/osx_test_strip/main | grep N_FUN || \
      fail "Stripping failed, debug symbols still found in the stripped binary"
}

# Regression test for https://github.com/bazelbuild/bazel/pull/12046
function test_osx_sandboxed_cc_library_build() {
  mkdir -p cpp/osx_sandboxed_cc_library_build
  cat > cpp/osx_sandboxed_cc_library_build/BUILD <<EOF
cc_library(
    name = "a",
    srcs = ["a.cc"],
)
EOF
  cat > cpp/osx_sandboxed_cc_library_build/a.cc <<EOF
void a() { }
EOF
  assert_build --spawn_strategy=sandboxed \
    --build_event_text_file=cpp/osx_sandboxed_cc_library_build/bep.json \
    //cpp/osx_sandboxed_cc_library_build:a
  grep -q -- "--spawn_strategy=sandboxed" \
    cpp/osx_sandboxed_cc_library_build/bep.json \
    || fail "Expected to see --spawn_strategy=sandboxed in BEP output"
  grep -Eo -- "--[a-z_-]strategy=[a-z_-]*" \
    cpp/osx_sandboxed_cc_library_build/bep.json \
    | grep -vq -- "--spawn_strategy=sandboxed" \
    && fail "Expected to see only --spawn_strategy=sandboxed in BEP output"
  grep -E "libtool_check_unique|No such file" bazel-out/_tmp/actions/std* \
    && fail "Missing tools in sandboxed build"
  return 0
}

# TODO: This test passes vacuously as the default Unix toolchain doesn't use
# the set_install_name feature yet.
function test_cc_test_with_explicit_install_name() {
  mkdir -p cpp
  cat > cpp/BUILD <<EOF
cc_library(
  name = "foo",
  srcs = ["foo.cc"],
  hdrs = ["foo.h"],
)
cc_shared_library(
  name = "foo_shared",
  deps = [":foo"],
)
cc_test(
  name = "test",
  srcs = ["test.cc"],
  deps = [":foo"],
  dynamic_deps = [":foo_shared"],
)
EOF
  cat > cpp/foo.h <<EOF
  int foo();
EOF
  cat > cpp/foo.cc <<EOF
  int foo() { return 0; }
EOF
  cat > cpp/test.cc <<EOF
  #include "cpp/foo.h"
  int main() {
    return foo();
  }
EOF

  bazel test --incompatible_macos_set_install_name //cpp:test || \
      fail "bazel test //cpp:test failed"
  # Ensure @rpath is correctly set in the binary.
  ./bazel-bin/cpp/test || \
      fail "//cpp:test workspace execution failed, expected return 0, got $?"
  cd bazel-bin
  ./cpp/test || \
      fail "//cpp:test execution failed, expected 0, but $?"
}

function test_cc_test_with_explicit_install_name_apple_support() {
  cat > MODULE.bazel <<EOF
bazel_dep(name = "apple_support", version = "1.16.0")
EOF

  mkdir -p cpp
  cat > cpp/BUILD <<EOF
cc_library(
  name = "foo",
  srcs = ["foo.cc"],
  hdrs = ["foo.h"],
)
cc_shared_library(
  name = "foo_shared",
  deps = [":foo"],
)
cc_test(
  name = "test",
  srcs = ["test.cc"],
  deps = [":foo"],
  dynamic_deps = [":foo_shared"],
)
EOF
  cat > cpp/foo.h <<EOF
  int foo();
EOF
  cat > cpp/foo.cc <<EOF
  int foo() { return 0; }
EOF
  cat > cpp/test.cc <<EOF
  #include "cpp/foo.h"
  int main() {
    return foo();
  }
EOF

  bazel test --incompatible_macos_set_install_name //cpp:test || \
      fail "bazel test //cpp:test failed"
  # Ensure @rpath is correctly set in the binary.
  ./bazel-bin/cpp/test || \
      fail "//cpp:test workspace execution failed, expected return 0, got $?"
  cd bazel-bin
  ./cpp/test || \
      fail "//cpp:test execution failed, expected 0, but $?"
}

run_suite "Tests for Bazel's C++ rules on Darwin"

