#!/bin/bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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


##############################################################################
# Tests stripping config prefixes from output paths for better caching.
#
# What is this?
# -------------------
# Output paths are the paths of files created in a build. For example:
# "bazel-out/x86-fastbuild/bin/myproj/my.output".
#
# The config prefix is the "/x86-fastbuild/" part. That includes the CPU and
# compilation mode, which means changing --cpu or --compilation_mode invalidates
# action cache hits even for actions that don't care about those values.
#
# This tests an experimental feature that strips those prefixes for such
# actions. So they run with "bazel-out/bin/myproject/my.output" and thus get
# better caching.

# --- begin runfiles.bash initialization ---
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "$(rlocation "io_bazel/src/test/shell/integration/config_stripped_outputs_lib.sh")" \
  || { echo "config_stripped_outputs_lib.sh not found!" >&2; exit 1; }


case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

add_to_bazelrc "test --notest_loasd"
add_to_bazelrc "build --package_path=%workspace%"

# This is what triggers config path stripping.
add_to_bazelrc "build --experimental_output_paths=strip"

# Remove this check when Bazel supports path mapping. This requires a complying
# executor. See https://github.com/bazelbuild/bazel/pull/18155.
function is_bazel() {
  output_path=$(bazel info | grep '^output_path:')
  bazel_out="${output_path##*/}"
  if [ $bazel_out == "bazel-out" ]; then
    # True for Bazel, false for Blaze/Google.
    return 0
  else
    return 1
  fi
}

# Tests built-in Java support for stripping config path prefixes from
# platform-independent actions.
function test_builtin_java_support() {
  # TODO(https://github.com/bazelbuild/bazel/pull/18155): support Bazel.
  if is_bazel; then return; fi

  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"
  cat > "$pkg/BUILD" <<EOF
java_library(
  name = "mylib",
  srcs = ["MyLib.java"],
)
java_binary(
  name = "mybin",
  srcs = ["MyBin.java"],
  deps = [":mylib"],
  main_class = "main.MyBin",
)
EOF

  cat > "$pkg/MyLib.java" <<EOF
package mylib;
public class MyLib {
  public static void runMyLib() {
    System.out.println("MyLib checking in.");
  }
}
EOF

  cat > "$pkg/MyBin.java" <<EOF
package main;
import mylib.MyLib;
public class MyBin {
  public static void main(String[] argv) {
    MyLib.runMyLib();
    System.out.println("MyBin running the main binary.");
  }
}
EOF

  # Verify the build succeeds:
  bazel clean
  bazel build -s "//$pkg:mybin" 2>"$TEST_log" || fail "Expected success"

  # Verify these output paths are stripped as expected:
  # java_library .jar compilation:
  assert_paths_stripped "$TEST_log" "bin/$pkg/libmylib.jar"
  # java_library header jar compilation:
  assert_paths_stripped "$TEST_log" "bin/$pkg/libmylib-hjar.jar"
  # java_binary .jar compilation:
  assert_paths_stripped "$TEST_log" "/bin/$pkg/mybin.jar"
}

function write_java_classpath_reduction_files() {
  local -r pkg="$1"
  mkdir -p "$pkg/java/hello/" || fail "Expected success"
  cat > "$pkg/java/hello/A.java" <<'EOF'
package hello;
public class A {
  public void f(B b) { b.getC().getD(); }
}
EOF
  cat > "$pkg/java/hello/B.java" <<'EOF'
package hello;
public class B {
  public C getC() { return null; }
}
EOF
  cat > "$pkg/java/hello/C.java" <<'EOF'
package hello;
public class C {
  public D getD() { return null; }
}
EOF
  cat > "$pkg/java/hello/D.java" <<'EOF'
package hello;
public class D {}
EOF
  cat > "$pkg/java/hello/BUILD" <<'EOF'
java_library(name='a', srcs=['A.java'], deps = [':b'])
java_library(name='b', srcs=['B.java'], deps = [':c'])
java_library(name='c', srcs=['C.java'], deps = [':d'])
java_library(name='d', srcs=['D.java'])
EOF
}

function test_inmemory_jdeps_support() {
  # TODO(https://github.com/bazelbuild/bazel/pull/18155): support Bazel.
  if is_bazel; then return; fi

  local -r pkg="${FUNCNAME[0]}"
  write_java_classpath_reduction_files "$pkg"

  bazel clean
  bazel build --experimental_java_classpath=bazel  \
    --experimental_output_paths=strip \
    --experimental_inmemory_jdeps_files \
    //"$pkg"/java/hello:a -s 2>"$TEST_log" \
    || fail "Expected success"

  # java_library .jar compilation:
  assert_paths_stripped "$TEST_log" "$pkg/java/hello/liba.jar-0.params"
  # java_library header jar compilation:
  assert_paths_stripped "$TEST_log" "bin/$pkg/java/hello/libb-hjar.jar"
}

# TODO(b/191411472): add a test for actions that would be stripped but aren't
# because they have inputs from multiple configurations.

run_suite "Tests stripping config prefixes from output paths for better action caching"
