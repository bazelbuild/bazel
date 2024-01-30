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

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

set -eu

add_to_bazelrc "build --package_path=%workspace%"


function test_java_launcher_classpath_limit() {
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p $pkg/java/hello || fail "Expected success"
  cat > $pkg/java/hello/HelloLib.java <<EOF
package hello;
public class HelloLib {
  public static String getHello() {
    return "Hello World!";
  }
}
EOF
  cat > $pkg/java/hello/Hello.java <<EOF
package hello;
public class Hello {
  public static void main(String[] args) {
    System.out.println(HelloLib.getHello());
  }
}
EOF
  cat > $pkg/java/hello/BUILD <<EOF
java_library(
    name = "hellolib",
    srcs = ["HelloLib.java"],
)
java_binary(
    name = "hello",
    srcs = ["Hello.java"],
    deps = [":hellolib"],
    main_class = "hello.Hello",
)
java_library(
    name = "hellolib%special%lib",
    srcs = ["HelloLib.java"],
)
java_binary(
    name = "hello_special",
    srcs = ["Hello.java"],
    deps = [":hellolib%special%lib"],
    main_class = "hello.Hello",
)
EOF
  bazel build //$pkg/java/hello:hello || fail "expected success"
  ${PRODUCT_NAME}-bin/$pkg/java/hello/hello >& "$TEST_log" || \
    fail "expected success"
  expect_log "Hello World!"

  ${PRODUCT_NAME}-bin/$pkg/java/hello/hello --classpath_limit=0 >& "$TEST_log" || \
    fail "expected success"
  expect_log "Hello World!"

  bazel build //$pkg/java/hello:hello_special || fail "expected success"
  ${PRODUCT_NAME}-bin/$pkg/java/hello/hello_special >& "$TEST_log" || \
    fail "expected success"
  expect_log "Hello World!"

  ${PRODUCT_NAME}-bin/$pkg/java/hello/hello_special --classpath_limit=0 >& "$TEST_log" || \
    fail "expected success"
  expect_log "Hello World!"
}

run_suite "Java launcher tests"

