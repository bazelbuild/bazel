#!/usr/bin/env bash
#
# Copyright 2012 The Bazel Authors. All rights reserved.
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
# incremental_analysis_integration_test.sh: a test for incremental analysis.
#

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

function set_up() {
  cd "${WORKSPACE_DIR}"
  rm -rf cc ticktock hello a
  add_rules_cc "MODULE.bazel"
  add_rules_java "MODULE.bazel"
  add_rules_shell "MODULE.bazel"
}

function write_hello_world_files() {
  mkdir -p hello || fail "mkdir hello failed"
  cat >hello/BUILD <<EOF
load("@rules_java//java:java_binary.bzl", "java_binary")
java_binary(
  name = 'hello',
  srcs = ['Hello.java'],
  main_class = 'hello.Hello',
)
EOF

  cat >hello/Hello.java <<EOF
package hello;
public class Hello {
  public static void main(String[] args) {
    System.out.println("Incrementally hello!");
  }
}
EOF
}

#### TESTS #############################################################

function test_tick_tock() {
  mkdir -p ticktock
  cat > ticktock/source.cc <<EOF
int main(void) {
  return 0;
}
EOF

  cat > ticktock/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
cc_binary(name='tick', srcs=['source.cc'])
cc_binary(name='tock', srcs=['source.cc'])
EOF

  bazel build //ticktock:tick || fail "Build failed"
  bazel build //ticktock:tock || fail "Build failed"
  bazel build //ticktock:tick || fail "Build failed"
}

function test_null_build_and_shared_actions() {
  mkdir -p cc
  DEPS=""
  cat > cc/BUILD <<EOF
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
EOF
  for i in $(seq 1 100); do
    echo "cc_binary(name='bin_${i}', srcs=['cc.cc'])" >> cc/BUILD
    DEPS="$DEPS ':bin_${i}',"
  done

  echo "sh_binary(name='sh', srcs=['sh.sh'], data=[${DEPS}])" >> cc/BUILD

  cat > cc/cc.cc <<EOF
int main(void) {
  return 0;
}
EOF

  touch cc/sh.sh
  chmod +x cc/sh.sh

  # Repeat this a couple of times -- in order for this to fail, the first
  # WriteBuildInfoHeaderAction needs to be different, which we cannot
  # guarantee. Therefore, we just do this a couple of times to make the chance
  # of getting a false negative very small.
  bazel build //cc:sh || fail "build failed"
  bazel build //cc:sh || fail "build failed"
  bazel build //cc:sh || fail "build failed"
  bazel build //cc:sh || fail "build failed"
  bazel build //cc:sh || fail "build failed"
}

function run_hello_binary() {
  (
    unset JAVA_RUNFILES TEST_SRCDIR RUNFILES_MANIFEST_FILE RUNFILES_MANIFEST_ONLY RUNFILES_DIR
    "${PRODUCT_NAME}-bin/hello/hello" >>"$TEST_log"
  )
}

function test_compile_helloworld() {
  write_hello_world_files
  bazel build //hello:hello >&"$TEST_log" \
      || fail "Build failed"
  run_hello_binary
  expect_log 'Incrementally hello!'
  bazel build //hello:hello >&"$TEST_log" \
      || fail "Build failed"
  run_hello_binary
  expect_log 'Incrementally hello!'
}

function test_exclusive_scheduling() {
  mkdir -p a
  cat > a/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name="b", tags=["exclusive"], srcs=["d.sh"])
sh_test(name="c", srcs=["d.sh"])
EOF

  cat > a/d.sh <<EOF
#!/bin/bash
exit 0
EOF

  chmod +x a/d.sh

  bazel test //a:b //a:c >&"$TEST_log" || fail "test failed"
  bazel test //a:b >&"$TEST_log" || fail "test failed"
}

run_suite "incremental analysis integration test"
