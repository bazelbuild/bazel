#!/usr/bin/env bash
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
# Integration tests for java resources packaging.

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

add_to_bazelrc "build --package_path=%workspace%"

function set_up() {
  add_rules_java MODULE.bazel
}

function assert_resource_path() {
  local -r pkg="$1"
  local -r resource_dir="$2"
  local -r dir_in_jar="$3"

  local -r resource_label="//${resource_dir}:hello.txt"
  mkdir -p "$pkg/java/deploy" || fail "mkdir $pkg/java/deploy"
  cat >"$pkg/java/deploy/BUILD" <<EOF
load("@rules_java//java:java_binary.bzl", "java_binary")
java_binary(name = 'Hello',
            srcs = ['Hello.java'],
            main_class = 'hello.Hello',
            resources = ['$resource_label'])
EOF

  cat >"$pkg/java/deploy/Hello.java" <<EOF
package deploy;
public class Hello {
  public static void main(String[] args) { }
}
EOF

  mkdir -p "$resource_dir" || fail "mkdir"
  echo 'exports_files(["hello.txt"])' >"${resource_dir}/BUILD"
  echo 'hello' >"${resource_dir}/hello.txt"

  bazel build "//$pkg/java/deploy:Hello.jar" || fail "build failed"
  local -r entry="$(unzip -l "${PRODUCT_NAME}-bin/$pkg/java/deploy/Hello.jar" \
      | tr -d '\r' | awk '$4 ~ /hello.txt/ {print $4}')"
  [[ -n "$entry" ]] || fail "did not find hello.txt in JAR"
  assert_equals "$entry" "${dir_in_jar}/hello.txt"
}

# Due to b/31452301 the resource's path be stripped of "java/"
function test_resource_prefix_strip_under_root_java_tree() {
  local -r pkg="${FUNCNAME[0]}"
  assert_resource_path "$pkg" "java/hello" "hello"
}

# Due to b/31452301 the resource's path won't be stripped of "pkg/java/" in Blaze,
# while Bazel's OSS rules_java strips any ".../java/" prefix by default.
function test_resource_prefix_strip_under_alternative_java_tree() {
  local -r pkg="${FUNCNAME[0]}"
  if [[ "$PRODUCT_NAME" == "bazel" ]]; then
    assert_resource_path "$pkg" "$pkg/java/hello" "hello"
  else
    assert_resource_path "$pkg" "$pkg/java/hello" "$pkg/java/hello"
  fi
}

# Control test to see that the resource's path won't be stripped of "pkg/blah/"
function test_resource_prefix_strip_under_non_java_tree() {
  local -r pkg="${FUNCNAME[0]}"
  assert_resource_path "$pkg" "$pkg/blah/hello" "$pkg/blah/hello"
}

run_suite "Java integration tests"
