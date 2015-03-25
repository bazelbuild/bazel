#!/bin/bash
#
# Copyright 2015 Google Inc. All rights reserved.
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
# Setup bazel for integration tests
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/testenv.sh \
  || { echo "testenv.sh not found!" >&2; exit 1; }

# OS X as a limit in the pipe length, so force the root to a shorter one
bazel_root="${TEST_TMPDIR}/root"
mkdir -p "${bazel_root}"

bazel_javabase="${jdk_dir}"
bazel="${bazel_path}/bazel --output_user_root=${bazel_root} --host_javabase=${bazel_javabase}"

echo "bazel binary is at $bazel"

# Here we unset variable that were set by the invoking Blaze instance
unset JAVA_RUNFILES

function bazel() {
  ${bazel} "$@"
}

function setup_protoc_support() {
  mkdir -p third_party
  [ -e third_party/protoc ] || ln -s ${protoc_compiler} third_party/protoc
  [ -e third_party/protobuf-java.jar ] \
    || ln -s ${protoc_jar} third_party/protobuf-java.jar

cat <<EOF > third_party/BUILD
package(default_visibility = ["//visibility:public"])
exports_files(["protoc"])
filegroup(
  name = "protobuf",
  srcs = [ "protobuf-java.jar"])

EOF
}

function setup_javatest_common() {
  # TODO(bazel-team): we should use remote repositories.
  mkdir -p third_party
  if [ ! -f third_party/BUILD ]; then
    cat <<EOF >third_party/BUILD
package(default_visibility = ["//visibility:public"])
EOF
  fi

  [ -e third_party/junit.jar ] || ln -s ${junit_jar} third_party/junit.jar
  [ -e third_party/hamcrest.jar ] \
    || ln -s ${hamcrest_jar} third_party/hamcrest.jar
}

function setup_javatest_support() {
  setup_javatest_common
  cat <<EOF >>third_party/BUILD
java_import(
    name = "junit4",
    jars = [
        "junit.jar",
        "hamcrest.jar",
    ],
)
EOF
}

function setup_skylark_javatest_support() {
  setup_javatest_common
  cat <<EOF >>third_party/BUILD
load("/tools/build_rules/java_rules_skylark", "java_library")

java_library(
    name = "junit4-skylark",
    jars = [
        "junit.jar",
        "hamcrest.jar",
    ],
)
EOF
}

workspaces=()
# Set-up a new, clean workspace with only the tools installed.
function create_new_workspace() {
  set -e
  new_workspace_dir=${1:-$(mktemp -d ${TEST_TMPDIR}/workspace.XXXXXXXX)}
  mkdir -p ${new_workspace_dir}
  workspaces+=(${new_workspace_dir})
  cd ${new_workspace_dir}
  mkdir tools

  copy_tools_directory

  ln -s "${javabuilder_path}" tools/jdk/JavaBuilder_deploy.jar
  ln -s "${singlejar_path}"  tools/jdk/SingleJar_deploy.jar
  ln -s "${ijar_path}" tools/jdk/ijar

  if [[ -d ${jdk_dir} ]] ; then
    ln -s ${jdk_dir} tools/jdk/jdk
  fi

  touch WORKSPACE
}

# Set-up a clean default workspace.
function setup_clean_workspace() {
  export WORKSPACE_DIR=${TEST_TMPDIR}/workspace
  echo "setting up client in ${WORKSPACE_DIR}"
  create_new_workspace ${WORKSPACE_DIR}
  [ "${new_workspace_dir}" = "${WORKSPACE_DIR}" ] || \
    { echo "Failed to create workspace" >&2; exit 1; }
  export BAZEL_INSTALL_BASE=$(bazel info install_base)
  export BAZEL_GENFILES_DIR=$(bazel info bazel-genfiles)
}

# Clean-up all files that are not in tools or third_party to
# restart from a clean workspace
function cleanup_workspace() {
  if [ -d "${WORKSPACE_DIR:-}" ]; then
    echo "Cleaning up workspace"
    cd ${WORKSPACE_DIR}
    bazel clean  # Cleanup the output base

    for i in $(ls); do
      if [ "$i" != '*' -a "$i" != "tools" -a "$i" != "third_party" ]; then
        rm -fr "$i"
      fi
    done
    touch WORKSPACE
  fi
  for i in ${workspaces}; do
    if [ "$i" != "${WORKSPACE_DIR:-}" ]; then
      rm -fr $i
    fi
  done
  workspaces=()
}

# Clean-up the bazel install base
function cleanup() {
  if [ -d "${BAZEL_INSTALL_BASE:-__does_not_exists__}" ]; then
    rm -fr "${BAZEL_INSTALL_BASE}"
  fi
}

function tear_down() {
  cleanup_workspace
}

#
# Simples assert to make the tests more readable
#
function assert_build() {
  bazel build -s $1 || fail "Failed to build $1"

  if [ -n "${2:-}" ]; then
    test -f "$2" || fail "Output $2 not found for target $1"
  fi
}

function assert_test_ok() {
  bazel test --test_output=errors $1 \
    || fail "Test $1 failed while expecting success"
}

function assert_test_fails() {
  bazel test --test_output=errors $1 >& $TEST_log \
    && fail "Test $1 succeed while expecting failure" \
    || true
  expect_log "$1.*FAILED"
}

function assert_binary_run() {
  $1 >& $TEST_log || fail "Failed to run $1"
  [ -z "${2:-}" ] || expect_log "$2"
}

function assert_bazel_run() {
  bazel run $1 >& $TEST_log || fail "Failed to run $1"
    [ -z "${2:-}" ] || expect_log "$2"

  assert_binary_run "./bazel-bin/$(echo "$1" | sed 's|^//||' | sed 's|:|/|')" "${2:-}"
}

setup_clean_workspace
