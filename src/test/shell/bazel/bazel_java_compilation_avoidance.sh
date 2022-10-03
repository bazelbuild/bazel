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
# Tests the examples provided in Bazel
#

# --- begin runfiles.bash initialization ---
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

JAVA_TOOLCHAIN="@bazel_tools//tools/jdk:toolchain"

JAVA_TOOLS_ZIP="$1"; shift
if [[ "${JAVA_TOOLS_ZIP}" != "released" ]]; then
  if [[ "${JAVA_TOOLS_ZIP}" == file* ]]; then
    JAVA_TOOLS_ZIP_FILE_URL="${JAVA_TOOLS_ZIP}"
  elif "$is_windows"; then
    JAVA_TOOLS_ZIP_FILE_URL="file:///$(rlocation io_bazel/$JAVA_TOOLS_ZIP)"
  else
    JAVA_TOOLS_ZIP_FILE_URL="file://$(rlocation io_bazel/$JAVA_TOOLS_ZIP)"
  fi
fi
JAVA_TOOLS_ZIP_FILE_URL=${JAVA_TOOLS_ZIP_FILE_URL:-}

JAVA_TOOLS_PREBUILT_ZIP="$1"; shift
if [[ "${JAVA_TOOLS_PREBUILT_ZIP}" != "released" ]]; then
  if [[ "${JAVA_TOOLS_PREBUILT_ZIP}" == file* ]]; then
    JAVA_TOOLS_PREBUILT_ZIP_FILE_URL="${JAVA_TOOLS_PREBUILT_ZIP}"
  elif "$is_windows"; then
    JAVA_TOOLS_PREBUILT_ZIP_FILE_URL="file:///$(rlocation io_bazel/$JAVA_TOOLS_PREBUILT_ZIP)"
  else
    JAVA_TOOLS_PREBUILT_ZIP_FILE_URL="file://$(rlocation io_bazel/$JAVA_TOOLS_PREBUILT_ZIP)"
  fi
  # Remove the repo overrides that are set up for some Bazel CI workers.
  inplace-sed "/override_repository=remote_java_tools=/d" "$TEST_TMPDIR/bazelrc"
  inplace-sed "/override_repository=remote_java_tools_linux=/d" "$TEST_TMPDIR/bazelrc"
  inplace-sed "/override_repository=remote_java_tools_windows=/d" "$TEST_TMPDIR/bazelrc"
  inplace-sed "/override_repository=remote_java_tools_darwin=/d" "$TEST_TMPDIR/bazelrc"
fi
JAVA_TOOLS_PREBUILT_ZIP_FILE_URL=${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL:-}

if [[ $# -gt 0 ]]; then
  JAVA_LANGUAGE_VERSION="$1"; shift
  add_to_bazelrc "build --java_language_version=${JAVA_LANGUAGE_VERSION}"
  add_to_bazelrc "build --tool_java_language_version=${JAVA_LANGUAGE_VERSION}"
fi


if [[ $# -gt 0 ]]; then
  JAVA_RUNTIME_VERSION="$1"; shift
  add_to_bazelrc "build --java_runtime_version=${JAVA_RUNTIME_VERSION}"
  add_to_bazelrc "build --tool_java_runtime_version=${JAVA_RUNTIME_VERSION}"
  if [[ "${JAVA_RUNTIME_VERSION}" == 8 ]]; then
    JAVA_TOOLCHAIN="@bazel_tools//tools/jdk:toolchain_java8"
  elif [[ "${JAVA_RUNTIME_VERSION}" == 11 ]]; then
    JAVA_TOOLCHAIN="@bazel_tools//tools/jdk:toolchain_java11"
  else
    JAVA_TOOLCHAIN="@bazel_tools//tools/jdk:toolchain_jdk_${JAVA_RUNTIME_VERSION}"
  fi
fi

export TESTENV_DONT_BAZEL_CLEAN=1

function set_up() {
  cat >>WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
# java_tools versions only used to test Bazel with various JDK toolchains.
EOF

  if [[ ! -z "${JAVA_TOOLS_ZIP_FILE_URL}" ]]; then
    cat >>WORKSPACE <<EOF
http_archive(
    name = "remote_java_tools",
    urls = ["${JAVA_TOOLS_ZIP_FILE_URL}"]
)
http_archive(
    name = "remote_java_tools_linux",
    urls = ["${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL}"]
)
http_archive(
    name = "remote_java_tools_windows",
    urls = ["${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL}"]
)
http_archive(
    name = "remote_java_tools_darwin",
    urls = ["${JAVA_TOOLS_PREBUILT_ZIP_FILE_URL}"]
)
EOF
  fi

  cat $(rlocation io_bazel/src/test/shell/bazel/testdata/jdk_http_archives) >> WORKSPACE
}

function tear_down() {
  rm -rf "$(bazel info bazel-bin)/java"
}

function write_project_files() {
  mkdir -p java/libA
  cat >java/libA/BUILD <<EOF
package(default_visibility=['//visibility:public'])
java_library(name = 'libA', srcs = ['A.java'], deps = ['//java/libC']);
EOF

  cat >java/libA/A.java <<EOF
package compilation_avoidance;
public class A {
}
EOF

  mkdir -p java/libB
  cat >java/libB/BUILD <<EOF
package(default_visibility=['//visibility:public'])
java_library(name = 'libB', srcs = ['B.java'], deps = ['//java/libC']);
EOF

  cat >java/libB/B.java <<EOF
package compilation_avoidance;

public class B {
  public void func() {
    new Used();
  }
}
EOF

mkdir -p java/libC
  cat >java/libC/BUILD <<EOF
package(default_visibility=['//visibility:public'])
java_library(name = 'libC', srcs = ['Used.java', 'Unused.java']);
EOF

  cat >java/libC/Used.java <<EOF
package compilation_avoidance;
public class Used {
  public int myVar = 0;
}
EOF

  cat >java/libC/Unused.java <<EOF
package compilation_avoidance;
public class Unused {
  public int myVar = 0;
}
EOF
}

# Java lib A depends on Java lib C, but no class from C are not used.
#   -> ABI change of C will *not* recompile A.
function test_unused_java_dependency() {
  write_project_files
  bazel build //java/libA &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_dependencies || fail "Expected to build"

  inplace-sed "s/myVar/myVar_/g" "java/libC/Used.java"
  bazel build //java/libA &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_dependencies || fail "Expected to build"
  expect_log " processes: .* 1 worker"
}

# Java lib B depends on Java lib C, and uses classes from B.
#   -> ABI change of C will recompile B.
function test_used_java_dependency() {
  write_project_files
  bazel build //java/libB &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_dependencies || fail "Expected to build"

  inplace-sed "s/myVar/myVar_/g" "java/libC/Used.java"
  bazel build //java/libB &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_dependencies || fail "Expected to build"
  expect_log " processes: .* 2 worker"
}

# Java lib B depends on Java lib C, and uses class 'UsedClass' from Java lib C.
#   -> ABI change to class 'UsedClass' will recompile B.
function test_used_java_class() {
  write_project_files
  bazel build //java/libB &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_classes --experimental_track_class_usage || fail "Expected to build"

  inplace-sed "s/myVar/myVar_/g" "java/libC/Used.java"
  bazel build //java/libB &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_classes --experimental_track_class_usage || fail "Expected to build"
  expect_log " processes: .* 2 worker"
}

# Java lib B depends on Java lib C, and use class 'UsedClass' from Java lib C.
#   -> ABI change to class 'UnusedClass' from C will *not* recompile B.
function test_unused_java_class() {
  write_project_files
  bazel build //java/libB &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_classes --experimental_track_class_usage || fail "Expected to build"

  inplace-sed "s/myVar/myVar_/g" "java/libC/Unused.java"
  bazel build //java/libB &>"${TEST_log}" --nojava_header_compilation --experimental_action_input_usage_tracker=unused_classes --experimental_track_class_usage || fail "Expected to build"
  expect_log " processes: .* 1 worker"
}

run_suite "Java compilation avoidance integration tests"