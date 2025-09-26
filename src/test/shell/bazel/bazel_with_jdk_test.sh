#!/usr/bin/env bash
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
# Tests detection of local JDK and that Bazel executes with a bundled JDK.
#

set -euo pipefail
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

javabase="$1"
if [[ $javabase = external/* ]]; then
  javabase=${javabase#external/}
fi
javabase="$(rlocation "${javabase}/bin/java${EXE_EXT}")"
javabase=${javabase%/bin/java${EXE_EXT}}

function bazel() {
  $(rlocation io_bazel/src/bazel) --bazelrc=$TEST_TMPDIR/bazelrc "$@"
  return $?
}

function set_up() {
  # TODO(philwo) remove this when the testenv improvement change is in
  if is_windows; then
    export PATH=/c/python_27_amd64/files:$PATH
    EXTRA_BAZELRC="build --cpu=x64_windows_msvc"
    setup_bazelrc
  fi

  add_rules_java "MODULE.bazel"

  mkdir -p java/main
  cat >java/main/BUILD <<EOF
load("@rules_java//java:java_binary.bzl", "java_binary")
java_binary(
    name = 'JavaExample',
    srcs = ['JavaExample.java'],
    main_class = 'JavaExample',
)
EOF

  cat >java/main/JavaExample.java <<EOF
public class JavaExample {
}
EOF

  # ... but ensure JAVA_HOME is set, so we can find a default local jdk
  export JAVA_HOME="${javabase}"
}

function test_bazel_uses_bundled_jdk() {
  bazel --batch info &> "$TEST_log" || fail "bazel info failed"
  install_base="$(bazel --batch info install_base)"

  # Case-insensitive match, because Windows paths are case-insensitive.
  grep -sqi -- "^java-home: ${install_base}/embedded_tools/jdk" $TEST_log || \
      fail "bazel's java-home is not inside the install base"
}

# Tests that "bazel license" prints the license of the bundled JDK by grepping for
# representative strings from those files. If this test breaks after upgrading the version of the
# bundled JDK, the strings may have to be updated.
function test_bazel_license_prints_jdk_license() {
  bazel --batch license \
      &> "$TEST_log" || fail "running bazel license failed"

  expect_log "OPENJDK ASSEMBLY EXCEPTION" || \
      fail "'bazel license' did not print an expected string from ASSEMBLY_EXCEPTION"

  expect_log "Provided you have not received the software directly from Azul and have already" || \
      fail "'bazel license' did not print an expected string from DISCLAIMER"

  expect_log '"CLASSPATH" EXCEPTION TO THE GPL' || \
      fail "'bazel license' did not print an expected string from LICENSE"
}

# JVM selection: Do not automatically use remote JDK for execution JVM if local
# JDK is not found. Print an error message guiding the user how to use remote JDK.
# Rationale: Keeping build systems stable upon Bazel releases.
function test_bazel_reports_missing_local_jdk() {
  # Make a JAVA_HOME with javac and without java
  # This fails discovery on systems that rely on JAVA_HOME, rely on PATH and
  # also on Darwin that is using /usr/libexec/java_home for discovery

  mkdir bin
  touch bin/javac
  chmod +x bin/javac
  export JAVA_HOME="$PWD"
  export PATH="$PWD/bin:$PATH"

  bazel build \
      --java_runtime_version=local_jdk \
       java/main:JavaExample &>"${TEST_log}" \
      && fail "build with missing local JDK should have failed" || true
  expect_log "Auto-Configuration Error: Cannot find Java binary"
}

# Bazel shall detect JDK version and configure it with "local_jdk_{version}" and "{version}" setting.
function test_bazel_detects_local_jdk_version8() {
  # Fake Java version 8
  mkdir -p jdk/bin
  touch jdk/bin/javac
  chmod +x jdk/bin/javac
  cat >jdk/bin/java <<EOF
#!/usr/bin/env bash

echo " Property settings:" >&2
echo "  java.version = 1.8.0 " >&2
EOF
  chmod +x jdk/bin/java
  export JAVA_HOME="$PWD/jdk"
  export PATH="$PWD/jdk/bin:$PATH"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=8 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 8 and selecting it failed"
  expect_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=local_jdk_8 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 8 and selecting it failed"
  expect_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=11 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 8 and selecting it failed"
  expect_not_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"
}

# Bazel shall detect JDK version and configure it with "local_jdk_{version}" and "{version}" setting.
function test_bazel_detects_local_jdk_version11() {
  # Fake Java version 11
  mkdir -p jdk/bin
  touch jdk/bin/javac
  chmod +x jdk/bin/javac
  cat >jdk/bin/java <<EOF
#!/usr/bin/env bash

echo " Property settings:" >&2
echo "  java.version = 11.0.1 " >&2
EOF
  chmod +x jdk/bin/java
  export JAVA_HOME="$PWD/jdk"
  export PATH="$PWD/jdk/bin:$PATH"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=11 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 11 and selecting it failed"
  expect_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=local_jdk_11 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 11 and selecting it failed"
  expect_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=17 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 11 and selecting it failed"
  expect_not_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"
}

# Bazel shall detect JDK version and configure it with "local_jdk_{version}" and "{version}" setting.
function test_bazel_detects_local_jdk_version11_with_only_major() {
  # Fake Java version 11
  mkdir -p jdk/bin
  touch jdk/bin/javac
  chmod +x jdk/bin/javac
  cat >jdk/bin/java <<EOF
#!/usr/bin/env bash

echo " Property settings:" >&2
echo "  java.version = 11 " >&2
EOF
  chmod +x jdk/bin/java
  export JAVA_HOME="$PWD/jdk"
  export PATH="$PWD/jdk/bin:$PATH"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=11 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 11 and selecting it failed"
  expect_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=local_jdk_11 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 11 and selecting it failed"
  expect_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"

  bazel cquery \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      --java_runtime_version=17 \
      //java/main:JavaExample &>"${TEST_log}" || fail "Autodetecting a fake JDK version 11 and selecting it failed"
  expect_not_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"
}

# Failure to detect JDK version shall be handled gracefully.
function test_bazel_gracefully_handles_unknown_java() {
  # Fake Java version 11
  mkdir -p jdk/bin
  touch jdk/bin/javac
  chmod +x jdk/bin/javac
  cat >jdk/bin/java <<EOF
#!/usr/bin/env bash

echo " Property settings:" >&2
echo "  java.version = xxx.superfuture.version " >&2
EOF
  chmod +x jdk/bin/java
  export JAVA_HOME="$PWD/jdk"
  export PATH="$PWD/jdk/bin:$PATH"

  bazel cquery \
      --java_runtime_version=local_jdk \
      --toolchain_resolution_debug=tools/jdk:runtime_toolchain_type \
      //java/main:JavaExample &>"${TEST_log}" \
      || fail "Failed to resolve Java toolchain when version cannot be detected"
  expect_log "@@bazel_tools//tools/jdk:runtime_toolchain_type -> toolchain @@rules_java+.*+toolchains+local_jdk//:jdk"
}

# Bazel shall provide Java compilation toolchains that use local JDK.
function test_bazel_compiles_with_localjdk() {
  cat >> MODULE.bazel <<EOF
java_toolchains = use_extension("@rules_java//java:extensions.bzl", "toolchains")
use_repo(java_toolchains, "local_jdk")
EOF

  bazel aquery '//java/main:JavaExample' --extra_toolchains=@local_jdk//:all &>"${TEST_log}" \
      || fail "Failed to use extra toolchains provided by @local_jdk repository."

  expect_log "exec external/rules_java++toolchains+local_jdk/bin/java"
  expect_not_log "remotejdk11_linux/bin/java"
}

run_suite "Tests detection of local JDK and that Bazel executes with a bundled JDK."
