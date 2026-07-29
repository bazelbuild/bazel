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
# Test that bazel can be compiled out of the distribution artifact.
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

if [[ -n "${1:-}" ]]; then
  declare -r DISTFILE=$(rlocation io_bazel/${1#./})
else
  fail "missing argument for DISTFILE"
fi

if [[ -n "${2:-}" ]]; then
  declare -r EMBEDDED_JDK=$(rlocation io_bazel/${2#./})
else
  declare -r EMBEDDED_JDK=""
fi

# The prerequisites documented at
# https://bazel.build/install/compile-source#bootstrap-unix-prereq ask for a JDK
# 21, so bootstrap with a hermetic JDK 21 from the runfiles rather than with
# whichever JDK happens to be installed on the machine running this test. This
# both keeps the test hermetic and verifies that a JDK 21 is in fact sufficient.
setup_javabase
export JAVA_HOME="${bazel_javabase}"
# JAVA_TOOL_OPTIONS and _JAVA_OPTIONS make the JVM print an extra line, so unset
# them for this check just like compile.sh does for the bootstrap itself.
declare -r JAVAC_VERSION="$(unset JAVA_TOOL_OPTIONS _JAVA_OPTIONS; \
  "${JAVA_HOME}/bin/javac" -version 2>&1)"
if [[ ! "${JAVAC_VERSION}" =~ ^javac\ 21(\.|$) ]]; then
  log_fatal "expected a JDK 21 in JAVA_HOME, but got '${JAVAC_VERSION}'"
fi

function test_bootstrap() {
    cd "$(mktemp -d ${TEST_TMPDIR}/bazelbootstrap.XXXXXXXX)"
    export SOURCE_DATE_EPOCH=1501234567

    case "${DISTFILE}" in
      *.zip)
        unzip -q "${DISTFILE}"
        ;;
      *.tar)
        tar xf "${DISTFILE}"
        ;;
      *)
        fail "Unknown distfile format: ${DISTFILE}"
        ;;
    esac

    case "${EMBEDDED_JDK}" in
      *.zip)
        unzip -q "$EMBEDDED_JDK"
        ;;
      *.tar.gz)
        tar zxf "$EMBEDDED_JDK"
        ;;
      *.tar)
        tar xf "$EMBEDDED_JDK"
        ;;
      *)
        fail "Unknown embedded JDK format: ${EMBEDDED_JDK}"
        ;;
    esac

    JAVABASE=$(echo reduced*)

    # TODO: Remove once rules_python exports runtime_env_toolchain_interpreter.sh
    # See https://github.com/bazel-contrib/rules_python/pull/3471
    # TODO: Enable macOS layering_check when dependencies are updated
    export EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS:-} \
      --noincompatible_no_implicit_file_export \
      --repo_env=APPLE_SUPPORT_LAYERING_CHECK_BETA=0"

    ./compile.sh || fail "Expected to be able to bootstrap bazel.\
 If you updated MODULE.bazel, see the NOTE in that file."

    ./output/bazel \
      --server_javabase=$JAVABASE --host_jvm_args=--add-opens=java.base/java.nio=ALL-UNNAMED \
      version --nognu_format &> "${TEST_log}" \
      || fail "Generated bazel not working"
    expect_log "${SOURCE_DATE_EPOCH}"

    # the repo cache is currently missing canonical IDs
    export BAZEL_HTTP_RULES_URLS_AS_DEFAULT_CANONICAL_ID=0

    JAVA_VERSION=100 # so we don't accidentally match something else

    # create a fake toolchain to avoid @remote_java_tools
    # this can't actually build anything, but we're just testing analysis
    mkdir fake_java_toolchain
    touch fake_java_toolchain/dummy.jar
    cat << EOF > fake_java_toolchain/BUILD
load("@rules_java//toolchains:default_java_toolchain.bzl", "default_java_toolchain")
default_java_toolchain(
    name = "fake_java_toolchain",
    bootclasspath = [":dummy.jar"],
    genclass = ":dummy.jar",
    header_compiler = ":dummy.jar",
    header_compiler_direct = ":dummy.jar",
    ijar = ":dummy.jar",
    jacocorunner = ":dummy.jar",
    java_runtime = "@local_jdk//:jdk",
    javabuilder = ":dummy.jar",
    oneversion = ":dummy.jar",
    singlejar = ":dummy.jar",
    source_version = "${JAVA_VERSION}",
    target_version = "${JAVA_VERSION}",
    tags = ["manual"],
    visibility = ["//visibility:public"],
)
EOF

    ./output/bazel \
      --server_javabase=$JAVABASE --host_jvm_args=--add-opens=java.base/java.nio=ALL-UNNAMED \
      build --nobuild --repository_cache=derived/repository_cache --repo_contents_cache= \
      --repo_env=APPLE_SUPPORT_LAYERING_CHECK_BETA=0 \
      --override_repository=$(cat derived/maven/MAVEN_CANONICAL_REPO_NAME)=derived/maven \
      --java_language_version=${JAVA_VERSION} --tool_java_language_version=${JAVA_VERSION} \
      --java_runtime_version=local_jdk --tool_java_runtime_version=local_jdk \
      --extra_toolchains=@rules_python//python/runtime_env_toolchains:all \
      --extra_toolchains=fake_java_toolchain:all \
      --noincompatible_no_implicit_file_export \
      src:bazel_nojdk &> "${TEST_log}" || fail "analysis with bootstrapped Bazel failed"
}

run_suite "bootstrap test"
