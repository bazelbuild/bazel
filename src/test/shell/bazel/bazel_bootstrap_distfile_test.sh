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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-01-15, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

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

    export BAZEL_DEV_VERSION_OVERRIDE="7.7.0"
    env EXTRA_BAZEL_ARGS="--tool_java_runtime_version=local_jdk" ./compile.sh \
        || fail "Expected to be able to bootstrap bazel. If you updated MODULE.bazel, see the NOTE in that file."

    ./output/bazel \
      --server_javabase=$JAVABASE --host_jvm_args=--add-opens=java.base/java.nio=ALL-UNNAMED \
      version --nognu_format &> "${TEST_log}" \
      || fail "Generated bazel not working"
    expect_log "${SOURCE_DATE_EPOCH}"

    # TODO: make the build work without network and check for success
    # the repo cache is currently missing:
    # 1) canonical IDs
    # 2) remote jdks & java_tools (if not using a custom java_toolchain)
    ./output/bazel \
      --server_javabase=$JAVABASE --host_jvm_args=--add-opens=java.base/java.nio=ALL-UNNAMED \
      build --nobuild --repository_cache=derived/repository_cache \
      //src:bazel_nojdk &> "${TEST_log}" || true
    expect_not_log "FATAL: bazel crashed due to an internal error"
}

run_suite "bootstrap test"
