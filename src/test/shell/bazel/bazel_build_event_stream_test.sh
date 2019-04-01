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
# An end-to-end test for bazel-specific parts of the build-event stream.

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

#### SETUP #############################################################

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

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

function set_up() {
  mkdir -p pkg
  touch remote_file
  if $is_windows; then
    # Windows needs "file:///c:/foo/bar".
    FILE_URL="file:///$(cygpath -m "$PWD")/remote_file"
  else
    # Non-Windows needs "file:///foo/bar".
    FILE_URL="file://${PWD}/remote_file"
  fi

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")
http_file(name="remote", urls=["${FILE_URL}"])
EOF
  cat > pkg/BUILD <<'EOF'
genrule(
  name="main",
  srcs=["@remote//file"],
  outs = ["main.out"],
  cmd = "cp $< $@",
)
EOF
}

#### TESTS #############################################################


function test_fetch_test() {
  # We expect the "fetch" command to generate at least a minimally useful
  # build-event stream.
  bazel clean --expunge
  rm -f "${TEST_log}"
  bazel fetch --build_event_text_file="${TEST_log}" //pkg:main \
      || fail "bazel fetch failed"
  [ -f "${TEST_log}" ] \
      || fail "fetch did not generate requested build-event file"
  expect_log '^started'
  expect_log '^finished'
  expect_log 'name: "SUCCESS"'
  expect_log 'uuid: "'
  expect_log '^fetch'
  # on second attempt, the fetched file should already be cached.
  bazel shutdown
  rm -f "${TEST_log}"
  bazel fetch --build_event_text_file="${TEST_log}" //pkg:main \
      || fail "bazel fetch failed"
  [ -f "${TEST_log}" ] \
      || fail "fetch did not generate requested build-event file"
  expect_log '^started'
  expect_log '^finished'
  expect_log 'name: "SUCCESS"'
  expect_log 'uuid: "'
  expect_not_log '^fetch'
}

function test_fetch_in_build() {
  # We expect a fetch that happens as a consequence of a build to be reported.
  bazel clean --expunge
  bazel build --build_event_text_file="${TEST_log}" //pkg:main \
      || fail "bazel build failed"
  expect_log 'name: "SUCCESS"'
  expect_log '^fetch'
  bazel shutdown
  bazel build --build_event_text_file="${TEST_log}" //pkg:main \
      || fail "bazel build failed"
  expect_log 'name: "SUCCESS"'
  expect_not_log '^fetch'
}

function test_query() {
  # Verify that at least a minimally meaningful event stream is generated
  # for non-build. In particular, we expect bazel not to crash.
  bazel query --build_event_text_file=$TEST_log  //pkg:main \
    || fail "bazel query failed"
  expect_log '^started'
  expect_log 'command: "query"'
  expect_log 'args: "--build_event_text_file='
  expect_log 'build_finished'
  expect_not_log 'aborted'
  expect_log '^finished'
  expect_log 'name: "SUCCESS"'
  expect_log 'last_message: true'
}

run_suite "Bazel-specific integration tests for the build-event stream"
