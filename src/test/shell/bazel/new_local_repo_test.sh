#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-02-20, Bazel on Windows only supports MSYS Bash.
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

# Regression test for GitHub issue #6351, see
# https://github.com/bazelbuild/bazel/issues/6351#issuecomment-465488344
function test_glob_in_synthesized_build_file() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  mkdir $pkg/A || fail "mkdir $pkg/A"
  mkdir $pkg/B || fail "mkdir $pkg/B"

  cat >$pkg/A/WORKSPACE <<'eof'
new_local_repository(
    name = "B",
    build_file_content = """
filegroup(
    name = "F",
    srcs = glob(["*.txt"]),
    visibility = ["//visibility:public"],
)
""",
    path = "../B",
)
eof

  cat >$pkg/A/BUILD <<'eof'
genrule(
    name = "G",
    srcs = ["@B//:F"],
    outs = ["g.txt"],
    cmd = "echo $(SRCS) > $@",
)
eof

  echo "dummy" >$pkg/B/a.txt

  # Build 1: x.txt should contain external/B/a.txt
  cd $pkg/A
  bazel build //:G || fail "build failed"
  cat bazel-genfiles/g.txt >$TEST_log
  expect_log "external/B/a.txt"
  expect_not_log "external/B/b.txt"

  # Build 2: add B/b.txt and see if the glob picks it up.
  echo "dummy" > ../B/b.txt
  bazel build //:G || fail "build failed"
  cat bazel-genfiles/g.txt >$TEST_log
  expect_log "external/B/a.txt"
  expect_log "external/B/b.txt"
}

run_suite "new_local_repository correctness tests"
