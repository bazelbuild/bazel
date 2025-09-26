#!/usr/bin/env bash
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

# Regression test for GitHub issue #6351, see
# https://github.com/bazelbuild/bazel/issues/6351#issuecomment-465488344
function test_glob_in_synthesized_build_file() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  mkdir $pkg/A || fail "mkdir $pkg/A"
  mkdir $pkg/B || fail "mkdir $pkg/B"

  setup_module_dot_bazel "$pkg/A/MODULE.bazel"
  cat >$pkg/A/MODULE.bazel <<'eof'
new_local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
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

  # Build 1: g.txt should contain external/+new_local_repository+B/a.txt
  ( cd $pkg/A
    bazel build //:G
    cat bazel-genfiles/g.txt >$TEST_log
  )
  expect_log "external/+new_local_repository+B/a.txt"
  expect_not_log "external/+new_local_repository+B/b.txt"

  # Build 2: add B/b.txt and see if the glob picks it up.
  # Shut down the server afterwards so the test cleanup can remove $pkg/A.
  ( cd $pkg/A
    echo "dummy" > ../B/b.txt
    bazel build //:G || fail "build failed"
    cat bazel-genfiles/g.txt >$TEST_log
    bazel shutdown >& /dev/null
  )
  expect_log "external/+new_local_repository+B/a.txt"
  expect_log "external/+new_local_repository+B/b.txt"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/9176
function test_recursive_glob_in_new_local_repository() {
  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg/A" "$pkg/B/subdir/inner"
  touch "$pkg/B/root.txt"
  touch "$pkg/B/subdir/outer.txt"
  touch "$pkg/B/subdir/inner/inner.txt"
  setup_module_dot_bazel "$pkg/A/MODULE.bazel"
  cat >"$pkg/A/MODULE.bazel" <<eof
new_local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")
new_local_repository(
    name = "myext",
    path = "../B",
    build_file = "//:BUILD.myext",
)
eof
  touch "$pkg/A/BUILD.bazel"
  cat >"$pkg/A/BUILD.myext" <<eof
filegroup(name = "all_files", srcs = glob(["**"]))
eof

  # Shut down the server afterwards so the test cleanup can remove $pkg/A.
  ( cd "$pkg/A"
    bazel query 'deps(@myext//:all_files)' >& "$TEST_log"
    bazel shutdown >& /dev/null
  )
  expect_log '@myext//:all_files'
  expect_log '@myext//:subdir/outer.txt'
  expect_log '@myext//:subdir/inner/inner.txt'
  expect_log '@myext//:root.txt'
  expect_log '@myext//:REPO.bazel'
  expect_log '@myext//:BUILD.bazel'
}

run_suite "new_local_repository correctness tests"
