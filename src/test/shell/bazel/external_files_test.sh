#!/usr/bin/env bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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
# Test the local files overlay patching functionality of external repositories.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


src_only_repo_with_local_include() {
  # Generates source files in the current directory.
  mkdir src
  cat > src/main.c <<'EOF'
#include <stdio.h>
#include "src/consts/greeting.h"

int main(int argc, char **argv) {
  printf("%s\n", GREETING);
  return 0;
}
EOF
  mkdir src/consts
  cat > src/consts/greeting.h <<'EOF'
#define GREETING "Hello World"
EOF
}

build_def() {
  # Generates build and module definition files for `src_only_repo_with_local_include` in the
  # current directory.
  cat > _.MODULE <<'EOF'
module(name = "remote")
EOF
  mkdir src
  cat > _.BUILD <<'EOF'
cc_binary(
  name = "hello",
  srcs = ["main.c", "consts/greeting.h"],
)
EOF
}

test_files_repository() {
  # Verify that overlaid files are correctly positioned.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && src_only_repo_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  build_def
  touch BUILD
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "remote",
  strip_prefix = "remote",
  urls = ["file://${WRKDIR}/remote.tar"],
  files = {
    "MODULE.bazel": "//:_.MODULE",
    "src/BUILD.bazel": "//:_.BUILD",
  },
)
EOF

  bazel build @remote//src:hello || fail "Expected build to succeed"
  bazel run @remote//src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_files_repository_changed() {
  # Verify that changes to overlaid files are applied.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && src_only_repo_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  build_def
  cat > greeting.h <<'EOF'
#define GREETING "Hello World"
EOF
  touch BUILD
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "remote",
  strip_prefix = "remote",
  urls = ["file://${WRKDIR}/remote.tar"],
  files = {
    "MODULE.bazel": "//:_.MODULE",
    "src/BUILD.bazel": "//:_.BUILD",
    "src/consts/greeting.h": "//:greeting.h",
  },
)
EOF

  bazel build @remote//src:hello || fail "Expected build to succeed"
  bazel run @remote//src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
  cat > greeting.h <<'EOF'
#define GREETING "Goodbye World"
EOF
  bazel build @remote//src:hello || fail "Expected build to succeed"
  bazel run @remote//src:hello | grep 'Goodbye World' \
      || fail "Expected output 'Goodbye World'"
}

test_files_repository_collision() {
  # Verify that overlaid files overwrite existing files.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && src_only_repo_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  build_def
  touch BUILD
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "remote",
  strip_prefix = "remote",
  urls = ["file://${WRKDIR}/remote.tar"],
  files = {
    "MODULE.bazel": "//:_.MODULE",
    "src/BUILD.bazel": "//:_.BUILD",
  },
)
EOF

  bazel build @remote//src:hello || fail "Expected build to succeed"
  bazel run @remote//src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_files_repository_bad_label() {
  # Verify that labels for non-existant source files raise an error.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && src_only_repo_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  build_def
  touch BUILD
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name = "remote",
  strip_prefix = "remote",
  urls = ["file://${WRKDIR}/remote.tar"],
  files = {
    "MODULE.bazel": "//:_.MODULE",
    "src/BUILD.bazel": "//:_incorrect_.BUILD",
  },
)
EOF

  bazel build @remote//src:hello >& "${TEST_log}" && fail "Expected to fail"
  expect_log "ERROR: Input @@//:_incorrect_.BUILD does not exist"
}

test_files_module() {
  # Verify that overlaid files are correctly positioned and that `MODULE.bazel` is detected.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && src_only_repo_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  build_def
  touch BUILD
  cat >> $(setup_module_dot_bazel) <<EOF
bazel_dep(name = "remote")
archive_override(
  module_name = "remote",
  strip_prefix = "remote",
  urls = ["file://${WRKDIR}/remote.tar"],
  files = {
    "MODULE.bazel": "//:_.MODULE",
    "src/BUILD.bazel": "//:_.BUILD",
  },
)
EOF

  bazel build @remote//src:hello || fail "Expected build to succeed"
  bazel run @remote//src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

run_suite "files tests"
