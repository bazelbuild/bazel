#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
# Test legitimate path assumptions when working with external repositories.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }


repo_with_local_include() {
  # Generate a repository, in the current working directory, with a target
  # //src:hello that includes a file via a local path.

  touch WORKSPACE
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
  cat > src/BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c", "consts/greeting.h"],
)
EOF
}

library_with_local_include() {
  # Generates a repository, in the current directory, where a target //lib:hello
  # is a library with headers that include via paths relative to the root of
  # that repository

  touch WORKSPACE
  mkdir lib
  cat > lib/lib.h <<'EOF'
#include "lib/constants.h"

int greet(char *);

EOF
  cat > lib/constants.h <<'EOF'
#define TARGET "World"
EOF
  cat > lib/lib.c <<'EOF'
int greet(char *s) {
  printf("Hello %s\n", s);
  return 0;
}
EOF
  cat > lib/BUILD <<EOF
cc_library(
  name="lib",
  srcs=["lib.c"],
  hdrs=["lib.h", "constants.h"],
  visibility = ["//visibility:public"],
)
EOF
}


test_local_paths_main () {
  # Verify that a target in the main repository may refer to a truely source
  # file in its own repository by a path relative to the repository root.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  cd main
  repo_with_local_include

  bazel build //src:hello || fail "Expected build to succeed"
  bazel run //src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_local_paths_remote() {
  # Verify that a target in an external repository may refer to a truely source
  # file in its own repository by a path relative to the root of that repository
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && repo_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remote",
  strip_prefix="remote",
  urls=["file://${WRKDIR}/remote.tar"],
)
EOF

  bazel build @remote//src:hello || fail "Expected build to succeed"
  bazel run @remote//src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_lib_paths_main() {
  # Verify that libaries from the main repostiory can be used via include
  # path relative to their repository root and that they may refer to other
  # truely source files from the same libary via paths relative to their
  # repository root.

  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  cd main
  library_with_local_include

  touch WORKSPACE
  cat > main.c <<'EOF'
#include "lib/lib.h"

int main(int argc, char **argv) {
  greet(TARGET);
  return 0;
}
EOF
  cat > BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c"],
  deps=["//lib:lib"],
)
EOF

  bazel build //:hello || fail "Expected build to succeed"
  bazel run //:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_lib_paths_remote() {
  # Verify that libaries from an external repository can be used via include
  # path relative to their repository root and that they may refer to other
  # truely source files from the same libary via paths relative to their
  # repository root.

  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && library_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remote",
  strip_prefix="remote",
  urls=["file://${WRKDIR}/remote.tar"],
)
EOF
  cat > main.c <<'EOF'
#include "lib/lib.h"

int main(int argc, char **argv) {
  greet(TARGET);
  return 0;
}
EOF
  cat > BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c"],
  deps=["@remote//lib:lib"],
)
EOF

  bazel build //:hello || fail "Expected build to succeed"
  bazel run //:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_lib_paths_all_remote() {
  # Verify that libaries from an external repository can be used by another
  # external repository via include path relative to their repository root and
  # that they may refer to other truely source files from the same libary via
  # paths relative to their repository root.

  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remotelib
  (cd remotelib && library_with_local_include)
  tar cvf remotelib.tar remotelib
  rm -rf remotelib

  mkdir remotemain
  (cd remotemain
  cat > main.c <<'EOF'
#include "lib/lib.h"

int main(int argc, char **argv) {
  greet(TARGET);
  return 0;
}
EOF
  cat > BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c"],
  deps=["@remotelib//lib:lib"],
)
EOF
)
  tar cvf remotemain.tar remotemain
  rm -rf remotemain

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remotelib",
  strip_prefix="remotelib",
  urls=["file://${WRKDIR}/remotelib.tar"],
)
http_archive(
  name="remotemain",
  strip_prefix="remotemain",
  urls=["file://${WRKDIR}/remotemain.tar"],
)
EOF
  bazel build @remotemain//:hello || fail "Expected build to succeed"
  bazel run @remotemain//:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

run_suite "path tests for multiple repositories"
