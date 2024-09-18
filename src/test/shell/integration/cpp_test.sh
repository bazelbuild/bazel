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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if ! type try_with_timeout >&/dev/null; then
  # Bazel's testenv.sh defines try_with_timeout but the Google-internal version
  # uses a different testenv.sh.
  function try_with_timeout() { $* ; }
fi

#### TESTS #############################################################

function test_no_rebuild_on_irrelevant_header_change() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=["b"])
cc_library(name="b", srcs=["b1.h", "b2.h"])
EOF

  cat > $pkg/a.cc <<EOF
#include "$pkg/b1.h"

int main(void) {
  return B_RETURN_VALUE;
}
EOF

  cat > $pkg/b1.h <<EOF
#define B_RETURN_VALUE 31
EOF

  cat > $pkg/b2.h <<EOF
=== BANANA ===
EOF

  bazel build //$pkg:a || fail "build failed"
  echo "CHERRY" > $pkg/b2.h
  bazel build //$pkg:a >& $TEST_log || fail "build failed"
  expect_not_log "Compiling $pkg/a.cc"
}

function test_new_header_is_required() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=[":b"])
cc_library(name="b", srcs=["b1.h", "b2.h"])
EOF

  cat > $pkg/a.cc << EOF
#include "$pkg/b1.h"

int main(void) {
    return B1;
}
EOF

  cat > $pkg/b1.h <<EOF
#define B1 3
EOF

  cat > $pkg/b2.h <<EOF
#define B2 4
EOF

  bazel build //$pkg:a || fail "build failed"
  cat > $pkg/a.cc << EOF
#include "$pkg/b1.h"
#include "$pkg/b2.h"

int main(void) {
    return B1 + B2;
}
EOF

  bazel build //$pkg:a || fail "build failed"
}

function test_no_recompile_on_shutdown() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
cc_binary(name="a", srcs=["a.cc"], deps=["b"])
cc_library(name="b", includes=["."], hdrs=["b.h"])
EOF

  cat > $pkg/a.cc <<EOF
#include "b.h"

int main(void) {
  return B_RETURN_VALUE;
}
EOF

  cat > $pkg/b.h <<EOF
#define B_RETURN_VALUE 31
EOF

  bazel build -s //$pkg:a >& $TEST_log || fail "build failed"
  expect_log "Compiling $pkg/a.cc"
  try_with_timeout bazel shutdown || fail "shutdown failed"
  bazel build -s //$pkg:a >& $TEST_log || fail "build failed"
  expect_not_log "Compiling $pkg/a.cc"
}

run_suite "Tests for Bazel's C++ rules"
