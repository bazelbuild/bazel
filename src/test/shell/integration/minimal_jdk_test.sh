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
# Some basic tests to check if a bazel with minimal embedded JDK is functional.

# --- begin runfiles.bash initialization ---
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$TEST_SRCDIR/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
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

export BAZEL_SUFFIX="_jdk_minimal"
source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Bazel's install base is < 460MB with minimal JDK and > 460MB with an all
# modules JDK.
function test_size_less_than_460MB() {
  bazel info
  ib=$(bazel info install_base)
  size=$(du -s "$ib" | cut -d\	 -f1)
  maxsize=$((1024*460))
  if [ $size -gt $maxsize ]; then
    echo "$ib was too big:" 1>&2
    du -a "$ib" 1>&2
    fail "Size of install_base is $size kB, expected it to be less than $maxsize kB."
  fi
}

function test_cc() {
  local -r pkg=$FUNCNAME
  mkdir -p "$pkg" || fail "Couldn't create $pkg."
  cat > "$pkg/BUILD" <<EOF
cc_binary(
    name = "foo",
    srcs = ["foo.cc"],
)
EOF
  cat > "$pkg/foo.cc" <<EOF
int main() {
  return 42;
}
EOF

  bazel build "//$pkg:foo" || fail "Expected success."
}

function test_java() {
  local -r pkg=$FUNCNAME
  mkdir -p "$pkg" || fail "Couldn't create $pkg."
  cat > "$pkg/BUILD" <<EOF
java_binary(
    name = "foo",
    srcs = ["foo.java"],
    main_class = "foo",
)
EOF
  cat > "$pkg/foo.java" <<EOF
public class foo {
  public static void main(String[] args) {
    System.out.println("Hello Bazel!");
  }
}
EOF

  bazel build "//$pkg:foo" || fail "Expected success."
}

run_suite "Test bazel with minimal embedded JDK."
