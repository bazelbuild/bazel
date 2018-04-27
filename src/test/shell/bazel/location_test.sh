#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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

set -euo pipefail
# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/io_bazel/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  elif [[ -f "$TEST_SRCDIR/io_bazel/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$TEST_SRCDIR"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/io_bazel/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/io_bazel/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^io_bazel/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find //tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_external_location() {
  local -r pkg=${FUNCNAME}
  cat > WORKSPACE <<EOF
bind(
   name = "foo",
   actual = "//$pkg:baz"
)
EOF
  mkdir $pkg
  cat > $pkg/BUILD <<EOF
genrule(
    name = "baz-rule",
    outs = ["baz"],
    cmd = "echo 'hello' > \"\$@\"",
    visibility = ["//visibility:public"],
)

genrule(
    name = "use-loc",
    srcs = ["//external:foo"],
    outs = ["loc"],
    cmd = "cat \$(location //external:foo) > \"\$@\"",
)
EOF

  bazel build //$pkg:loc &> $TEST_log || fail "Referencing external genrule didn't build"
  assert_contains "hello" bazel-genfiles/$pkg/loc
}

function test_external_location_tool() {
  local -r pkg=${FUNCNAME}
  cat > WORKSPACE <<EOF
bind(
   name = "foo",
   actual = "//$pkg:baz"
)
EOF
  mkdir $pkg
  cat > $pkg/BUILD <<EOF
genrule(
    name = "baz-rule",
    outs = ["baz"],
    cmd = "echo '#!/bin/echo hello' > \"\$@\"",
    visibility = ["//visibility:public"],
)

genrule(
    name = "use-loc",
    tools = ["//external:foo"],
    outs = ["loc"],
    cmd = "\$(location //external:foo) > \"\$@\"",
)
EOF

  bazel build //$pkg:loc &> $TEST_log || fail "Referencing external genrule in tools didn't build"
  assert_contains "hello" bazel-genfiles/$pkg/loc
}

function test_location_trim() {
  local -r pkg=${FUNCNAME}
  mkdir $pkg
  cat > $pkg/BUILD <<EOF
genrule(
    name = "baz-rule",
    outs = ["baz"],
    cmd = "echo helloworld > \"\$@\"",
)

genrule(
    name = "loc-rule",
    srcs = [":baz-rule"],
    outs = ["loc"],
    cmd = "echo \$(location  :baz-rule ) > \"\$@\"",
)
EOF

  bazel build //$pkg:loc || fail "Label was not trimmed before lookup"
}

run_suite "location tests"
