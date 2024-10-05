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
#
# Tests the examples provided in Bazel
#

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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

function test_rules_cc_can_be_overridden() {
  # rules_java relies on rules_cc contents
  mock_rules_java_to_avoid_downloading

  # The bazelrc file might contain an --override_repository flag for rules_cc,
  # which would cause this test to fail to override the repo via a WORKSPACE file.
  sed -i.bak '/override_repository=rules_cc/d' $TEST_TMPDIR/bazelrc

  # We test that a custom repository can override @rules_cc in their
  # WORKSPACE file.
  mkdir -p rules_cc_can_be_overridden || fail "couldn't create directory"
  touch rules_cc_can_be_overridden/BUILD || \ fail "couldn't touch BUILD file"
  cat > rules_cc_can_be_overridden/WORKSPACE <<EOF
local_repository(
  name = 'rules_cc',
  path = '../override',
)
EOF

  mkdir -p override || fail "couldn't create override directory"
  touch override/WORKSPACE || fail "couldn't touch override/WORKSPACE"
  cat > override/BUILD <<EOF
filegroup(name = 'yolo')
EOF

  cd rules_cc_can_be_overridden || fail "couldn't cd into workspace"
  bazel build --noenable_bzlmod --enable_workspace --incompatible_autoload_externally= @rules_cc//:yolo &> $TEST_log || \
    fail "Bazel failed to build @rules_cc"
}

function test_rules_cc_repository_builds_itself() {
  add_rules_cc "MODULE.bazel"
  write_default_bazelrc
  # can be removed with protobuf v28.x onwards
  if $is_windows; then
    CXXOPTS=""
  else
    CXXOPTS="--cxxopt=-Wno-deprecated-declarations --host_cxxopt=-Wno-deprecated-declarations"
  fi
  # We test that a built-in @rules_cc repository is buildable.
  bazel build $CXXOPTS @rules_cc//cc/... &> $TEST_log \
      || fail "Build failed unexpectedly"
}

run_suite "rules_cc tests"
