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
# Test related to @platforms embedded repository
#

# --- begin runfiles.bash initialization ---
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

function test_platforms_repository_builds_itself() {
  # We test that a built-in @platforms repository is buildable.
  bazel build @platforms//:all &> $TEST_log \
      || fail "Build failed unexpectedly"
}

function test_platforms_can_be_overridden() {
  # We test that a custom repository can override @platforms in their
  # WORKSPACE file.
  mkdir -p platforms_can_be_overridden || fail "couldn't create directory"
  touch platforms_can_be_overridden/BUILD || \ fail "couldn't touch BUILD file"
  cat > platforms_can_be_overridden/WORKSPACE <<EOF
local_repository(
  name = 'platforms',
  path = '../override',
)
EOF

  mkdir -p override || fail "couldn't create override directory"
  touch override/WORKSPACE || fail "couldn't touch override/WORKSPACE"
  cat > override/BUILD <<EOF
# Have to use a rule that doesn't require a target platform, or else there will
# be a cycle.
toolchain_type(name = 'yolo')
EOF

  cd platforms_can_be_overridden || fail "couldn't cd into workspace"
  bazel build @platforms//:yolo &> $TEST_log || \
    fail "Bazel failed to build @platforms"
}

function test_incompatible_use_platforms_repo_for_constraints() {
  # We test that a built-in @platforms repository is buildable.
  bazel build --incompatible_use_platforms_repo_for_constraints @bazel_tools//platforms:all &> \
    $TEST_log && fail "Build passed when we expected an error."
  expect_log "Constraints from @bazel_tools//platforms have been removed."
}

function test_incompatible_use_platforms_repo_for_constraints_exec() {
  # We test that a built-in @platforms repository is buildable.
  cat > BUILD << 'EOF'
genrule(
    name = "hello",
    outs = ["hello.txt"],
    cmd = "echo 'Hello' > $@",
    tools = ["tool"],
)

genrule(
    name = "tool",
    outs = ["tool.sh"],
    cmd = "touch $@",
    target_compatible_with = select({
        "@bazel_tools//platforms:linux": [],
        "//conditions:default": [],
    })
)
EOF

  bazel build //:all &> \
    $TEST_log && fail "Build passed when we expected an error."
  expect_log "Constraints from @bazel_tools//platforms have been removed."
}

function test_platform_accessor() {
  cat > rules.bzl <<'EOF'
def _impl(ctx):
  platform = ctx.attr.platform[platform_common.PlatformInfo]
  properties = platform.exec_properties
  print("The properties are:", properties)
  return []

print_props = rule(
  implementation = _impl,
  attrs = {
      'platform': attr.label(providers = [platform_common.PlatformInfo]),
  }
)
EOF
  cat > BUILD << 'EOF'
load("//:rules.bzl", "print_props")

print_props(
    name = "a",
    platform = ":my_platform",
)

platform(
    name = "my_platform",
    exec_properties = {
        "key": "value",
        "key2": "value2",
        }
)
EOF

  bazel build --experimental_platforms_api=true :a &> $TEST_log || fail "Build failed"
  grep 'The properties are: {"key2": "value2", "key": "value"}' $TEST_log || fail "Did not find expected properties"
}

run_suite "platform repo test"

