#!/usr/bin/env bash
# Copyright 2026 The Bazel Authors. All rights reserved.
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
# Tests for the mobile-install command's handling of the deployer process.
# Uses a fake mobile-install aspect so that no Android SDK or device is
# required: mobile-install just builds the requested output groups and then
# executes <bindir>/<target>_mi/launcher as a child process.

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Exit code that MobileInstall.Code.NON_ZERO_EXIT maps to in
# failure_details.proto.
readonly MOBILE_INSTALL_NON_ZERO_EXIT=6

function setup_fake_mi_workspace() {
  mkdir -p mi
  cat > mi/mi.bzl <<'EOF'
# Minimal stand-in for @rules_android//mobile_install:mi.bzl. The launcher
# emitted for targets whose name starts with "fail" exits with code 1,
# simulating a deployer failure (e.g. an adb install timeout).
def _mi_aspect_impl(target, ctx):
    launcher = ctx.actions.declare_file(target.label.name + "_mi/launcher")
    if target.label.name.startswith("fail"):
        script = "#!/bin/bash\necho 'fake deployer: install failed' >&2\nexit 1\n"
    else:
        script = "#!/bin/bash\necho 'fake deployer: install finished'\n"
    ctx.actions.write(output = launcher, content = script, is_executable = True)
    return [OutputGroupInfo(
        mobile_install_INTERNAL_ = depset([launcher]),
        mobile_install_launcher_INTERNAL_ = depset([launcher]),
    )]

MIASPECT = aspect(implementation = _mi_aspect_impl)

def _fake_binary_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(out, "fake app")
    return [DefaultInfo(files = depset([out]))]

fake_binary = rule(implementation = _fake_binary_impl)
EOF
  cat > mi/BUILD <<'EOF'
load(":mi.bzl", "fake_binary")

fake_binary(name = "fail_app")

fake_binary(name = "ok_app")
EOF
}

function mobile_install() {
  bazel mobile-install \
      --mobile_install_aspect=//mi:mi.bzl \
      --mobile_install_supported_rules=fake_binary \
      "$@"
}

function test_deployer_success() {
  setup_fake_mi_workspace
  mobile_install //mi:ok_app &> "$TEST_log" \
      || fail "mobile-install should succeed when the deployer exits 0"
  expect_log "fake deployer: install finished"
  expect_log "Build completed successfully"
}

function test_deployer_failure_sets_exit_code() {
  setup_fake_mi_workspace
  local exit_code=0
  mobile_install //mi:fail_app &> "$TEST_log" || exit_code=$?
  assert_equals "$MOBILE_INSTALL_NON_ZERO_EXIT" "$exit_code"
  expect_log "fake deployer: install failed"
  expect_log "Non-zero return code '1' from command"
}

function test_deployer_failure_reported_after_build_summary() {
  # Regression test: the deployer runs after the build has completed, so its
  # failure must be reported after the "Build completed successfully" summary.
  # It used to run as a post-build callback inside the build request, which
  # made a successful build summary the last thing printed after a deploy
  # failure.
  setup_fake_mi_workspace
  mobile_install //mi:fail_app &> "$TEST_log" \
      && fail "mobile-install should fail when the deployer exits 1"
  expect_log "Build completed successfully"
  local summary_line error_line
  summary_line="$(grep -n "Build completed successfully" "$TEST_log" | head -1 | cut -d: -f1)"
  error_line="$(grep -n "Non-zero return code '1' from command" "$TEST_log" | head -1 | cut -d: -f1)"
  if [[ -z "$error_line" ]]; then
    fail "deployer failure was not reported"
  fi
  if (( error_line < summary_line )); then
    fail "deployer failure (line $error_line) was reported before the build \
summary (line $summary_line), so the log ends with a success message"
  fi
}

run_suite "mobile-install integration tests"
