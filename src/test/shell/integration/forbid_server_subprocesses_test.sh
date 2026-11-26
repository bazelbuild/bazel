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
# End-to-end tests for --noallow_server_subprocesses.

set -euo pipefail

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function is_supported_platform() {
  if [[ "$(uname -s)" != "Linux" ]]; then
    return 1
  fi
  case "$(uname -m)" in
    x86_64|aarch64) return 0 ;;
    *) return 1 ;;
  esac
}

function workspace_status_flag() {
  if [[ "$PRODUCT_NAME" == "blaze" ]]; then
    echo "--workspace_status_command=/bin/true"
  else
    echo "--workspace_status_command="
  fi
}

function test_forbid_server_subprocesses_blocks_local_action() {
  if ! is_supported_platform; then
    echo "Skipping on unsupported platform $(uname -s)/$(uname -m)"
    return 0
  fi

  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"
  cat > "$pkg/BUILD" <<'EOF'
genrule(
    name = "local_action",
    outs = ["out.txt"],
    cmd = "echo hello > $@",
    local = True,
)
EOF

  bazel --noallow_server_subprocesses build \
      "$(workspace_status_flag)" \
      "//${pkg}:local_action" >&"$TEST_log" && fail "Expected local action to fail" || true
  expect_log "Operation not permitted"
}

function test_forbid_server_subprocesses_allows_write_action() {
  if ! is_supported_platform; then
    echo "Skipping on unsupported platform $(uname -s)/$(uname -m)"
    return 0
  fi

  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"
  cat > "$pkg/write_file.bzl" <<'EOF'
def _write_file_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(output = out, content = "hello\n")
    return [DefaultInfo(files = depset([out]))]

write_file = rule(implementation = _write_file_impl)
EOF
  cat > "$pkg/BUILD" <<'EOF'
load(":write_file.bzl", "write_file")

write_file(name = "write_only")
EOF

  bazel --noallow_server_subprocesses build \
      "$(workspace_status_flag)" \
      "//${pkg}:write_only" >&"$TEST_log" || fail "Expected write-only action to succeed"
}

function test_forbid_server_subprocesses_batch_mode() {
  if ! is_supported_platform; then
    echo "Skipping on unsupported platform $(uname -s)/$(uname -m)"
    return 0
  fi

  local -r pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"
  cat > "$pkg/write_file.bzl" <<'EOF'
def _write_file_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(output = out, content = "hello\n")
    return [DefaultInfo(files = depset([out]))]

write_file = rule(implementation = _write_file_impl)
EOF
  cat > "$pkg/BUILD" <<'EOF'
load(":write_file.bzl", "write_file")

write_file(name = "write_only")

genrule(
    name = "local_action",
    outs = ["out.txt"],
    cmd = "echo hello > $@",
    local = True,
)
EOF

  bazel --batch --noallow_server_subprocesses build \
      "$(workspace_status_flag)" \
      "//${pkg}:write_only" >&"$TEST_log" || fail "Expected batch write-only action to succeed"

  bazel --batch --noallow_server_subprocesses build \
      "$(workspace_status_flag)" \
      "//${pkg}:local_action" >&"$TEST_log" && fail "Expected batch local action to fail" || true
  expect_log "Operation not permitted"
}

function test_forbid_server_subprocesses_clean_async_falls_back() {
  if ! is_supported_platform; then
    echo "Skipping on unsupported platform $(uname -s)/$(uname -m)"
    return 0
  fi

  bazel --noallow_server_subprocesses clean --async >&"$TEST_log" \
      || fail "Expected clean --async to fall back to synchronous clean"
  expect_log "--async cannot be used when --noallow_server_subprocesses is set, falling back to synchronous clean"
}

run_suite "Tests for --noallow_server_subprocesses"
