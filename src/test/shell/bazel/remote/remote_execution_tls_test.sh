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
# Tests remote caching with TLS.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

cert_path="${BAZEL_RUNFILES}/src/test/testdata/test_tls_certificate"

function set_up() {
  work_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  cas_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  pid_file=$(mktemp -u "${TEST_TMPDIR}/remote.XXXXXXXX")
  attempts=1
  while [ $attempts -le 3 ]; do
    (( attempts++ ))
    worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
    "${BAZEL_RUNFILES}/src/tools/remote/worker" \
        --work_path="${work_path}" \
        --listen_port=${worker_port} \
        --cas_path=${cas_path} \
        --tls_certificate="${cert_path}/server.crt" \
        --tls_private_key="${cert_path}/server.pem" \
        --pid_file="${pid_file}" >& $TEST_log &
    local wait_seconds=0
    until [ -s "${pid_file}" ] || [ "$wait_seconds" -eq 15 ]; do
      sleep 1
      ((wait_seconds++)) || true
    done
    if [ -s "${pid_file}" ]; then
      break
    fi
  done
  if [ ! -s "${pid_file}" ]; then
    fail "Timed out waiting for remote worker to start."
  fi
}

function _prepareBasicRule(){
  mkdir -p a
  cat > a/BUILD <<EOF
genrule(
  name = 'foo',
  outs = ["foo.txt"],
  cmd = "echo \"foo bar\" > \$@",
)
EOF

}
function tear_down() {
  bazel clean >& $TEST_log
  if [ -s "${pid_file}" ]; then
    local pid=$(cat "${pid_file}")
    kill "${pid}" || true
  fi
  rm -rf "${pid_file}"
  rm -rf "${work_path}"
  rm -rf "${cas_path}"
}

function test_remote_grpcs_cache() {
  # Test that if 'grpcs' is provided as a scheme for --remote_cache flag, remote cache works.
  _prepareBasicRule

  bazel build \
      --remote_cache=grpcs://localhost:${worker_port} \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo \
      || fail "Failed to build //a:foo with grpcs remote cache"
}

function test_remote_grpc_cache_with_legacy_tls_enabled() {
  # Test that if default scheme for --remote_cache flag with --tls_enabled, remote cache works.
  _prepareBasicRule

  bazel build \
      --remote_cache=localhost:${worker_port} \
      --tls_enabled=true \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo \
      || fail "Failed to build //a:foo with grpc --tls_enabled remote cache"
}

function test_remote_https_cache() {
  # Test that if 'https' is provided as a scheme for --remote_cache flag, remote cache works.
  _prepareBasicRule

  bazel build \
      --remote_cache=https://localhost:${worker_port} \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo \
      || fail "Failed to build //a:foo with https remote cache"
}

function test_remote_cache_with_incompatible_tls_enabled_removed_default_scheme() {
  # Test that if default scheme for --remote_cache flag with --incompatible_tls_enabled_removed, remote cache works.
  _prepareBasicRule

  bazel build \
      --remote_cache=localhost:${worker_port} \
      --incompatible_tls_enabled_removed=true \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo \
      || fail "Failed to build //a:foo with default(grpcs) remote cache"
}

function test_remote_cache_with_incompatible_tls_enabled_removed_grpcs_scheme() {
  # Test that if 'grpcs' scheme for --remote_cache flag with --incompatible_tls_enabled_removed, remote cache works.
  _prepareBasicRule

  bazel build \
      --remote_cache=grpcs://localhost:${worker_port} \
      --incompatible_tls_enabled_removed=true \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo \
      || fail "Failed to build //a:foo with grpcs remote cache"
}

function test_remote_cache_with_incompatible_tls_enabled_removed_grpc_scheme() {
  # Test that if 'grpc' scheme for --remote_cache flag with --incompatible_tls_enabled_removed, remote cache fails.
  _prepareBasicRule

  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --incompatible_tls_enabled_removed=true \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo \
      && fail "Expected test failure" || true
}

function test_remote_cache_with_incompatible_tls_enabled_removed() {
  # Test that if --incompatible_tls_enabled_removed=true and --tls_enabled=true an error is thrown
  _prepareBasicRule

  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --tls_enabled=true \
      --incompatible_tls_enabled_removed=true \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo \
      && fail "Expected test failure" || true
}

run_suite "Remote cache TLS tests"
