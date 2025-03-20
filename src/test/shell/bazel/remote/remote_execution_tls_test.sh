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

set -euo pipefail

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
source "$(rlocation "io_bazel/src/test/shell/bazel/remote/remote_utils.sh")" \
  || { echo "remote_utils.sh not found!" >&2; exit 1; }

cert_path="${BAZEL_RUNFILES}/src/test/testdata/test_tls_certificate"
client_mtls_flags=
enable_mtls=0
if [[ $1 == "--mtls" ]]; then
  enable_mtls=1
  client_mtls_flags="--tls_client_certificate=${cert_path}/client.crt --tls_client_key=${cert_path}/client.pem"
fi

function set_up() {
  mtls_flag=
  if [[ $enable_mtls == 1 ]]; then
    mtls_flag=--tls_ca_certificate="${cert_path}/ca.crt"
  fi
  start_worker \
        --tls_certificate="${cert_path}/server.crt" \
        --tls_private_key="${cert_path}/server.pem" \
        $mtls_flag
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
  stop_worker
}

function test_remote_grpcs_cache() {
  # Test that if 'grpcs' is provided as a scheme for --remote_cache flag, remote cache works.
  _prepareBasicRule

  bazel build \
      --remote_cache=grpcs://localhost:${worker_port} \
      --tls_certificate="${cert_path}/ca.crt" \
      ${client_mtls_flags} \
      //a:foo \
      || fail "Failed to build //a:foo with grpcs remote cache"
}

# Tests that bazel fails if no client cert is provided but the server requires one.
function test_mtls_fails_if_client_has_no_cert() {
  # This test only makes sense when we test mtls.
  [[ $enable_mtls == 1 ]] || return 0
  _prepareBasicRule

  bazel build \
      --remote_cache=grpcs://localhost:${worker_port} \
      --tls_certificate="${cert_path}/ca.crt" \
      //a:foo &> $TEST_log \
      && fail "Expected bazel to fail without a client cert" || true
  expect_log "Failed to query remote execution capabilities:"
}

function test_remote_grpc_cache() {
  # Test that if default scheme for --remote_cache flag, remote cache works.
  _prepareBasicRule

  bazel build \
      --remote_cache=localhost:${worker_port} \
      --tls_certificate="${cert_path}/ca.crt" \
      ${client_mtls_flags} \
      //a:foo \
      || fail "Failed to build //a:foo with grpc remote cache"
}

function test_remote_cache_with_incompatible_tls_enabled_removed_grpc_scheme() {
  # Test that if 'grpc' scheme for --remote_cache flag, remote cache fails.
  _prepareBasicRule

  bazel build \
      --remote_cache=grpc://localhost:${worker_port} \
      --tls_certificate="${cert_path}/ca.crt" \
      ${client_mtls_flags} \
      //a:foo &> $TEST_log \
      && fail "Expected test failure" || true
  expect_log "Failed to query remote execution capabilities:"
}

run_suite "Remote cache TLS tests"
