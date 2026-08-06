#!/bin/bash
#
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

function set_up() {
  start_worker
}

function tear_down() {
  bazel clean >& "$TEST_log"
  stop_worker
}

function write_go_build() {
  local -r linkopts="$1"
  cat > pkg/BUILD.bazel <<EOF
load("@rules_go//go:def.bzl", "go_test")

go_test(
    name = "pkg_test",
    srcs = ["pkg_test.go"],
    data = ["data.txt"],
    gc_linkopts = ${linkopts},
)

go_test(
    name = "cgo_test",
    srcs = ["cgo_test.go"],
    cgo = True,
)

go_test(
    name = "sharded_test",
    srcs = ["pkg_test.go"],
    data = ["data.txt"],
    shard_count = 2,
)
EOF
}

function extract_early_keys() {
  sed -n 's/.* early_key=\([0-9a-f]*\).*/\1/p' "$TEST_log" | sort -u
}

function extract_producer_digest() {
  sed -n 's/.* producer_digest=\([0-9a-f]*\) early_key=.*/\1/p' "$TEST_log" | tail -1
}

function extract_metric() {
  local -r name="$1"
  sed -n "s/.* ${name}=\([0-9]*\).*/\1/p" "$TEST_log" | tail -1
}

function run_clean_test() {
  local -r target="$1"
  shift
  bazel clean >& /dev/null
  bazel test \
    --experimental_producer_keyed_test_cache \
    --experimental_producer_keyed_test_cache_debug \
    --remote_cache="grpc://localhost:${worker_port}" \
    --spawn_strategy=local \
    --test_strategy=standalone \
    "$@" \
    "$target" >& "$TEST_log" \
    || fail "Producer-keyed rules_go invocation failed for ${target}"
}

function run_clean_key() {
  local -r target="$1"
  shift
  run_clean_test "$target" "$@"
  local key
  key="$(extract_early_keys)"
  [[ -n "$key" && "$(echo "$key" | wc -l | tr -d ' ')" == 1 ]] \
    || fail "Expected one producer-keyed identity for ${target}, got: ${key}"
  echo "$key"
}

function assert_key_changed() {
  local -r before="$1"
  local -r after="$2"
  local -r mutation="$3"
  [[ "$before" != "$after" ]] \
    || fail "Producer-keyed identity did not change after ${mutation}"
}

function test_rules_go_mutation_and_performance_matrix() {
  cat > MODULE.bazel <<'EOF'
module(name = "producer_keyed_rules_go_test")

bazel_dep(name = "rules_go", version = "0.48.0")

go_sdk = use_extension("@rules_go//go:extensions.bzl", "go_sdk")
go_sdk.download(version = "1.22.5")
EOF

  cat > BUILD.bazel <<'EOF'
platform(
    name = "exec_a",
    exec_properties = {"producer_keyed_matrix": "a"},
    parents = ["@local_config_platform//:host"],
)

platform(
    name = "exec_b",
    exec_properties = {"producer_keyed_matrix": "b"},
    parents = ["@local_config_platform//:host"],
)
EOF

  mkdir -p pkg
  cat > pkg/pkg_test.go <<'EOF'
package pkg

import (
	"os"
	"testing"
)

func TestDataIsAvailable(t *testing.T) {
	if _, err := os.ReadFile("data.txt"); err != nil {
		t.Fatal(err)
	}
}
EOF
  cat > pkg/cgo_test.go <<'EOF'
package pkg

/*
static int answer(void) { return 42; }
*/
import "C"
import "testing"

func TestCgo(t *testing.T) {
	if got := int(C.answer()); got != 42 {
		t.Fatalf("answer = %d", got)
	}
}
EOF
  echo one > pkg/data.txt
  write_go_build '[]'

  # Establish the real rules_go baseline through normal remote execution. The execution log's
  # GoLink digest must exactly match the pre-execution producer digest.
  bazel clean >& /dev/null
  bazel test \
    --experimental_producer_keyed_test_cache \
    --experimental_producer_keyed_test_cache_debug \
    --execution_log_json_file=baseline.json \
    --remote_cache="grpc://localhost:${worker_port}" \
    --remote_executor="grpc://localhost:${worker_port}" \
    --spawn_strategy=remote,local \
    --test_strategy=standalone \
    //pkg:pkg_test >& "$TEST_log" \
    || fail "Failed to establish the real rules_go baseline"

  local baseline_key reported_digest execution_data executed_digest link_seconds
  local producer_compute_ms runfiles_compute_ms synthetic_compute_ms
  baseline_key="$(extract_early_keys)"
  reported_digest="$(extract_producer_digest)"
  execution_data="$(python3 - baseline.json <<'PY'
import json
import sys

data = open(sys.argv[1], encoding="utf-8").read()
decoder = json.JSONDecoder()
offset = 0
while offset < len(data):
    while offset < len(data) and data[offset].isspace():
        offset += 1
    if offset == len(data):
        break
    entry, offset = decoder.raw_decode(data, offset)
    if entry.get("mnemonic") == "GoLink":
        digest = entry.get("digest", {}).get("hash", "")
        duration = entry.get("metrics", {}).get("totalTime", "0s")
        print(digest, duration.removesuffix("s"))
        break
PY
)"
  executed_digest="${execution_data%% *}"
  link_seconds="${execution_data#* }"
  assert_equals "$executed_digest" "$reported_digest"

  producer_compute_ms="$(extract_metric producer_digest_compute_ms)"
  runfiles_compute_ms="$(extract_metric runfiles_fingerprint_ms)"
  synthetic_compute_ms="$(extract_metric synthetic_key_compute_ms)"
  [[ -n "$producer_compute_ms" && -n "$runfiles_compute_ms" && -n "$synthetic_compute_ms" ]] \
    || fail "Missing producer-keyed timing diagnostics"

  [[ ! -f "$cas_path/ac/${baseline_key:0:2}/$baseline_key" ]] \
    || fail "Compute-only rules_go baseline unexpectedly wrote an alias"
  bazel clean >& /dev/null
  bazel test \
    --experimental_producer_keyed_test_cache \
    --experimental_producer_keyed_test_cache_write_aliases \
    --experimental_producer_keyed_test_cache_debug \
    --execution_log_json_file=write_alias.json \
    --remote_cache="grpc://localhost:${worker_port}" \
    --remote_executor="grpc://localhost:${worker_port}" \
    --spawn_strategy=remote,local \
    --test_strategy=standalone \
    //pkg:pkg_test >& "$TEST_log" \
    || fail "Failed to write the real rules_go producer-keyed alias"
  assert_equals "$baseline_key" "$(extract_early_keys)"
  expect_log "producer-keyed test cache: alias written early_key=$baseline_key"
  [[ -f "$cas_path/ac/${baseline_key:0:2}/$baseline_key" ]] \
    || fail "Real rules_go alias was not present in the remote action cache"
  grep -q '"mnemonic": "GoLink"' write_alias.json \
    || fail "Write-only rules_go mode unexpectedly skipped GoLink"

  bazel clean >& /dev/null
  bazel test \
    --experimental_producer_keyed_test_cache \
    --experimental_producer_keyed_test_cache_shadow \
    --experimental_producer_keyed_test_cache_debug \
    --execution_log_json_file=shadow_hit.json \
    --remote_cache="grpc://localhost:${worker_port}" \
    --remote_executor="grpc://localhost:${worker_port}" \
    --spawn_strategy=remote,local \
    --test_strategy=standalone \
    //pkg:pkg_test >& "$TEST_log" \
    || fail "Failed real rules_go shadow-hit invocation"
  assert_equals "$baseline_key" "$(extract_early_keys)"
  expect_log "producer-keyed test cache: shadow_lookup=hit early_key=$baseline_key"
  expect_log "producer-keyed test cache: shadow_compare=match early_key=$baseline_key"
  grep -q '"mnemonic": "GoLink"' shadow_hit.json \
    || fail "Shadow rules_go mode unexpectedly skipped GoLink"

  bazel clean >& /dev/null
  bazel test \
    --experimental_producer_keyed_test_cache \
    --experimental_producer_keyed_test_cache_enabled \
    --experimental_producer_keyed_test_cache_debug \
    --execution_log_json_file=short_circuit.json \
    --build_event_json_file=short_circuit.bep.json \
    --remote_cache="grpc://localhost:${worker_port}" \
    --remote_executor="grpc://localhost:${worker_port}" \
    --spawn_strategy=remote,local \
    --test_strategy=standalone \
    //pkg:pkg_test >& "$TEST_log" \
    || fail "Failed real rules_go early short-circuit invocation"
  expect_log "producer-keyed test cache: early_short_circuit=hit early_key=$baseline_key"
  expect_log "//pkg:pkg_test.*\(cached\).*PASSED"
  if grep -q '"mnemonic": "GoLink"' short_circuit.json; then
    fail "Early rules_go hit unexpectedly requested GoLink"
  fi
  if grep -q '"mnemonic": "TestRunner"' short_circuit.json; then
    fail "Early rules_go hit unexpectedly executed TestRunner"
  fi
  [[ -s bazel-testlogs/pkg/pkg_test/test.log ]] \
    || fail "Early rules_go hit did not restore test.log"
  grep -q '"testResult"' short_circuit.bep.json \
    || fail "Early rules_go hit did not publish a BEP test result"
  grep -q '"id":{"targetCompleted".*"completed"' short_circuit.bep.json \
    || fail "Early rules_go hit did not publish target completion"

  local stable_key noop_source_key source_key data_key linkopts_key args_key env_key run_under_key
  stable_key="$(run_clean_key //pkg:pkg_test)"
  assert_equals "$baseline_key" "$stable_key"

  echo '// non-semantic source mutation' >> pkg/pkg_test.go
  noop_source_key="$(run_clean_key //pkg:pkg_test)"
  assert_equals "$stable_key" "$noop_source_key"

  sed -i.bak 's/t.Fatal(err)/t.Fatalf("read data: %v", err)/' pkg/pkg_test.go
  source_key="$(run_clean_key //pkg:pkg_test)"
  assert_key_changed "$noop_source_key" "$source_key" "compiled Go source mutation"

  echo two > pkg/data.txt
  data_key="$(run_clean_key //pkg:pkg_test)"
  assert_key_changed "$source_key" "$data_key" "test data mutation"

  write_go_build '["-s"]'
  linkopts_key="$(run_clean_key //pkg:pkg_test)"
  assert_key_changed "$data_key" "$linkopts_key" "gc_linkopts mutation"

  args_key="$(run_clean_key //pkg:pkg_test --test_arg=-test.v)"
  assert_key_changed "$linkopts_key" "$args_key" "test argument mutation"

  env_key="$(run_clean_key //pkg:pkg_test --test_env=PRODUCER_KEYED_MATRIX=one)"
  assert_key_changed "$linkopts_key" "$env_key" "test environment mutation"

  run_under_key="$(run_clean_key //pkg:pkg_test --run_under=/usr/bin/env)"
  assert_key_changed "$linkopts_key" "$run_under_key" "run_under mutation"

  local platform_a_key platform_b_key
  platform_a_key="$(run_clean_key //pkg:pkg_test --extra_execution_platforms=//:exec_a)"
  platform_b_key="$(run_clean_key //pkg:pkg_test --extra_execution_platforms=//:exec_b)"
  assert_key_changed "$platform_a_key" "$platform_b_key" "execution platform mutation"

  run_clean_test //pkg:cgo_test
  expect_log "producer-keyed test cache: //pkg:cgo_test: eligible producer=GoLink"

  run_clean_test //pkg:pkg_test \
    --@rules_go//go/config:pure=false \
    --@rules_go//go/config:race=true
  expect_log "producer-keyed test cache: //pkg:pkg_test: eligible producer=GoLink"

  run_clean_test //pkg:sharded_test
  assert_equals 2 "$(extract_early_keys | wc -l | tr -d ' ')"

  run_clean_test //pkg:pkg_test --cache_test_results=yes --runs_per_test=2
  assert_equals 2 "$(extract_early_keys | wc -l | tr -d ' ')"

  bazel clean >& /dev/null
  bazel test \
    --experimental_producer_keyed_test_cache \
    --experimental_producer_keyed_test_cache_debug \
    --nocache_test_results \
    --remote_cache="grpc://localhost:${worker_port}" \
    --spawn_strategy=local \
    //pkg:pkg_test >& "$TEST_log" \
    || fail "Cache-policy test failed"
  expect_log "CACHE_POLICY_DISALLOWS"
  expect_not_log "producer_digest="

  bazel clean >& /dev/null
  bazel test \
    --experimental_producer_keyed_test_cache \
    --experimental_producer_keyed_test_cache_debug \
    --spawn_strategy=local \
    //pkg:pkg_test >& "$TEST_log" \
    || fail "Cacheless test failed"
  expect_log "CACHE_UNCONFIGURED"
  expect_not_log "producer_digest="

  local -r report="${TEST_UNDECLARED_OUTPUTS_DIR:-$TEST_TMPDIR}/producer_keyed_rules_go_metrics.txt"
  cat > "$report" <<EOF
rules_go_version=0.48.0
go_sdk_version=1.22.5
producer_digest_compute_ms=${producer_compute_ms}
runfiles_fingerprint_ms=${runfiles_compute_ms}
synthetic_key_compute_ms=${synthetic_compute_ms}
golink_total_seconds=${link_seconds}
EOF
}

run_suite "Producer-keyed test cache rules_go compatibility and mutation tests"
