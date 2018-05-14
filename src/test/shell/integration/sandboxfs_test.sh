#!/bin/bash
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

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Bazel build arguments to disable the use of the sandbox.  We have tests below
# that configure a fake sandboxfs and they would fail if they were to use it.
DISABLE_SANDBOX_ARGS=(
  --genrule_strategy=local
  --spawn_strategy=local
)

function create_fake_sandboxfs() {
  local path="${1}"; shift

  cat >"${path}" <<EOF
#! /bin/sh
echo "ARGS: \${*}" 1>&2

while read line; do
  echo "Received: \${line}" 1>&2
  if [ -z "\${line}" ]; then
    echo "Done"
  fi
done
EOF
  chmod +x "${path}"
}

function create_hello_package() {
  mkdir -p hello

  cat >hello/BUILD <<EOF
cc_binary(name = "hello", srcs = ["hello.cc"])
EOF

  cat >hello/hello.cc <<EOF
#include <stdio.h>
int main(void) { printf("Hello, world!\n"); return 0; }
EOF
}

function test_default_sandboxfs_from_path() {
  mkdir -p fake-tools
  create_fake_sandboxfs fake-tools/sandboxfs
  PATH="$(pwd)/fake-tools:${PATH}"; export PATH

  create_hello_package

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"

  # This test relies on a PATH change that is only recognized when the server
  # first starts up, so ensure there are no Bazel servers left behind.
  #
  # TODO(philwo): This is awful.  The testing infrastructure should ensure
  # that tests cannot pollute each other's state, but at the moment that's not
  # the case.
  bazel shutdown

  bazel build \
    "${DISABLE_SANDBOX_ARGS[@]}" \
    --experimental_use_sandboxfs \
    //hello >"${TEST_log}" 2>&1 || fail "Build should have succeeded"

  expect_log "Mounting sandboxfs instance"
}

function test_explicit_sandboxfs_not_found() {
  create_hello_package

  bazel build \
    --experimental_use_sandboxfs \
    --experimental_sandboxfs_path="/non-existent/sandboxfs" \
    //hello >"${TEST_log}" 2>&1 && fail "Build succeeded but should have failed"

  expect_log "Failed to initialize sandbox: .*Cannot run .*/non-existent/"
}

function test_mount_unmount() {
  create_fake_sandboxfs fake-sandboxfs.sh
  create_hello_package

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"

  bazel build \
    "${DISABLE_SANDBOX_ARGS[@]}" \
    --experimental_use_sandboxfs \
    --experimental_sandboxfs_path="$(pwd)/fake-sandboxfs.sh" \
    --sandbox_debug \
    //hello >"${TEST_log}" 2>&1 || fail "Build should have succeeded"

  expect_log "Mounting sandboxfs instance"
  expect_log "unmounting sandboxfs"

  # Dump fake sandboxfs' log for debugging.
  sed -e 's,^,SANDBOXFS: ,' "${sandbox_base}/sandboxfs.log" >>"${TEST_log}"

  grep -q "ARGS: .*${sandbox_base}/sandboxfs" "${sandbox_base}/sandboxfs.log" \
    || fail "Cannot find expected mount point in sandboxfs mount call"
}

run_suite "sandboxfs-based sandboxing tests"
