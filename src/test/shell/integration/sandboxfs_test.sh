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

# Creates a fake sandboxfs process in "path" that logs interactions with it in
# the given "log" file and reports the given "version".
function create_fake_sandboxfs() {
  local path="${1}"; shift
  local log="${1}"; shift
  local version="${1}"; shift

  cat >"${path}" <<EOF
#! /bin/sh

rm -f "${log}"
trap 'echo "Terminated" >>"${log}"' EXIT TERM

echo "PID: \${$}" >>"${log}"
echo "ARGS: \${*}" >>"${log}"

if [ "\${1}" = --version ]; then
  echo "${version}"
  exit 0
fi

while read line; do
  echo "Received: \${line}" >>"${log}"
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
  create_fake_sandboxfs fake-tools/sandboxfs "$(pwd)/log" "sandboxfs 0.0.0"
  PATH="$(pwd)/fake-tools:${PATH}"; export PATH

  create_hello_package

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

  # Dump fake sandboxfs' log for debugging.
  sed -e 's,^,SANDBOXFS: ,' log >>"${TEST_log}"

  grep -q "Terminated" log \
    || fail "sandboxfs process was not terminated (not executed?)"
}

function test_explicit_sandboxfs_not_found() {
  create_hello_package

  bazel build \
    --experimental_use_sandboxfs \
    --experimental_sandboxfs_path="/non-existent/sandboxfs" \
    //hello >"${TEST_log}" 2>&1 && fail "Build succeeded but should have failed"

  expect_log "/non-existent/sandboxfs.*not be found"
}

function test_explicit_sandboxfs_is_invalid() {
  mkdir -p fake-tools
  create_hello_package
  do_build() {
    bazel build \
      "${DISABLE_SANDBOX_ARGS[@]}" \
      --experimental_use_sandboxfs=yes \
      --experimental_sandboxfs_path="$(pwd)/fake-tools/sandboxfs" \
      //hello
  }

  # Try with a binary that prints an invalid version string.
  create_fake_sandboxfs fake-tools/sandboxfs "$(pwd)/log" "not-sandboxfs 0.0.0"
  do_build >"${TEST_log}" 2>&1 && fail "Build should have failed"

  # Now try with a valid binary just to ensure our test scenario works.
  create_fake_sandboxfs fake-tools/sandboxfs "$(pwd)/log" "sandboxfs 0.0.0"
  do_build >"${TEST_log}" 2>&1 || fail "Build should have succeeded"
  sed -e 's,^,SANDBOXFS: ,' log >>"${TEST_log}"

  grep -q "Terminated" log \
    || fail "sandboxfs process was not terminated (not executed?)"
}

function test_use_sandboxfs_if_present() {
  # This test relies on a PATH change that is only recognized when the server
  # first starts up, so ensure there are no Bazel servers left behind.
  #
  # TODO(philwo): This is awful.  The testing infrastructure should ensure
  # that tests cannot pollute each other's state, but at the moment that's not
  # the case.
  bazel shutdown

  mkdir -p fake-tools
  PATH="$(pwd)/fake-tools:${PATH}"; export PATH
  create_hello_package
  do_build() {
    bazel build \
      "${DISABLE_SANDBOX_ARGS[@]}" \
      --experimental_use_sandboxfs=auto \
      //hello
  }

  # Try with sandboxfs not in the PATH.
  do_build >"${TEST_log}" 2>&1 || fail "Build should have succeeded"
  [[ ! -f log ]] || echo "sandboxfs was used but should not have"

  # Now try with sandboxfs in the PATH to ensure our test scenario works.
  create_fake_sandboxfs fake-tools/sandboxfs "$(pwd)/log" "sandboxfs 0.0.0"
  do_build >"${TEST_log}" 2>&1 || fail "Build should have succeeded"
  sed -e 's,^,SANDBOXFS: ,' log >>"${TEST_log}"
  grep -q "Terminated" log \
    || fail "sandboxfs process was not terminated (not executed?)"
}

# Runs a build of the given target using a fake sandboxfs that captures its
# activity and dumps it to the given log file.
function build_with_fake_sandboxfs() {
  local log="${1}"; shift

  create_fake_sandboxfs fake-sandboxfs.sh "${log}" "sandboxfs 0.0.0"

  local ret=0
  bazel build \
    "${DISABLE_SANDBOX_ARGS[@]}" \
    --experimental_use_sandboxfs \
    --experimental_sandboxfs_path="$(pwd)/fake-sandboxfs.sh" \
    "${@}" >"${TEST_log}" 2>&1 || ret="${?}"

  # Dump fake sandboxfs' log for debugging.
  sed -e 's,^,SANDBOXFS: ,' log >>"${TEST_log}"

  return "${ret}"
}

function test_mount_unmount() {
  create_hello_package

  local output_base="$(bazel info output_base)"
  local sandbox_base="${output_base}/sandbox"

  build_with_fake_sandboxfs "$(pwd)/log" //hello \
    || fail "Build should have succeeded"

  grep -q "ARGS: .*${sandbox_base}/sandboxfs" log \
    || fail "Cannot find expected mount point in sandboxfs mount call"
  grep -q "Terminated" log \
    || fail "sandboxfs process was not terminated (not unmounted?)"
}

function test_debug_lifecycle() {
  create_hello_package

  function sandboxfs_pid() {
    case "$(uname)" in
      Darwin)
        # We cannot use ps to look for the sandbox process because this is
        # not allowed when running with macOS's App Sandboxing.
        grep -q "Terminated" log && return
        grep "^PID:" log | awk '{print $2}'
        ;;

      *)
        # We could use the same approach we follow on Darwin to look for the
        # PID of the subprocess, but it's better if we look at the real
        # process table if we are able to.
        ps ax | grep [f]ake-sandboxfs | awk '{print $1}'
        ;;
    esac
  }

  # Want sandboxfs to be left mounted after a build with debugging on.
  build_with_fake_sandboxfs "$(pwd)/log" --sandbox_debug //hello
  grep -q "ARGS:" log || fail "sandboxfs was not run"
  grep -q "Terminated" log \
    && fail "sandboxfs process was terminated but should not have been"
  local pid1="$(sandboxfs_pid)"
  [[ -n "${pid1}" ]] || fail "sandboxfs process not found in process table"

  # Want sandboxfs to be restarted if the previous build had debugging on.
  build_with_fake_sandboxfs "$(pwd)/log" --sandbox_debug //hello
  local pid2="$(sandboxfs_pid)"
  [[ -n "${pid2}" ]] || fail "sandboxfs process not found in process table"
  [[ "${pid1}" -ne "${pid2}" ]] || fail "sandboxfs was not restarted"

  # Want build to finish successfully and to clear the mount point.
  build_with_fake_sandboxfs "$(pwd)/log" --nosandbox_debug //hello
  local pid3="$(sandboxfs_pid)"
  [[ -z "${pid3}" ]] || fail "sandboxfs was not terminated"
}

function test_always_unmounted_on_exit() {
  create_hello_package

  # Want sandboxfs to be left mounted after a build with debugging on.
  build_with_fake_sandboxfs "$(pwd)/log" --sandbox_debug //hello
  grep -q "ARGS:" log || fail "sandboxfs was not run"
  grep -q "Terminated" log \
    && fail "sandboxfs process was terminated but should not have been"

  # Want Bazel to unmount the sandboxfs instance on exit no matter what.
  #
  # Note that we do not even tell Bazel where the sandboxfs binary lives
  # but we expect changes to the log of the currently-running sandboxfs
  # binary.  This is intentional to verify that the already-mounted
  # instance is the one shut down.
  bazel shutdown
  grep -q "Terminated" log \
    || fail "sandboxfs process was not terminated but should have been"
}

run_suite "sandboxfs-based sandboxing tests"
