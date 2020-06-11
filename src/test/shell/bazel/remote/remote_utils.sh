#!/bin/bash
#
# Copyright 2020 The Bazel Authors. All rights reserved.
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
# Utilities for remote execution.

REMOTE_WORKER="$(rlocation io_bazel/src/tools/remote/worker)"

function start_worker() {
  work_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  cas_path=$(mktemp -d "${TEST_TMPDIR}/remote.XXXXXXXX")
  pid_file=$(mktemp -u "${TEST_TMPDIR}/remote.XXXXXXXX")
  worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
  native_lib="${BAZEL_RUNFILES}/src/main/native/"
  "${REMOTE_WORKER}" \
      --singlejar \
      --jvm_flag=-Djava.library.path="${native_lib}" \
      --work_path="${work_path}" \
      --cas_path="${cas_path}" \
      --listen_port=${worker_port} \
      --pid_file="${pid_file}" \
      "$@" >& $TEST_log &
  local wait_seconds=0
  until [[ -s "${pid_file}" || "$wait_seconds" -eq 30 ]]; do
    sleep 1
    wait_seconds=$((${wait_seconds} + 1))
  done
  if [ ! -s "${pid_file}" ]; then
    fail "Timed out waiting for remote worker to start."
  fi
}

function stop_worker() {
  if [ -s "${pid_file}" ]; then
    local pid=$(cat "${pid_file}")
    kill -9 "${pid}"
  rm -rf "${pid_file}"
  fi
  if [ -d "${work_path}" ]; then
    rm -rf "${work_path}"
  fi
  if [ -d "${cas_path}" ]; then
    rm -rf "${cas_path}"
  fi
}
