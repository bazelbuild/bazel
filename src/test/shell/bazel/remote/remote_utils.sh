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
  work_path="${TEST_TMPDIR}/remote.work_path"
  cas_path="${TEST_TMPDIR}/remote.cas_path"
  pid_file="${TEST_TMPDIR}/remote.pid_file"
  mkdir -p "${work_path}"
  mkdir -p "${cas_path}"
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
  work_path="${TEST_TMPDIR}/remote.work_path"
  cas_path="${TEST_TMPDIR}/remote.cas_path"
  pid_file="${TEST_TMPDIR}/remote.pid_file"
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

# Pass in the root of the disk cache and count number of files under /ac directory
# output int to stdout
function count_disk_ac_files() {
  if [ -d "$1/ac" ]; then
    expr $(find "$1/ac" -type f | wc -l)
  else
    echo 0
  fi
}

# Pass in the root of the disk cache and count number of files under /cas directory
# output int to stdout
function count_disk_cas_files() {
  if [ -d "$1/cas" ]; then
    expr $(find "$1/cas" -type f | wc -l)
  else
    echo 0
  fi
}

function count_remote_ac_files() {
  if [ -d "$cas_path/ac" ]; then
    expr $(find "$cas_path/ac" -type f | wc -l)
  else
    echo 0
  fi
}

function count_remote_cas_files() {
  if [ -d "$cas_path/cas" ]; then
    expr $(find "$cas_path/cas" -type f | wc -l)
  else
    echo 0
  fi
}

function remote_cas_file_exist() {
  local file=$1
  [ -f "$cas_path/cas/${file:0:2}/$file" ]
}

function append_remote_cas_files() {
  find "$cas_path/cas" -type f >> $1
}

function delete_remote_cas_files() {
  rm -rf "$cas_path/cas"
}