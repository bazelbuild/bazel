#!/usr/bin/env bash
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
  # The remote worker background process needs a separate log file in
  # order to prevent concurrent logging to the same $TEST_log.
  remote_worker_log_file="${TEST_TMPDIR}/remote_worker.log"
  mkdir -p "${work_path}"
  mkdir -p "${cas_path}"
  
  if [ -s "${pid_file}" ]; then
      cat "${pid_file}"
      ps aux
      fail "There is already a worker pid_file."
  fi

  worker_port=$(pick_random_unused_tcp_port) || fail "no port found"
  native_lib="${BAZEL_RUNFILES}/src/main/native/"
  "${REMOTE_WORKER}" \
      --singlejar \
      --jvm_flag=-Djava.library.path="${native_lib}" \
      --work_path="${work_path}" \
      --cas_path="${cas_path}" \
      --listen_port="${worker_port}" \
      --pid_file="${pid_file}" \
      "$@" >> "${remote_worker_log_file}" 2>&1 &
  local background_pid=$!
  echo "Starting remote worker: pid=${background_pid}" port=${worker_port} >> "${TEST_log}"
  if ! wait_for_file_to_have_content "${pid_file}"; then
    echo "Calling kill -SIGKILL ${background_pid}"
    kill -SIGKILL "${background_pid}" 2>/dev/null || true
    # Import the remote worker log to ensure any startup error messages
    # are displayed along with the timeout failure.
    import_to_test_log "${remote_worker_log_file}"
    fail "Timed out waiting for remote worker to start. Expected pid ${background_pid}"
  fi
}

function stop_worker() {
  work_path="${TEST_TMPDIR}/remote.work_path"
  cas_path="${TEST_TMPDIR}/remote.cas_path"
  pid_file="${TEST_TMPDIR}/remote.pid_file"
  remote_worker_log_file="${TEST_TMPDIR}/remote_worker.log"
  local failure=""
  if [ -s "${pid_file}" ]; then
    local pid=$(cat "${pid_file}")
    echo "Stopping remote worker: pid=${pid}" >> "${TEST_log}"
    kill -TERM "${pid}" || true
    # Waiting gives the remote worker an opportunity to flush logs and prevents
    # interference between workers by ensuring that the previous worker has
    # fully completed before starting any new worker.
    if ! wait_for_pid_to_terminate "${pid}"; then
        echo "WARNING: Remote worker pid ${pid} was not responding to SIGTERM signal."
        echo "WARNING: Terminating remote worker abruptly. Logs may be incomplete."
        echo "Calling kill -SIGKILL ${pid}"        
        kill -SIGKILL "${pid}" 2>/dev/null || true
        if ! wait_for_pid_to_terminate "${pid}"; then
           failure="Remote worker pid ${pid} is still alive after SIGKILL signal."
        fi
    fi
    rm -f "${pid_file}"
  fi
  if [ -f "${remote_worker_log_file}" ]; then
    # Import the remote worker log file into $TEST_log so that remote worker messages
    # are included when unittest.bash displays the output for failed tests. This
    # also enables the use of expect_log_* functions from unittest.bash on remote
    # worker log content.
    import_to_test_log "${remote_worker_log_file}"
    rm -f "${remote_worker_log_file}"
  fi
  if [ -d "${work_path}" ]; then
    rm -rf "${work_path}"
  fi
  if [ -d "${cas_path}" ]; then
    rm -rf "${cas_path}"
  fi
  if [ -n "$failure" ]; then
      fail "$failure"
  fi
}

function import_to_test_log() {
    local log_file=$1
    if [ -s "${log_file}" ]; then
        echo "Imported from: ${log_file} to test log:" >> "${TEST_log}"
        cat "${log_file}" >> "${TEST_log}"
        echo >> "${TEST_log}"
    else
        echo "No ${log_file} content." >> $TEST_log
    fi
}

function wait_for_condition() {
    set -x # TODO
    local condition="$1"
    local grace_seconds=30
    local poll_interval_seconds=0.2
    local remaining_polls=$(awk "BEGIN{print int($grace_seconds/$poll_interval_seconds)}")
    while (( remaining_polls > 0 )); do
        # if eval "$condition" 2>/dev/null; then
        if (set +e; eval "$condition" 2>/dev/null); then            
            set +x # TODO
            return 0  # Condition fulfilled
        fi
        sleep "$poll_interval_seconds"
        (( remaining_polls-- ))
    done
    set +x # TODO
    return 1  # Timed out

    
    
    #while ! eval "$condition" 2>/dev/null && [ $remaining_polls -gt 0 ]; do
    #    sleep "$poll_interval_seconds"
    #    remaining_polls=$((remaining_polls - 1))
    #done
    #if eval "$condition" 2>/dev/null; then
    #    return 0  # Condition fulfilled
    #else
    #    return 1  # Timed out
    #fi
}

function wait_for_file_to_have_content() {
    local file_path="$1"
    wait_for_condition "[ -s \"${file_path}\" ]"
}

function wait_for_pid_to_terminate() {
    local pid="$1"
    wait_for_condition "! kill -0 \"${pid}\""
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
