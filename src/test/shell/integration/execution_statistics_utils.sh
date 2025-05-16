#!/usr/bin/env bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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

readonly CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly STATS_PROTO_PATH="${CURRENT_DIR}/../../../main/protobuf/execution_statistics.proto"
readonly STATS_PROTO_DIR="$(cd "$(dirname "${STATS_PROTO_PATH}")" && pwd)"

# Checks that user CPU time and system CPU time, as read from an execution
# statistics proto file, are within expected bounds.
#
# This relies on ${protoc_compiler} being set (currently set in testenv.sh)
#
function assert_execution_time_in_range() {
  local utime_low="$1"; shift
  local utime_high="$1"; shift
  local stime_low="$1"; shift
  local stime_high="$1"; shift
  local stats_out_path="$1"; shift

  if ! [[ -e "${stats_out_path}" ]]; then
    fail "Stats file not found: '${stats_out_path}'"
  fi

  "${protoc_compiler}" --proto_path="${STATS_PROTO_DIR}" \
      --decode tools.protos.ExecutionStatistics execution_statistics.proto \
      < "${stats_out_path}" > "${stats_out_decoded_path}"

  if ! [[ -e "${stats_out_decoded_path}" ]]; then
    fail "Decoded stats file not found: '${stats_out_decoded_path}'"
  fi

  local utime=0
  if grep -q utime_sec "${stats_out_decoded_path}"; then
    utime="$(grep utime_sec ${stats_out_decoded_path} | cut -f2 -d':' | \
      tr -dc '0-9')"
  fi

  local stime=0
  if grep -q stime_sec "${stats_out_decoded_path}"; then
    stime="$(grep stime_sec ${stats_out_decoded_path} | cut -f2 -d':' | \
      tr -dc '0-9')"
  fi

  if ! [[ ${utime} -ge ${utime_low} && ${utime} -le ${utime_high} ]]; then
    fail "reported utime of '${utime}' is out of expected range"
  fi
  if ! [[ ${stime} -ge ${stime_low} && ${stime} -le ${stime_high} ]]; then
    fail "reported stime of '${stime}' is out of expected range"
  fi
}
