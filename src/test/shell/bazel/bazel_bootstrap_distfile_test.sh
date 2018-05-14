#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# Test that bazel can be compiled out of the distribution artifact.
#

set -u
DISTFILE=$(rlocation io_bazel/${1#./})
shift 1

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function _log_progress() {
  if is_windows; then
    log_info "test_bootstrap: $*"
  fi
}

function test_bootstrap()  {
    _log_progress "start"
    local olddir=$(pwd)
    WRKDIR=$(mktemp -d ${TEST_TMPDIR}/bazelbootstrap.XXXXXXXX)
    mkdir -p "${WRKDIR}" || fail "Could not create workdir"
    trap "rm -rf \"$WRKDIR\"" EXIT
    cd "${WRKDIR}" || fail "Could not change to work directory"
    export SOURCE_DATE_EPOCH=1501234567
    _log_progress "unzip"
    unzip -q "${DISTFILE}"
    _log_progress "bootstrap"
    env EXTRA_BAZEL_ARGS="--curses=no --strategy=Javac=standalone" ./compile.sh \
        || fail "Expected to be able to bootstrap bazel"
    _log_progress "run"
    ./output/bazel version > "${TEST_log}" || fail "Generated bazel not working"
    ./output/bazel shutdown
    _log_progress "assert"
    expect_log "${SOURCE_DATE_EPOCH}"
    cd "${olddir}"
    _log_progress "done"
}

run_suite "bootstrap test"
