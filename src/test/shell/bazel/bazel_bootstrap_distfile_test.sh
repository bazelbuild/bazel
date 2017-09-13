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

if [ "${JAVA_VERSION:-}" == "1.7" ] ; then
  echo "Warning: bootstrapping not tested for java 1.7"
  exit 0
fi

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function log_progress() {
  # TODO(laszlocsomor): remove this method after we fixed
  # https://github.com/bazelbuild/bazel/issues/3618. I added this method only to
  # catch that bug.
  echo "DEBUG[$0 ($date)] test_bootstrap, $@"
}

function test_bootstrap()  {
    local olddir=$(pwd)
    WRKDIR=$(mktemp -d ${TEST_TMPDIR}/bazelbootstrap.XXXXXXXX)
    log_progress "WRKDIR=($WRKDIR)"
    log_progress "TMPDIR=($TMPDIR)"
    log_progress "TEST_TMPDIR=($TEST_TMPDIR)"
    mkdir -p "${WRKDIR}" || fail "Could not create workdir"
    trap "rm -rf \"$WRKDIR\"" EXIT
    cd "${WRKDIR}" || fail "Could not change to work directory"
    unzip -q ${DISTFILE}
    log_progress "unzipped DISTFILE=($DISTFILE)"
    find . -type f -exec chmod u+w {} \;
    log_progress "running compile.sh"
    ./compile.sh || fail "Expected to be able to bootstrap bazel"
    log_progress "done running compile.sh, WRKDIR contains this many files: $(find "$WRKDIR" -type f | wc -l)"
    log_progress "running bazel info install_base"
    # TODO(laszlocsomor): remove the "bazel info install_base" call at the same
    # time as removing log_progress, i.e. after we fixed
    # https://github.com/bazelbuild/bazel/issues/3618
    ./output/bazel info install_base || fail "Generated bazel not working"
    log_progress "done running bazel info install_base, WRKDIR contains this many files: $(find "$WRKDIR" -type f | wc -l)"
    log_progress "running bazel info version"
    ./output/bazel version || fail "Generated bazel not working"
    log_progress "done running bazel version, WRKDIR contains this many files: $(find "$WRKDIR" -type f | wc -l)"
    log_progress "running bazel shutdown"
    ./output/bazel shutdown
    log_progress "done running bazel shutdown"
    cd "${olddir}"
}

run_suite "bootstrap test"
