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

function test_bootstrap()  {
    local olddir=$(pwd)
    WRKDIR=$(mktemp -d ${TEST_TMPDIR}/bazelbootstrap.XXXXXXXX)
    mkdir -p "${WRKDIR}" || fail "Could not create workdir"
    trap "rm -rf \"$WRKDIR\"" EXIT
    cd "${WRKDIR}" || fail "Could not change to work directory"
    unzip -q ${DISTFILE}
    find . -type f -exec chmod u+w {} \;
    ./compile.sh || fail "Expected to be able to bootstrap bazel"
    ./output/bazel version || fail "Generated bazel not working"
    ./output/bazel shutdown
    cd "${olddir}"
}

run_suite "bootstrap test"
