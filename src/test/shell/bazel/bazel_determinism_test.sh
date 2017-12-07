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
# Test that bootstrapping bazel is a fixed point
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

function hash_outputs() {
  [ -n "${BAZEL_TEST_XTRACE:-}" ] && set +x  # Avoid garbage in the output
  # runfiles/MANIFEST & runfiles_manifest contain absolute path, ignore.
  # ar on OS-X is non-deterministic, ignore .a files.
  for i in $(find bazel-bin/ -type f -a \! -name MANIFEST -a \! -name '*.runfiles_manifest' -a \! -name '*.a'); do
    sha256sum $i
  done
  for i in $(find bazel-genfiles/ -type f); do
    sha256sum $i
  done
  [ -n "${BAZEL_TEST_XTRACE:-}" ] && set -x
}

function get_outputs_sum() {
  hash_outputs | sort -k 2
}

function test_determinism()  {
    local olddir=$(pwd)
    WRKDIR=$(mktemp -d ${TEST_TMPDIR}/bazelbootstrap.XXXXXXXX)
    mkdir -p "${WRKDIR}" || fail "Could not create workdir"
    trap "rm -rf \"$WRKDIR\"" EXIT
    cd "${WRKDIR}" || fail "Could not change to work directory"
    unzip -q ${DISTFILE}
    # Compile bazel a first time
    bazel build --nostamp src:bazel
    cp bazel-bin/src/bazel bazel1
    get_outputs_sum >"${TEST_TMPDIR}/sum1"
    ./bazel1 clean --expunge
    ./bazel1 build --nostamp src:bazel
    get_outputs_sum >"${TEST_TMPDIR}/sum2"
    if ! (diff -q "${TEST_TMPDIR}/sum1" "${TEST_TMPDIR}/sum2"); then
      diff -U0 "${TEST_TMPDIR}/sum1" "${TEST_TMPDIR}/sum2" >$TEST_log
      fail "Non deterministic outputs found!"
    fi
}

run_suite "determinism test"
