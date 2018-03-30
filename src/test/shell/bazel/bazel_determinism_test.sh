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

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# sha1sum is easily twice as fast as shasum, but not always available on macOS.
if hash sha1sum 2>/dev/null; then
  shasum="sha1sum"
elif hash shasum 2>/dev/null; then
  shasum="shasum"
else
  fail "Could not find sha1sum or shasum on the PATH"
fi

function hash_outputs() {
  # runfiles/MANIFEST & runfiles_manifest contain absolute path, ignore.
  # ar on OS-X is non-deterministic, ignore .a files.
  find bazel-bin/ bazel-genfiles/ \
      -type f \
      -a \! -name MANIFEST \
      -a \! -name '*.runfiles_manifest' \
      -a \! -name '*.a' \
      -exec $shasum {} + \
      | awk '{print $2, $1}' \
      | sort \
      | sed "s://:/:"
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
    hash_outputs >"${TEST_TMPDIR}/sum1"
    ./bazel1 clean --expunge
    ./bazel1 build --nostamp src:bazel
    hash_outputs >"${TEST_TMPDIR}/sum2"
    if ! (diff -q "${TEST_TMPDIR}/sum1" "${TEST_TMPDIR}/sum2"); then
      diff -U0 "${TEST_TMPDIR}/sum1" "${TEST_TMPDIR}/sum2" >$TEST_log
      fail "Non deterministic outputs found!"
    fi
}

run_suite "determinism test"
