#!/bin/bash
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

# Tests to check if dependencies of //src:embedded_tools_srcs are modified
# due to the current change.
#
# The current allowed dependencies of //src:embedded_tools_srcs are stored in
# //src/test/shell/bazel/testdata/embedded_tools_srcs_deps.
#
# If this test failed, and you are sure that the modification to the
# dependencies of //src:embedded_tools_srcs is valid, you may update
# //src/test/shell/bazel/testdata/embedded_tools_srcs_deps
# to incorporate the dependency change due to your change.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if [ "${PLATFORM-}" != "darwin" ] && [ "${PLATFORM-}" != "linux" ]; then
  echo "We only run this test on a Darwin or Linux machine."
  exit 0
fi

# Load the dependencies of //src:embedded_tools_srcs in the current workspace
# using the output of genquery //src/test/shell/bazel:embedded_tools_deps
# and removing everything under @bazel_tools because the exact contents of the
# latter depends on the bazel binary used to run the test.
# Sort the targets for a deterministic diffing experience.
current_deps="${TEST_TMPDIR}/current_deps"
grep -v "^@bazel_tools//\|^@remote_java_tools\|^@debian_cc_deps" \
  "$(rlocation io_bazel/src/test/shell/bazel/embedded_tools_deps)" \
  | sort >"${current_deps}"

# TODO: This is a temporary hack to make this test works both before and after
# https://github.com/bazelbuild/bazel/pull/11300
# Remove the following line after the PR is merged.
sed -i.bak s/\:zlib$/\:zlib_checked_in/ "${current_deps}"

# Load the current allowed dependencies of //src:embedded_tools_srcs
allowed_deps=${testdata_path}/embedded_tools_srcs_deps

diff_result=$(diff -ay --suppress-common-lines ${current_deps} \
  <(sort ${allowed_deps})) || \
  fail "Dependencies of //src:embedded_tools_srcs are modified. The diff \
between the new dependencies and the current allowed dependencies is \
$(printf "\n${diff_result}\nThe new dependencies are ")$(cat ${current_deps})"

echo "PASS"
