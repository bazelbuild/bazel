#!/bin/bash
# Copyright 2021 The Bazel Authors. All rights reserved.
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

# Tests to check that spurious dependencies are not added to
# //src/main/java/com/google/devtools/build/skydoc:skydoc_lib.
#
# This test is a response to https://github.com/bazelbuild/stardoc/issues/64,
# where an unwanted Bazel dependency caused downstream uses of the Stardoc
# binary to fail. Since this is a denylist, it won't catch all unwanted
# dependencies.

# Load the dependencies of
# //src/main/java/com/google/devtools/build/skydoc:skydoc_lib in the current
# workspace using the output of genquery //src/test/shell/bazel:stardoc_deps
# and removing everything under @bazel_tools because the exact contents of the
# latter depends on the bazel binary used to run the test.
# Sort the targets for a deterministic diffing experience.
current_deps="${TEST_TMPDIR}/current_deps"
grep -v "^@bazel_tools//" \
  "${TEST_SRCDIR}/io_bazel/src/test/shell/bazel/stardoc_deps" \
  | sort >"${current_deps}"

# Check the current dependencies for forbidden deps (from Bazel's @google
# and @remote APIs).
disallowed_deps=$(grep "^//google/bytestream\|^//google/longrunning\|^//google/rpc\|^//third_party/bazel_remoteapis" \
  "${current_deps}") && \
  echo "The following spurious dependencies were added to \
  //src/main/java/com/google/devtools/build/skydoc:skydoc_lib \
  and should be removed: ${disallowed_deps}" && exit

echo "PASS"
