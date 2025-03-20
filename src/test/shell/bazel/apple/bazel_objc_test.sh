#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

if [ "${PLATFORM}" != "darwin" ]; then
  echo "This test suite requires running on OS X" >&2
  exit 0
fi

function test_xcodelocator_embedded_tool() {
  rm -rf ios
  mkdir -p ios

  cat >ios/BUILD <<EOF
genrule(
    name = "invoke_tool",
    srcs = ["@bazel_tools//tools/osx:xcode-locator"],
    outs = ["tool_output"],
    cmd = "\$< > \$@",
    tags = ["requires-darwin"],
)
EOF

  bazel build --verbose_failures //ios:invoke_tool >$TEST_log 2>&1 \
      || fail "should be able to resolve xcode-locator"
}

run_suite "objc test suite"
