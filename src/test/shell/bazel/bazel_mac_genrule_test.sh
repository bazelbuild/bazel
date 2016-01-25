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

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

if [ "${PLATFORM}" != "darwin" ]; then
  echo "This test suite requires running on OS X" >&2
  exit 0
fi

function test_developer_dir() {
  mkdir -p package
  # TODO(bazel-team): Local host genrule execution should convert
  # the XCODE_VERSION_OVERRIDE environment variable to a DEVELOPER_DIR
  # absolute path, preventing us from needing to set this environment
  # variable ourself.
  export DEVELOPER_DIR="/Applications/Xcode_5.4.app/Contents/Developer/"

  echo "\\\${XCODE_VERSION_OVERRIDE}" ${DEVELOPER_DIR}

  cat > package/BUILD <<EOF
genrule(
  name = "foo",
  srcs = [],
  outs = ["developerdir.out"],
  tags = ["requires-darwin"],
  cmd = "DEVELOPER_DIR='${DEVELOPER_DIR}' && echo \$\${XCODE_VERSION_OVERRIDE} \$(DEVELOPER_DIR) > \$@",
)
EOF

  bazel build -s //package:foo --xcode_version=5.4 || fail "Should build"
  assert_equals "5.4 /Applications/Xcode_5.4.app/Contents/Developer/" \
      "$(cat bazel-genfiles/package/developerdir.out)" || \
      fail "Unexpected value for DEVELOPER_DIR make variable"
}

run_suite "genrule test suite for Apple Mac hosts"
