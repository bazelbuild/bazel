#!/usr/bin/env bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh"

TESTER="$(rlocation io_bazel/src/main/java/com/google/devtools/build/lib/skyframe/packages/testing/BazelPackageLoaderTester)"

function test_bazel_package_loader() {
  install_base="$(bazel info install_base)"
  mkdir foo bar
  cat > foo/BUILD << 'EOF'
filegroup(name = 'a')
filegroup(name = 'b')
filegroup(name = 'c')
EOF
  cat > bar/BUILD << 'EOF'
filegroup(name = 'x')
filegroup(name = 'y')
filegroup(name = 'z')
EOF

  # Skip WORKSPACE suffix to avoid loading rules_java_builtin repo.
  cat > WORKSPACE <<EOF
# __SKIP_WORKSPACE_SUFFIX__
EOF

  "$TESTER" "$install_base" foo bar >& "$TEST_log"
  expect_log //foo:a
  expect_log //foo:b
  expect_log //foo:c
  expect_log //bar:x
  expect_log //bar:y
  expect_log //bar:z
}

run_suite "End-to-end tests for BazelPackageLoader"
