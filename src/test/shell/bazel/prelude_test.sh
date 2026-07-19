#!/usr/bin/env bash
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
#
# An end-to-end test of the behavior of tools/build_rules/prelude_bazel.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

#### TESTS #############################################################

function test_prelude() {

  mkdir -p tools/build_rules
  touch tools/build_rules/BUILD

  cat > tools/build_rules/prelude_bazel << EOF
PRELUDE_VAR = 'from test_prelude'
EOF

  cat > BUILD << EOF
genrule(
  name = 'gr',
  srcs = [],
  outs = ['gr.out'],
  cmd = 'echo "%s" > \$(location gr.out)' % PRELUDE_VAR,
)
EOF

  bazel build :gr >&$TEST_log 2>&1 || fail "build failed"

  output=$(cat bazel-genfiles/gr.out)
  check_eq "from test_prelude" "$output" "unexpected output in gr.out"
}

function test_prelude_external_repository() {
  cat > MODULE.bazel << EOF
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
  name = "imported_workspace",
  path = "$PWD/imported_workspace",
)
EOF

  mkdir -p tools/build_rules
  touch tools/build_rules/BUILD

  cat > tools/build_rules/prelude_bazel << EOF
PRELUDE_VAR = 'from test_prelude_external_repository, outer workspace'
EOF

  cat > BUILD << EOF
genrule(
  name = 'gr',
  srcs = [],
  outs = ['gr.out'],
  cmd = 'echo "outer %s" > \$(location gr.out)' % PRELUDE_VAR,
)
EOF

  mkdir -p imported_workspace
  touch imported_workspace/REPO.bazel

  mkdir -p imported_workspace/tools/build_rules
  touch imported_workspace/tools/build_rules/BUILD

  cat > imported_workspace/tools/build_rules/prelude_bazel << EOF
PRELUDE_VAR = 'from test_prelude_external_repository, inner workspace'
EOF

  cat > imported_workspace/BUILD << EOF
genrule(
  name = 'gr_inner',
  srcs = [],
  outs = ['gr_inner.out'],
  cmd = 'echo "inner %s" > \$(location gr_inner.out)' % PRELUDE_VAR,
)
EOF

  bazel build :gr @imported_workspace//:gr_inner >&$TEST_log 2>&1 || fail "build failed"

  output=$(cat bazel-genfiles/gr.out)
  check_eq "outer from test_prelude_external_repository, outer workspace" "$output" "unexpected output in gr.out"

  output=$(cat bazel-genfiles/external/+local_repository+imported_workspace/gr_inner.out)
  check_eq "inner from test_prelude_external_repository, inner workspace" "$output" "unexpected output in gr_inner.out"
}

run_suite "prelude"
