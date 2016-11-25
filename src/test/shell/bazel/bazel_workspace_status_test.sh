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
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_workspace_status_invalidation() {
  create_new_workspace

  local ok=$TEST_TMPDIR/ok.sh
  local bad=$TEST_TMPDIR/bad.sh
  cat > $ok <<EOF
#!/bin/bash
exit 0
EOF
  cat >$bad <<EOF
#!/bin/bash
exit 1
EOF
  chmod +x $ok $bad

  mkdir -p a
  cat > a/BUILD <<'EOF'
genrule(name="a", srcs=[], outs=["a.out"], stamp=1, cmd="touch $@")
EOF

  bazel build --stamp //a --workspace_status_command=$bad \
    && fail "build succeeded"
  bazel build --stamp //a --workspace_status_command=$ok \
    || fail "build failed"
}

function test_workspace_status_parameters() {
  create_new_workspace

  local cmd=$TEST_TMPDIR/status.sh
  cat > $cmd <<EOF
#!/bin/bash

echo BUILD_SCM_STATUS funky
EOF
  chmod +x $cmd

  mkdir -p a
  cat > a/BUILD <<'EOF'
genrule(
    name="a",
    srcs=[],
    outs=["a.out"],
    stamp=1,
    cmd="touch $@")
EOF

  bazel build --stamp //a --workspace_status_command=$cmd || fail "build failed"
  grep -sq "BUILD_SCM_STATUS funky" bazel-out/volatile-status.txt \
    || fail "BUILD_SCM_STATUS not found"
}

function test_workspace_status_cpp() {
  create_new_workspace

  local cmd=$TEST_TMPDIR/status.sh
  cat > $cmd <<EOF
#!/bin/bash

echo BUILD_SCM_STATUS funky
EOF
  chmod +x $cmd

  mkdir -p a
  cat > a/linkstamped_library.cc <<'EOF'
#include <string>

::std::string BuildScmStatus() { return BUILD_SCM_STATUS; }
EOF
  cat > a/verify_scm_status.cc <<'EOF'
#include <string>
#include <iostream>

::std::string BuildScmStatus();

int main() {
  ::std::cout << "BUILD_SCM_STATUS is: " << BuildScmStatus();

  return ("funky" == BuildScmStatus()) ? 0 : 1;
}
EOF

  cat > a/BUILD <<'EOF'
cc_library(
    name="linkstamped_library",
    linkstamp="linkstamped_library.cc")
cc_test(
    name="verify_scm_status",
    stamp=True,
    srcs=["verify_scm_status.cc"],
    deps=[":linkstamped_library"])
EOF

  bazel test --stamp //a:verify_scm_status --workspace_status_command=$cmd || fail "build failed"
}

function test_stable_and_volatile_status() {
  create_new_workspace
  cat >$TEST_TMPDIR/wsc.sh <<EOF
#!/bin/bash

cat $TEST_TMPDIR/status
EOF

  chmod +x $TEST_TMPDIR/wsc.sh

  cat > BUILD <<'EOF'
genrule(
    name = "a",
    srcs = [],
    outs = ["ao"],
    cmd="(echo volatile; cat bazel-out/volatile-status.txt; echo; echo stable; cat bazel-out/stable-status.txt; echo) > $@",
    stamp=1)
EOF

  cat >$TEST_TMPDIR/status <<EOF
STABLE_NAME alice
NUMBER 1
EOF

  bazel build --workspace_status_command=$TEST_TMPDIR/wsc.sh --stamp //:a || fail "build failed"
  assert_contains "STABLE_NAME alice" bazel-genfiles/ao
  assert_contains "NUMBER 1" bazel-genfiles/ao


  cat >$TEST_TMPDIR/status <<EOF
STABLE_NAME alice
NUMBER 2
EOF

  # Changes to volatile fields should not result in a rebuild
  bazel build --workspace_status_command=$TEST_TMPDIR/wsc.sh --stamp //:a || fail "build failed"
  assert_contains "STABLE_NAME alice" bazel-genfiles/ao
  assert_contains "NUMBER 1" bazel-genfiles/ao

  cat >$TEST_TMPDIR/status <<EOF
STABLE_NAME bob
NUMBER 3
EOF

  # Changes to stable fields should result in a rebuild
  bazel build --workspace_status_command=$TEST_TMPDIR/wsc.sh --stamp //:a || fail "build failed"
  assert_contains "STABLE_NAME bob" bazel-genfiles/ao
  assert_contains "NUMBER 3" bazel-genfiles/ao

}

run_suite "workspace status tests"
