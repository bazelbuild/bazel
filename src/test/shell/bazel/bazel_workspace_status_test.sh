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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_workspace_status_parameters() {
  create_new_workspace

  local cmd=`mktemp $TEST_TMPDIR/wsc-XXXXXXXX`
  cat > $cmd <<EOF
#!/usr/bin/env bash

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

function test_workspace_status_overrides() {
  create_new_workspace

  local cmd=`mktemp $TEST_TMPDIR/wsc-XXXXXXXX`
  cat > $cmd <<EOF
#!/usr/bin/env bash

echo BUILD_USER fake_user
echo BUILD_HOST fake_host
echo BUILD_EMBED_LABEL fake_label
echo BUILD_TIMESTAMP 17
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
  cat bazel-out/volatile-status.txt > "$TEST_log"
  expect_log "BUILD_TIMESTAMP 17"
  cat bazel-out/stable-status.txt > "$TEST_log"
  expect_log "BUILD_USER fake_user"
  expect_log "BUILD_HOST fake_host"
  expect_log "BUILD_EMBED_LABEL fake_label"
}

function test_workspace_status_cpp() {
  create_new_workspace

  local cmd=`mktemp $TEST_TMPDIR/wsc-XXXXXXXX`
  cat > $cmd <<EOF
#!/usr/bin/env bash

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
    stamp=1,
    srcs=["verify_scm_status.cc"],
    deps=[":linkstamped_library"])
EOF

  bazel test --stamp //a:verify_scm_status --workspace_status_command=$cmd || fail "build failed"
}

function test_errmsg() {
  create_new_workspace
  cat > BUILD <<'EOF'
genrule(
    name = "a",
    srcs = [],
    outs = ["ao"],
    cmd="echo whatever > $@",
    stamp=1,
)
EOF

  bazel build --workspace_status_command=$TEST_TMPDIR/wscmissing.sh --stamp //:a &> $TEST_log \
    && fail "build succeeded"
  expect_log "wscmissing.sh: No such file or directory\|wscmissing.sh: not found"
}


function test_stable_and_volatile_status() {
  create_new_workspace
  local wsc=`mktemp $TEST_TMPDIR/wsc-XXXXXXXX`
  cat >$wsc <<EOF
#!/usr/bin/env bash

cat $TEST_TMPDIR/status
EOF

  chmod +x $wsc

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

  bazel build --workspace_status_command=$wsc --stamp //:a || fail "build failed"
  assert_contains "STABLE_NAME alice" bazel-genfiles/ao
  assert_contains "NUMBER 1" bazel-genfiles/ao


  cat >$TEST_TMPDIR/status <<EOF
STABLE_NAME alice
NUMBER 2
EOF

  # Changes to volatile fields should not result in a rebuild
  bazel build --workspace_status_command=$wsc --stamp //:a || fail "build failed"
  assert_contains "STABLE_NAME alice" bazel-genfiles/ao
  assert_contains "NUMBER 1" bazel-genfiles/ao

  cat >$TEST_TMPDIR/status <<EOF
STABLE_NAME bob
NUMBER 3
EOF

  # Changes to stable fields should result in a rebuild
  bazel build --workspace_status_command=$wsc --stamp //:a || fail "build failed"
  assert_contains "STABLE_NAME bob" bazel-genfiles/ao
  assert_contains "NUMBER 3" bazel-genfiles/ao

  # Test that generated timestamp is the same as its formatted version.
  volatile_file="bazel-out/volatile-status.txt"
  bazel build --stamp //:a || fail "build failed"
  assert_contains "BUILD_TIMESTAMP" $volatile_file
  assert_contains "FORMATTED_DATE" $volatile_file
  # Read key value pairs.
  timestamp_key_value=$(sed "1q;d" $volatile_file)
  formatted_timestamp_key_value=$(sed "2q;d" $volatile_file)
  # Extract values of the formatted date and timestamp.
  timestamp=${timestamp_key_value#* }
  formatted_date=${formatted_timestamp_key_value#* }
  if [[ $(uname -s) == "Darwin" ]]
  then
    timestamp_formatted_date=$(date -u -r "$timestamp" +'%Y %b %d %H %M %S %a')
  else
    timestamp_formatted_date=$(date -u -d "@$timestamp" +'%Y %b %d %H %M %S %a')
  fi

  if [[ $timestamp_formatted_date != $formatted_date ]]
  then
    fail "Timestamp formatted date: $timestamp_formatted_date differs from workspace module provided formatted date: $formatted_date"
  fi
}

function test_env_var_in_workspace_status() {
  create_new_workspace
  local wsc=`mktemp $TEST_TMPDIR/wsc-XXXXXXXX`
  cat >$wsc <<'EOF'
#!/usr/bin/env bash

echo "STABLE_ENV" ${STABLE_VAR}
echo "VOLATILE_ENV" ${VOLATILE_VAR}
EOF

  chmod +x $wsc

  cat > BUILD <<'EOF'
genrule(
    name = "a",
    srcs = [],
    outs = ["ao"],
    cmd="(echo volatile; cat bazel-out/volatile-status.txt; echo; echo stable; cat bazel-out/stable-status.txt; echo) > $@",
    stamp=1)
EOF

  STABLE_VAR=alice VOLATILE_VAR=one bazel build --workspace_status_command=$wsc --stamp //:a || fail "build failed"
  assert_contains "STABLE_ENV alice" bazel-out/stable-status.txt
  assert_contains "VOLATILE_ENV one" bazel-out/volatile-status.txt
  assert_contains "STABLE_ENV alice" bazel-genfiles/ao
  assert_contains "VOLATILE_ENV one" bazel-genfiles/ao

  # Changes to the env var should be reflected into the stable-status file, and thus trigger a rebuild
  STABLE_VAR=bob VOLATILE_VAR=two bazel build --workspace_status_command=$wsc --stamp //:a || fail "build failed"
  assert_contains "STABLE_ENV bob" bazel-out/stable-status.txt
  assert_contains "VOLATILE_ENV two" bazel-out/volatile-status.txt
  assert_contains "STABLE_ENV bob" bazel-genfiles/ao
  assert_contains "VOLATILE_ENV two" bazel-genfiles/ao

  # Changes to volatile fields should not result in a rebuild (but should update the stable & volatile status files)
  STABLE_VAR=bob VOLATILE_VAR=three bazel build --workspace_status_command=$wsc --stamp //:a || fail "build failed"
  assert_contains "STABLE_ENV bob" bazel-out/stable-status.txt
  assert_contains "VOLATILE_ENV three" bazel-out/volatile-status.txt
  # We did not rebuild, so the output remains at the previous values
  assert_contains "STABLE_ENV bob" bazel-genfiles/ao
  assert_contains "VOLATILE_ENV two" bazel-genfiles/ao

}

run_suite "workspace status tests"
