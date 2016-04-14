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
# An end-to-end test that Bazel's experimental UI produces reasonable output.

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/testenv.sh \
  || { echo "testenv.sh not found!" >&2; exit 1; }

create_and_cd_client
put_bazel_on_path
write_default_bazelrc

#### SETUP #############################################################

set -e

function set_up() {
  mkdir -p pkg
  cat > pkg/true.sh <<EOF
#!/bin/sh
exit 0
EOF
  chmod 755 pkg/true.sh
  cat > pkg/slow.sh <<EOF
#!/bin/sh
sleep 10
exit 0
EOF
  chmod 755 pkg/slow.sh
  cat > pkg/false.sh <<EOF
#!/bin/sh
exit 1
EOF
  chmod 755 pkg/false.sh
  cat > pkg/BUILD <<EOF
sh_test(
  name = "true",
  srcs = ["true.sh"],
)
sh_test(
  name = "slow",
  srcs = ["slow.sh"],
)
sh_test(
  name = "false",
  srcs = ["false.sh"],
)
genrule(
  name = "gentext",
  outs = ["gentext.txt"],
  cmd = "echo here be dragons > \"\$@\""
)
EOF
}

#### TESTS #############################################################

function test_basic_progress() {
  bazel test --experimental_ui --curses=yes --color=yes pkg:true 2>$TEST_log || fail "bazel test failed"
  # some progress indicator is shown
  expect_log '\[[0-9,]* / [0-9,]*\]'
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
}

function test_basic_progress_no_curses() {
  bazel test --experimental_ui --curses=no --color=yes pkg:true 2>$TEST_log \
    || fail "bazel test failed"
  # some progress indicator is shown
  expect_log '\[[0-9,]* / [0-9,]*\]'
  # cursor is not moved up
  expect_not_log $'\x1b\[1A'
  # no line is deleted
  expect_not_log $'\x1b\[K'
  # but some green color is used
  expect_log $'\x1b\[32m'
}

function test_no_curses_no_linebreak() {
  bazel test --experimental_ui --curses=no --color=yes --terminal_columns=9 \
    pkg:true 2>$TEST_log || fail "bazel test failed"
  # expect a long-ish status line
  expect_log '\[[0-9,]* / [0-9,]*\]......'
}

function test_pass() {
  bazel test --experimental_ui --curses=yes --color=yes pkg:true >$TEST_log || fail "bazel test failed"
  # PASS is written in green on the same line as the test target
  expect_log 'pkg:true.*'$'\x1b\[32m''.*PASS'
}

function test_fail() {
  bazel test --experimental_ui --curses=yes --color=yes pkg:false >$TEST_log && fail "expected failure"
  # FAIL is written in red bold on the same line as the test target
  expect_log 'pkg:false.*'$'\x1b\[31m\x1b\[1m''.*FAIL'
}

function test_info_spacing() {
  # Verify that the output of "bazel info" is suitable for backtick escapes,
  # in particular free carriage-return characters.
  BAZEL_INFO_OUTPUT=XXX`bazel info --experimental_ui workspace`XXX
  echo "$BAZEL_INFO_OUTPUT" | grep -q 'XXX[^'$'\r'']*XXX' \
    || fail "bazel info output spaced as $BAZEL_INFO_OUTPUT"
}

function test_query_spacing() {
  # Verify that the output of "bazel query" is suitable for consumption by
  # other tools, i.e., contains only result lines, separated only by newlines.
  BAZEL_QUERY_OUTPUT=`bazel query --experimental_ui 'deps(//pkg:true)'`
  echo "$BAZEL_QUERY_OUTPUT" | grep -q -v '^[@/]' \
   && fail "bazel query output is >$BAZEL_QUERY_OUTPUT<"
  echo "$BAZEL_QUERY_OUTPUT" | grep -q $'\r' \
   && fail "bazel query output is >$BAZEL_QUERY_OUTPUT<"
  true
}

function test_clean_nobuild {
  bazel clean --experimental_ui 2>$TEST_log \
   || fail "bazel shutdown failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_clean_color_nobuild {
  bazel clean --experimental_ui --color=yes 2>$TEST_log \
   || fail "bazel shutdown failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_help_nobuild {
  bazel help --experimental_ui 2>$TEST_log \
   || fail "bazel help failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_help_color_nobuild {
  bazel help --experimental_ui --color=yes 2>$TEST_log \
   || fail "bazel help failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_subcommand {
  bazel clean || fail "bazel clean failed"
  bazel build --experimental_ui -s pkg:gentext 2>$TEST_log \
    || fail "bazel build failed"
  expect_log "here be dragons"
}

function test_subcommand_notdefault {
  bazel clean || fail "bazel clean failed"
  bazel build --experimental_ui pkg:gentext 2>$TEST_log \
    || fail "bazel build failed"
  expect_not_log "dragons"
}

function test_loading_progress {
  bazel clean || fail "bazel clean failed"
  bazel test --experimental_ui \
    --experimental_skyframe_target_pattern_evaluator pkg:true 2>$TEST_log \
    || fail "bazel test failed"
  # some progress indicator is shown during loading
  expect_log 'Loading.*[0-9,]* / [0-9,]*'
}

function test_failure_scrollback_buffer_curses {
  bazel clean || fail "bazel clean failed"
  bazel test --experimental_ui --curses=yes --color=yes \
    --nocache_test_results pkg:false pkg:slow 2>$TEST_log \
    && fail "expected failure"
  # Some line starting with FAIL in red bold replaces a previous one
  # (the old progress bar that is deleted
  expect_log $'\x1b\[K\x1b\[31m\x1b\[1mFAIL:'
}

function test_failure_scrollback_buffer {
  bazel clean || fail "bazel clean failed"
  bazel test --experimental_ui --curses=no --color=yes \
    --nocache_test_results pkg:false pkg:slow 2>$TEST_log \
    && fail "expected failure"
  # Some line starts with FAIL in red bold.
  expect_log '^'$'\x1b\[31m\x1b\[1mFAIL:'
}


run_suite "Integration tests for bazel's experimental UI"
