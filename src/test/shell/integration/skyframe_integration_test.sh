#!/usr/bin/env bash
#
# Copyright 2026 The Bazel Authors. All rights reserved.
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
# skyframe_integration_test.sh: a few end-to-end tests for the Skyframe-based
# codepath in Bazel.

# --- begin runfiles.bash initialization v3 ---
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_java() {
  add_rules_java "MODULE.bazel"
  mkdir -p java/hello
  cat >java/hello/BUILD <<EOF
load("@rules_java//java:java_binary.bzl", "java_binary")
java_binary(name = 'hello',
    srcs = ['Hello.java'],
    main_class = 'hello.Hello')
EOF

  cat >java/hello/Hello.java <<EOF
package hello;
public class Hello {
  public static void main(String[] args) {
    System.out.println("Hello, World!");
  }
}
EOF

  bazel --batch build //java/hello:hello || fail "build failed"
  (
    unset JAVA_RUNFILES TEST_SRCDIR RUNFILES_MANIFEST_FILE RUNFILES_MANIFEST_ONLY RUNFILES_DIR
    "${PRODUCT_NAME}-bin/java/hello/hello"
  ) | grep -q 'Hello, World!' || fail "comparison failed"
}

function test_sh_binary() {
  add_rules_shell "MODULE.bazel"
  mkdir sh
  cat > sh/BUILD <<EOF
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
sh_binary(name = "bin", srcs = ["bin.sh"])
EOF
  cat > sh/bin.sh <<EOF
#!/bin/sh
echo "hi"
EOF

  chmod a+x sh/bin.sh

  bazel build //sh:bin || fail "build failed"
}

function test_loading_progress() {
  add_rules_shell "MODULE.bazel"
  mkdir pkg
  cat > pkg/BUILD <<EOF
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
sh_binary(name = "bin", srcs = ["bin.sh"])
EOF

  bazel --batch build --experimental_ui_debug_all_events --nobuild --noshow_loading_progress \
      //pkg:bin >& "$TEST_log" || fail "build failed"
  expect_not_log "Loading package"
  bazel --batch build --experimental_ui_debug_all_events --nobuild --show_loading_progress \
      //pkg:bin >& "$TEST_log" || fail "build failed"
  expect_log "Loading package: pkg"
}

function test_missing_target_yields_exit_code_1() {
  add_rules_shell "MODULE.bazel"
  mkdir foo
  cat > foo/BUILD <<EOF
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
sh_binary(name = "bin", srcs = ["bin.sh"])
EOF
  touch foo/bin.sh
  chmod +x foo/bin.sh

  local exit_code=0
  bazel build -k -- //foo:bin -//foo:does_not_exist >& "$TEST_log" || exit_code=$?
  expect_log 'does_not_exist.*not declared'

  assert_equals 1 $exit_code
}

function test_execution_progress() {
  mkdir x
  cat > x/BUILD <<'EOF'
genrule(
    name = 'foo',
    outs = ['foo.out'],
    cmd = 'touch $@',
    message = 'this is my funky genrule',
)
EOF

  bazel --batch build --experimental_ui_debug_all_events --curses=no --color=no //x:foo \
      >& "$TEST_log" || fail "build failed"

  expect_log "\[[0-9,]\{1,\} / [0-9,]\{1,\}\] this is my funky genrule"
  rm -rf x
}

function test_symlink_loop_detected() {
  if is_windows; then
    return 0
  fi
  mkdir loop_test
  touch loop_test/BUILD
  ln -s ../loop_test loop_test

  bazel query -k '//loop_test/...:*' &> "$TEST_log" && fail "build failed"
  expect_log '//loop_test:BUILD'
  expect_not_log '//loop_test/loop_test:BUILD'
}

# Regression test for b/12818115.
function test_pass_followed_by_fail_to_build() {
  add_rules_shell "MODULE.bazel"
  mkdir -p testing || die "Could not create test directory."
  FIFO=$PWD/testing/fifo
  mkfifo "$FIFO"

  echo "echo 'anything' > $FIFO; exit 0" > testing/fast.sh
  touch testing/slow.sh
  chmod +x testing/{fast,slow}.sh

  cat > testing/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name = 'fast', srcs = ['fast.sh'], local = 1)
sh_test(name = 'slow', srcs = ['slow.sh'], data = [':broken.out'])
genrule(name = 'broken', outs = ['broken.out'], local = 1,
EOF
  # The 'broken' genrule waits for the first test to run, then an extra second.
  echo "cmd ='cat $FIFO > /dev/null; sleep 1; exit 1')" >> testing/BUILD

  bazel test //testing:all -k >& "$TEST_log" && fail "Expected failure"

  expect_log "Executed 1 out of 2 tests: 1 test passes and 1 fails to build."
}

function test_backslashes() {
  mkdir -p back || die "Cannot create dir"
  touch back/BUILD
  touch back/\\
  bazel build --experimental_ui_debug_all_events back/... >& "$TEST_log" || fail "Expected success"
  expect_log "Building complete."
}

# Test that we show a proper error in the case of a visibility error instead of throwing an
# java.lang.IllegalStateException.
function test_invalid_visibility() {
  add_rules_cc "MODULE.bazel"
  mkdir -p visibility || die "Cannot create visibility directory"
  cat > visibility/BUILD <<EOF
load("@rules_cc//cc:cc_library.bzl", "cc_library")
cc_library(
    name = "foo",
    visibility = [":bar"],
)

cc_library(name = "bar")
EOF
  bazel build visibility:all &> "$TEST_log" && fail "Expected failure"
  exitcode="$?"
  # Build should fail with exit code 1 (BUILD_FAILURE)
  assert_equals 1 "$exitcode"
  expect_log "Invalid visibility label '//visibility:bar'; did you mean //visibility:public or //visibility:private?"
}

function test_mv_BUILD_file_with_same_mtime() {
  mkdir -p stampy
  cat > stampy/BUILD <<EOF
genrule(name = 'stampy',
        srcs = [], outs = ['stampy.out'],
        cmd = 'echo STAMPY > \$(OUTS)')
EOF

  cat > stampy/OTHER_BUILD <<EOF
genrule(name = 'stampy',
        srcs = [], outs = ['stampy.out'],
        cmd = 'echo OTHER_STAMPY > \$(OUTS)')
EOF

  SAME_TIME="198212160200"
  touch -t "$SAME_TIME" stampy/{BUILD,OTHER_BUILD}

  bazel --nobatch build //stampy || fail "Expected success"
  assert_equals "STAMPY" "$(cat "${PRODUCT_NAME}-genfiles/stampy/stampy.out")"

  mv stampy/{OTHER_BUILD,BUILD}
  bazel --nobatch build //stampy || fail "Expected success"
  assert_equals "OTHER_STAMPY" "$(cat "${PRODUCT_NAME}-genfiles/stampy/stampy.out")"
}

function test_mv_files_with_same_mtime() {
  mkdir -p stampy
  cat > stampy/BUILD <<EOF
genrule(name = 'stampy',
        srcs = ['f1'], outs = ['stampy.out'],
        cmd = 'cat \$(location f1) > \$(OUTS)')
EOF

  SAME_TIME="198212160200"
  echo 'STAMPY' > stampy/f1
  echo 'donovan_mcnabb' > stampy/f2
  touch -t "$SAME_TIME" stampy/{f1,f2}

  bazel --nobatch build //stampy || fail "Expected success"
  assert_equals "STAMPY" "$(cat "${PRODUCT_NAME}-genfiles/stampy/stampy.out")"

  mv stampy/{f2,f1}
  bazel --nobatch build //stampy || fail "Expected success"
  assert_equals "donovan_mcnabb" "$(cat "${PRODUCT_NAME}-genfiles/stampy/stampy.out")"
}

# Helper function for test below.  Takes sequences like [[[][[]]]]
# (with equal numbers of opening vs. closing brackets), representing a sequence
# of begin genrule/end genrule events.  Returns the maximum number of genrules
# that were running simultaneously.  For the example earlier, that number is 4.
function max_number_open_intervals() {
  local count=0
  local max_count=0
  for ((i=0; i < ${#1}; ++i))
  do
    local char=${1:i:1}
    if [[ "$char" == "[" ]]
    then
      count=$((count + 1))
    elif [[ "$char" == "]" ]]
    then
      count=$((count - 1))
    else
      fail "unexpected character \"$char\" in sequence :$1"
    fi
    if [[ $count -lt 0 ]]
    then
      fail "invalid sequence: too many decrements in sequence: $1"
    fi
    if [[ $count -gt $max_count ]]
    then
      max_count=$count
    fi
  done
  [[ $count -eq 0 ]] || fail "invalid sequence; too many increments in sequence: $1"
  echo "$max_count"
}

# This test was written in service of b/12801023.
function test_limited_number_of_simultaneous_actions() {
  # Tests for max_number_open_intervals.
  [[ 0 == $(max_number_open_intervals "") ]] || fail
  [[ 1 == $(max_number_open_intervals "[]") ]] || fail
  [[ 7 == $(max_number_open_intervals "[[[[]][[[[[][]]]]]]]") ]] || fail
  [[ 4 == $(max_number_open_intervals "[][][][[[[]]][][][[[]]]]") ]] || fail

  local num_genrules=100

  mkdir -p scratch
  cat > scratch/BUILD <<EOF
[genrule(
  name = "rule_%s" % i,
  srcs=[],
  # Single-byte writes in append mode are atomic, so no external lock utility
  # (such as Linux-only flock) is needed.
  cmd="shared_file=\$(@D)/shared; " +
    "printf '[' >> \$\$shared_file; " +
    "sleep 1; " +
    "printf ']' >> \$\$shared_file; " +
    "touch \$@;",
  outs=["rule_%s.out" % i],
  local=1,
) for i in [
EOF
  echo -n {1..100}, >> scratch/BUILD
  echo "]]" >> scratch/BUILD

  # Manually set the resources available on this machine. Some machines have
  # 30 cores or more these days.
  bazel build scratch:all --local_resources=cpu=29 || fail "Expected success"

  local sequence
  sequence=$(cat "${PRODUCT_NAME}-genfiles/scratch/shared")
  echo "start/stop sequence: $sequence"
  if [[ ${#sequence} -ne $(($num_genrules * 2)) ]]
  then
    fail "Sequence \"$sequence\"is too short, should be " $(($num_genrules * 2)) " bytes long".
  fi
  local max_open
  max_open=$(max_number_open_intervals "$sequence")
  [[ $max_open -lt 30 ]] || fail "Too many genrules ($max_open) ran simultaneously."
}

run_suite "Skyframe integration tests"
