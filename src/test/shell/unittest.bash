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
#
# Common utility file for Bazel shell tests
#
# unittest.bash: a unit test framework in Bash.
#
# A typical test suite looks like so:
#
#   ------------------------------------------------------------------------
#   #!/bin/bash
#
#   source path/to/unittest.bash || exit 1
#
#   # Test that foo works.
#   function test_foo() {
#     foo >$TEST_log || fail "foo failed";
#     expect_log "blah" "Expected to see 'blah' in output of 'foo'."
#   }
#
#   # Test that bar works.
#   function test_bar() {
#     bar 2>$TEST_log || fail "bar failed";
#     expect_not_log "ERROR" "Unexpected error from 'bar'."
#     ...
#     assert_equals $x $y
#   }
#
#   run_suite "Test suite for blah"
#   ------------------------------------------------------------------------
#
# Each test function is considered to pass iff fail() is not called
# while it is active.  fail() may be called directly, or indirectly
# via other assertions such as expect_log().  run_suite must be called
# at the very end.
#
# A test function may redefine functions "set_up" and/or "tear_down";
# these functions are executed before and after each test function,
# respectively.  Similarly, "cleanup" and "timeout" may be redefined,
# and these function are called upon exit (of any kind) or a timeout.
#
# The user can pass --test_arg to blaze test to select specific tests
# to run. Specifying --test_arg multiple times allows to select several
# tests to be run in the given order. Additionally the user may define
# TESTS=(test_foo test_bar ...) to specify a subset of test functions to
# execute, for example, a working set during debugging. By default, all
# functions called test_* will be executed.
#
# This file provides utilities for assertions over the output of a
# command.  The output of the command under test is directed to the
# file $TEST_log, and then the expect_log* assertions can be used to
# test for the presence of certain regular expressions in that file.
#
# The test framework is responsible for restoring the original working
# directory before each test.
#
# The order in which test functions are run is not defined, so it is
# important that tests clean up after themselves.
#
# Each test will be run in a new subshell.
#
# Functions named __* are not intended for use by clients.
#
# This framework implements the "test sharding protocol".
#

[ -n "$BASH_VERSION" ] ||
  { echo "unittest.bash only works with bash!" >&2; exit 1; }

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#### Configuration variables (may be overridden by testenv.sh or the suite):

# This function may be called by testenv.sh or a test suite to enable errexit
# in a way that enables us to print pretty stack traces when something fails.
function enable_errexit() {
  set -o errtrace
  set -eu
  trap __test_terminated_err ERR
}

function disable_errexit() {
  set +o errtrace
  set +eu
  trap - ERR
}

source ${DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

#### Global variables:

TEST_name=""                    # The name of the current test.

TEST_log=$TEST_TMPDIR/log       # The log file over which the
                                # expect_log* assertions work.  Must
                                # be absolute to be robust against
                                # tests invoking 'cd'!

TEST_passed="true"              # The result of the current test;
                                # failed assertions cause this to
                                # become false.

# These variables may be overridden by the test suite:

TESTS=()                        # A subset or "working set" of test
                                # functions that should be run.  By
                                # default, all tests called test_* are
                                # run.
if [ $# -gt 0 ]; then
  # Legacy behavior is to ignore missing regexp, but with errexit
  # the following line fails without || true.
  # TODO(dmarting): maybe we should revisit the way of selecting
  # test with that framework (use Bazel's environment variable instead).
  TESTS=($(for i in $@; do echo $i; done | grep ^test_ || true))
  if (( ${#TESTS[@]} == 0 )); then
    echo "WARNING: Arguments do not specifies tests!" >&2
  fi
fi

TEST_verbose="true"             # Whether or not to be verbose.  A
                                # command; "true" or "false" are
                                # acceptable.  The default is: true.

TEST_script="$(pwd)/$0"         # Full path to test script

#### Internal functions

function __show_log() {
    echo "-- Test log: -----------------------------------------------------------"
    [[ -e $TEST_log ]] && cat $TEST_log || echo "(Log file did not exist.)"
    echo "------------------------------------------------------------------------"
}

# Usage: __pad <title> <pad-char>
# Print $title padded to 80 columns with $pad_char.
function __pad() {
    local title=$1
    local pad=$2
    {
        echo -n "$pad$pad $title "
        printf "%80s" " " | tr ' ' "$pad"
    } | head -c 80
    echo
}

#### Exported functions

# Usage: init_test ...
# Deprecated.  Has no effect.
function init_test() {
    :
}


# Usage: set_up
# Called before every test function.  May be redefined by the test suite.
function set_up() {
    :
}

# Usage: tear_down
# Called after every test function.  May be redefined by the test suite.
function tear_down() {
    :
}

# Usage: cleanup
# Called upon eventual exit of the test suite.  May be redefined by
# the test suite.
function cleanup() {
    :
}

# Usage: timeout
# Called upon early exit from a test due to timeout.
function timeout() {
    :
}

# Usage: fail <message> [<message> ...]
# Print failure message with context information, and mark the test as
# a failure.  The context includes a stacktrace including the longest sequence
# of calls outside this module.  (We exclude the top and bottom portions of
# the stack because they just add noise.)  Also prints the contents of
# $TEST_log.
function fail() {
    __show_log >&2
    echo "$TEST_name FAILED:" "$@" "." >&2
    echo "$@" >$TEST_TMPDIR/__fail
    TEST_passed="false"
    __show_stack
    # Cleanup as we are leaving the subshell now
    tear_down
    exit 1
}

# Usage: warn <message>
# Print a test warning with context information.
# The context includes a stacktrace including the longest sequence
# of calls outside this module.  (We exclude the top and bottom portions of
# the stack because they just add noise.)
function warn() {
    __show_log >&2
    echo "$TEST_name WARNING: $1." >&2
    __show_stack

    if [ -n "${TEST_WARNINGS_OUTPUT_FILE:-}" ]; then
      echo "$TEST_name WARNING: $1." >> "$TEST_WARNINGS_OUTPUT_FILE"
    fi
}

# Usage: show_stack
# Prints the portion of the stack that does not belong to this module,
# i.e. the user's code that called a failing assertion.  Stack may not
# be available if Bash is reading commands from stdin; an error is
# printed in that case.
__show_stack() {
    local i=0
    local trace_found=0

    # Skip over active calls within this module:
    while (( i < ${#FUNCNAME[@]} )) && [[ ${BASH_SOURCE[i]:-} == ${BASH_SOURCE[0]} ]]; do
       (( ++i ))
    done

    # Show all calls until the next one within this module (typically run_suite):
    while (( i < ${#FUNCNAME[@]} )) && [[ ${BASH_SOURCE[i]:-} != ${BASH_SOURCE[0]} ]]; do
        # Read online docs for BASH_LINENO to understand the strange offset.
        # Undefined can occur in the BASH_SOURCE stack apparently when one exits from a subshell
        echo "${BASH_SOURCE[i]:-"Unknown"}:${BASH_LINENO[i - 1]:-"Unknown"}: in call to ${FUNCNAME[i]:-"Unknown"}" >&2
        (( ++i ))
        trace_found=1
    done

    [ $trace_found = 1 ] || echo "[Stack trace not available]" >&2
}

# Usage: expect_log <regexp> [error-message]
# Asserts that $TEST_log matches regexp.  Prints the contents of
# $TEST_log and the specified (optional) error message otherwise, and
# returns non-zero.
function expect_log() {
    local pattern=$1
    local message=${2:-Expected regexp "$pattern" not found}
    grep -sq -- "$pattern" $TEST_log && return 0

    fail "$message"
    return 1
}

# Usage: expect_log_warn <regexp> [error-message]
# Warns if $TEST_log does not match regexp.  Prints the contents of
# $TEST_log and the specified (optional) error message on mismatch.
function expect_log_warn() {
    local pattern=$1
    local message=${2:-Expected regexp "$pattern" not found}
    grep -sq -- "$pattern" $TEST_log && return 0

    warn "$message"
    return 1
}

# Usage: expect_log_once <regexp> [error-message]
# Asserts that $TEST_log contains one line matching <regexp>.
# Prints the contents of $TEST_log and the specified (optional)
# error message otherwise, and returns non-zero.
function expect_log_once() {
    local pattern=$1
    local message=${2:-Expected regexp "$pattern" not found exactly once}
    expect_log_n "$pattern" 1 "$message"
}

# Usage: expect_log_n <regexp> <count> [error-message]
# Asserts that $TEST_log contains <count> lines matching <regexp>.
# Prints the contents of $TEST_log and the specified (optional)
# error message otherwise, and returns non-zero.
function expect_log_n() {
    local pattern=$1
    local expectednum=${2:-1}
    local message=${3:-Expected regexp "$pattern" not found exactly $expectednum times}
    local count=$(grep -sc -- "$pattern" $TEST_log)
    [ $count = $expectednum ] && return 0
    fail "$message"
    return 1
}

# Usage: expect_not_log <regexp> [error-message]
# Asserts that $TEST_log does not match regexp.  Prints the contents
# of $TEST_log and the specified (optional) error message otherwise, and
# returns non-zero.
function expect_not_log() {
    local pattern=$1
    local message=${2:-Unexpected regexp "$pattern" found}
    grep -sq -- "$pattern" $TEST_log || return 0

    fail "$message"
    return 1
}

# Usage: expect_log_with_timeout <regexp> <timeout> [error-message]
# Waits for the given regexp in the $TEST_log for up to timeout seconds.
# Prints the contents of $TEST_log and the specified (optional)
# error message otherwise, and returns non-zero.
function expect_log_with_timeout() {
    local pattern=$1
    local timeout=$2
    local message=${3:-Regexp "$pattern" not found in "$timeout" seconds}
    local count=0
    while [ $count -lt $timeout ]; do
      grep -sq -- "$pattern" $TEST_log && return 0
      let count=count+1
      sleep 1
    done

    grep -sq -- "$pattern" $TEST_log && return 0
    fail "$message"
    return 1
}

# Usage: expect_cmd_with_timeout <expected> <cmd> [timeout]
# Repeats the command once a second for up to timeout seconds (10s by default),
# until the output matches the expected value. Fails and returns 1 if
# the command does not return the expected value in the end.
function expect_cmd_with_timeout() {
    local expected="$1"
    local cmd="$2"
    local timeout=${3:-10}
    local count=0
    while [ $count -lt $timeout ]; do
      local actual="$($cmd)"
      [ "$expected" = "$actual" ] && return 0
      let count=count+1
      sleep 1
    done

    [ "$expected" = "$actual" ] && return 0
    fail "Expected '$expected' within ${timeout}s, was '$actual'"
    return 1
}

# Usage: assert_equals <expected> <actual>
# Asserts [ expected = actual ].
function assert_equals() {
    local expected=$1 actual=$2
    [ "$expected" = "$actual" ] && return 0

    fail "Expected '$expected', was '$actual'"
    return 1
}

# Usage: assert_not_equals <unexpected> <actual>
# Asserts [ unexpected != actual ].
function assert_not_equals() {
    local unexpected=$1 actual=$2
    [ "$unexpected" != "$actual" ] && return 0;

    fail "Expected not '$unexpected', was '$actual'"
    return 1
}

# Usage: assert_contains <regexp> <file> [error-message]
# Asserts that file matches regexp.  Prints the contents of
# file and the specified (optional) error message otherwise, and
# returns non-zero.
function assert_contains() {
    local pattern=$1
    local file=$2
    local message=${3:-Expected regexp "$pattern" not found in "$file"}
    grep -sq -- "$pattern" "$file" && return 0

    cat "$file" >&2
    fail "$message"
    return 1
}

# Usage: assert_not_contains <regexp> <file> [error-message]
# Asserts that file does not match regexp.  Prints the contents of
# file and the specified (optional) error message otherwise, and
# returns non-zero.
function assert_not_contains() {
    local pattern=$1
    local file=$2
    local message=${3:-Expected regexp "$pattern" found in "$file"}
    grep -sq -- "$pattern" "$file" || return 0

    cat "$file" >&2
    fail "$message"
    return 1
}

# Updates the global variables TESTS if
# sharding is enabled, i.e. ($TEST_TOTAL_SHARDS > 0).
function __update_shards() {
    [ -z "${TEST_TOTAL_SHARDS-}" ] && return 0

    [ "$TEST_TOTAL_SHARDS" -gt 0 ] ||
      { echo "Invalid total shards $TEST_TOTAL_SHARDS" >&2; exit 1; }

    [ "$TEST_SHARD_INDEX" -lt 0 -o "$TEST_SHARD_INDEX" -ge  "$TEST_TOTAL_SHARDS" ] &&
      { echo "Invalid shard $shard_index" >&2; exit 1; }

    TESTS=$(for test in "${TESTS[@]}"; do echo "$test"; done |
      awk "NR % $TEST_TOTAL_SHARDS == $TEST_SHARD_INDEX")

    [ -z "${TEST_SHARD_STATUS_FILE-}" ] || touch "$TEST_SHARD_STATUS_FILE"
}

# Usage: __test_terminated <signal-number>
# Handler that is called when the test terminated unexpectedly
function __test_terminated() {
    __show_log >&2
    echo "$TEST_name FAILED: terminated by signal $1." >&2
    TEST_passed="false"
    __show_stack
    timeout
    exit 1
}

# Usage: __test_terminated_err
# Handler that is called when the test terminated unexpectedly due to "errexit".
function __test_terminated_err() {
    # When a subshell exits due to signal ERR, its parent shell also exits,
    # thus the signal handler is called recursively and we print out the
    # error message and stack trace multiple times. We're only interested
    # in the first one though, as it contains the most information, so ignore
    # all following.
    if [[ -f $TEST_TMPDIR/__err_handled ]]; then
      exit 1
    fi
    __show_log >&2
    if [[ ! -z "$TEST_name" ]]; then
      echo -n "$TEST_name "
    fi
    echo "FAILED: terminated because this command returned a non-zero status:" >&2
    touch $TEST_TMPDIR/__err_handled
    TEST_passed="false"
    __show_stack
    # If $TEST_name is still empty, the test suite failed before we even started
    # to run tests, so we shouldn't call tear_down.
    if [[ ! -z "$TEST_name" ]]; then
      tear_down
    fi
    exit 1
}

# Usage: __trap_with_arg <handler> <signals ...>
# Helper to install a trap handler for several signals preserving the signal
# number, so that the signal number is available to the trap handler.
function __trap_with_arg() {
    func="$1" ; shift
    for sig ; do
        trap "$func $sig" "$sig"
    done
}

# Usage: <node> <block>
# Adds the block to the given node in the report file. Quotes in the in
# arguments need to be escaped.
function __log_to_test_report() {
    local node="$1"
    local block="$2"
    if [[ ! -e "$XML_OUTPUT_FILE" ]]; then
        local xml_header='<?xml version="1.0" encoding="UTF-8"?>'
        echo "$xml_header<testsuites></testsuites>" > $XML_OUTPUT_FILE
    fi

    # replace match on node with block and match
    # replacement expression only needs escaping for quotes
    perl -e "\
\$input = @ARGV[0]; \
\$/=undef; \
open FILE, '+<$XML_OUTPUT_FILE'; \
\$content = <FILE>; \
if (\$content =~ /($node.*)\$/) { \
  seek FILE, 0, 0; \
  print FILE \$\` . \$input . \$1; \
}; \
close FILE" "$block"
}

# Usage: <total> <passed>
# Adds the test summaries to the xml nodes.
function __finish_test_report() {
    local total=$1
    local passed=$2
    local failed=$((total - passed))

    cat $XML_OUTPUT_FILE >$XML_OUTPUT_FILE.bak | \
    sed \
      "s/<testsuites>/<testsuites tests=\"$total\" failures=\"0\" errors=\"$failed\">/" | \
    sed \
      "s/<testsuite>/<testsuite tests=\"$total\" failures=\"0\" errors=\"$failed\">/"

    rm -f $XML_OUTPUT_FILE
    mv $XML_OUTPUT_FILE.bak $XML_OUTPUT_FILE
}

# Multi-platform timestamp function
if [ "$(uname -s | tr 'A-Z' 'a-z')" = "linux" ]; then
    function timestamp() {
      echo $(($(date +%s%N)/1000000))
    }
else
    function timestamp() {
      # OS X and FreeBSD do not have %N so python is the best we can do
      python -c 'import time; print int(round(time.time() * 1000))'
    }
fi

function get_run_time() {
  local ts_start=$1
  local ts_end=$2
  run_time_ms=$((${ts_end}-${ts_start}))
  echo $(($run_time_ms/1000)).${run_time_ms: -3}
}

# Usage: run_tests <suite-comment>
# Must be called from the end of the user's test suite.
# Calls exit with zero on success, non-zero otherwise.
function run_suite() {
    echo >&2
    echo "$1" >&2
    echo >&2

    __log_to_test_report "<\/testsuites>" "<testsuite></testsuite>"

    local total=0
    local passed=0

    atexit "cleanup"

    # If the user didn't specify an explicit list of tests (e.g. a
    # working set), use them all.
    if [ ${#TESTS[@]} = 0 ]; then
      TESTS=$(declare -F | awk '{print $3}' | grep ^test_)
    elif [ -n "${TEST_WARNINGS_OUTPUT_FILE:-}" ]; then
      if grep -q "TESTS=" "$TEST_script" ; then
        echo "TESTS variable overridden in Blaze sh_test. Please remove before submitting" \
          >> "$TEST_WARNINGS_OUTPUT_FILE"
      fi
    fi

    __update_shards

    for TEST_name in ${TESTS[@]}; do
      >$TEST_log # Reset the log.
      TEST_passed="true"

      total=$(($total + 1))
      if [[ "$TEST_verbose" == "true" ]]; then
          __pad $TEST_name '*' >&2
      fi

      local run_time="0.0"
      rm -f $TEST_TMPDIR/{__ts_start,__ts_end}

      if [ "$(type -t $TEST_name)" = function ]; then
        # Save exit handlers eventually set.
        local SAVED_ATEXIT="$ATEXIT";
        ATEXIT=

        # Run test in a subshell.
        rm -f $TEST_TMPDIR/__err_handled
        __trap_with_arg __test_terminated INT KILL PIPE TERM ABRT FPE ILL QUIT SEGV
        (
          timestamp >$TEST_TMPDIR/__ts_start
          set_up
          eval $TEST_name
          tear_down
          timestamp >$TEST_TMPDIR/__ts_end
          test $TEST_passed == "true"
        ) 2>&1 | tee $TEST_TMPDIR/__log
        # Note that tee will prevent the control flow continuing if the test
        # spawned any processes which are still running and have not closed
        # their stdout.

        test_subshell_status=${PIPESTATUS[0]}
        if [ "$test_subshell_status" != 0 ]; then
          TEST_passed="false"
          # Ensure that an end time is recorded in case the test subshell
          # terminated prematurely.
          [ -f $TEST_TMPDIR/__ts_end ] || timestamp >$TEST_TMPDIR/__ts_end
        fi

        # Calculate run time for the testcase.
        local ts_start=$(cat $TEST_TMPDIR/__ts_start)
        local ts_end=$(cat $TEST_TMPDIR/__ts_end)
        run_time=$(get_run_time $ts_start $ts_end)

        # Eventually restore exit handlers.
        if [ -n "$SAVED_ATEXIT" ]; then
          ATEXIT="$SAVED_ATEXIT"
          trap "$ATEXIT" EXIT
        fi
      else # Bad test explicitly specified in $TESTS.
        fail "Not a function: '$TEST_name'"
      fi

      local testcase_tag=""

      if [[ "$TEST_passed" == "true" ]]; then
        if [[ "$TEST_verbose" == "true" ]]; then
          echo "PASSED: $TEST_name" >&2
        fi
        passed=$(($passed + 1))
        testcase_tag="<testcase name=\"$TEST_name\" status=\"run\" time=\"$run_time\" classname=\"\"></testcase>"
      else
        echo "FAILED: $TEST_name" >&2
        # end marker in CDATA cannot be escaped, we need to split the CDATA sections
        log=$(cat $TEST_TMPDIR/__log | sed 's/]]>/]]>]]&gt;<![CDATA[/g')
        fail_msg=$(cat $TEST_TMPDIR/__fail 2> /dev/null || echo "No failure message")
        testcase_tag="<testcase name=\"$TEST_name\" status=\"run\" time=\"$run_time\" classname=\"\"><error message=\"$fail_msg\"><![CDATA[$log]]></error></testcase>"
      fi

      if [[ "$TEST_verbose" == "true" ]]; then
          echo >&2
      fi
      __log_to_test_report "<\/testsuite>" "$testcase_tag"
    done

    __finish_test_report $total $passed
    __pad "$passed / $total tests passed." '*' >&2
    [ $total = $passed ] || {
      __pad "There were errors." '*'
      exit 1
    } >&2

    exit 0
}
