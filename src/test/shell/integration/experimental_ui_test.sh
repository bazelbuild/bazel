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

# --- begin runfiles.bash initialization ---
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

#### SETUP #############################################################

add_to_bazelrc "build --genrule_strategy=local"
add_to_bazelrc "test --test_strategy=standalone"

function set_up() {
  if [[ -d pkg ]]; then
    # All tests share these scratch packages. No need to recreate them if they
    # already exist.
    return
  fi

  mkdir -p pkg
  touch remote_file
  cat > pkg/true.sh <<EOF
#!/bin/sh
exit 0
EOF
  chmod 755 pkg/true.sh
  cat > pkg/slow.sh <<EOF
#!/bin/sh
sleep 3
exit 0
EOF
  chmod 755 pkg/slow.sh
  cat > pkg/false.sh <<EOF
#!/bin/sh
exit 1
EOF
  chmod 755 pkg/false.sh
  cat > pkg/output.sh <<EOF
#!/bin/sh
`which echo` -n foo
sleep 1
`which echo` -n bar
exit 0
EOF
  chmod 755 pkg/output.sh
  cat > pkg/do_output.sh <<EOF
#!/bin/sh
echo Beginning \$1
for _ in \`seq 1 10240\`
do echo '1234567890'
done
echo Ending \$1
EOF
  chmod 755 pkg/do_output.sh
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
sh_test(
  name = "output",
  srcs = ["output.sh"],
)
genrule(
  name = "gentext",
  outs = ["gentext.txt"],
  cmd = "echo here be dragons > \"\$@\""
)
genrule(
  name = "withOutputA",
  outs = ["a"],
  tools = [":do_output.sh"],
  cmd = "\$(location :do_output.sh) A && touch \$@",
)
genrule(
  name = "withOutputB",
  outs = ["b"],
  tools = [":do_output.sh"],
  cmd = "\$(location :do_output.sh) B && touch \$@",
)
sh_library(
  name = "outputlib",
  data = [":withOutputA", ":withOutputB"],
)
sh_test(
  name = "truedependingonoutput",
  srcs = ["true.sh"],
  deps = [":outputlib"],
)
EOF
  mkdir -p error
  cat > error/BUILD <<'EOF'
genrule(
  name = "failwitherror",
  outs = ["fail.txt"],
  cmd = "echo Here is the error message; exit 1",
)
EOF
  mkdir -p pkg/errorAfterWarning
  cat > pkg/errorAfterWarning/BUILD <<'EOF'
RANGE = range(500)

[ genrule(
    name = "true%s_c" % i,
    outs = ["true%s.c" % i],
    cmd = "echo Build Warning...; echo 'int main(int argc, char **argv) { return 0; }' > $@",
) for i in RANGE]

[ cc_binary(
    name = "true_%s" % i,
    srcs = ["true%s.c" % i],
) for i in RANGE]

genrule(
  name = "failing",
  outs = ["failing.txt"],
  srcs = ["true_%s" % i for i in RANGE],
  cmd = "echo This is the error message; false",
)
EOF
  chmod -w pkg/* # prevent accidental editing
  # keep directories writable though, so that test clean up can work
  chmod 755 error
  chmod 755 pkg/errorAfterWarning
  mkdir -p pkg/debugMessages
  cat > pkg/debugMessages/rule.bzl <<'EOF'
def _impl(ctx):
  print("static debug message")
  ctx.actions.write(ctx.outputs.out, "Hello World")

withdebug = rule(
  implementation = _impl,
  attrs = {},
  outputs = {"out" : "%{name}.txt"},
)
EOF
  cat > pkg/debugMessages/BUILD <<'EOF'
load("//pkg/debugMessages:rule.bzl", "withdebug")

[ withdebug(name = "target%d" % (i,)) for i in range(50) ]
EOF
  mkdir -p bzl
  touch bzl/BUILD
  cat > bzl/bzl.bzl <<'EOF'
x = invalidsyntax
EOF
  mkdir -p pkgloadingerror
  cat > pkgloadingerror/BUILD <<'EOF'
load("//bzl:bzl.bzl", "x")
EOF
  mkdir -p fancyOutput
  cat > fancyOutput/BUILD <<'EOF'
genrule(
  name = "withFancyOutput",
  outs = ["out.txt"],
  cmd = "echo $$'\\xF0\\x9F\\x8D\\x83'; echo Hello World > $@",
)
EOF
}

#### TESTS #############################################################

function test_basic_progress() {
  bazel test --experimental_ui --curses=yes --color=yes pkg:true 2>$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # some progress indicator is shown
  expect_log '\[[0-9,]* / [0-9,]*\]'
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
  # As precisely one target is specified, it should be reported during
  # analysis phase.
  expect_log 'Analy.*pkg:true'
}

function test_noshow_progress() {
  bazel test --experimental_ui --noshow_progress --curses=yes --color=yes \
    pkg:true 2>$TEST_log || fail "${PRODUCT_NAME} test failed"
  # Info messages should still go through
  expect_log 'Elapsed time'
  # no progress indicator is shown
  expect_not_log '\[[0-9,]* / [0-9,]*\]'
}

function test_basic_progress_no_curses() {
  bazel test --experimental_ui --curses=no --color=yes pkg:true 2>$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
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
    pkg:true 2>$TEST_log || fail "${PRODUCT_NAME} test failed"
  # expect a long-ish status line
  expect_log '\[[0-9,]* / [0-9,]*\]......'
}

function test_pass() {
  bazel test --experimental_ui --curses=yes --color=yes pkg:true >$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # PASS is written in green on the same line as the test target
  expect_log 'pkg:true.*'$'\x1b\[32m''.*PASS'
}

function test_fail() {
  bazel test --experimental_ui --curses=yes --color=yes pkg:false >$TEST_log \
    && fail "expected failure"
  # FAIL is written in red bold on the same line as the test target
  expect_log 'pkg:false.*'$'\x1b\[31m\x1b\[1m''.*FAIL'
}

function test_timestamp() {
  bazel test --experimental_ui --show_timestamps pkg:true 2>$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # expect something that looks like HH:mm:ss
  expect_log '[0-2][0-9]:[0-5][0-9]:[0-6][0-9]'
}

function test_info_spacing() {
  # Verify that the output of "bazel info" is suitable for backtick escapes,
  # in particular free carriage-return characters.
  BAZEL_INFO_OUTPUT=XXX`bazel info --experimental_ui workspace`XXX
  echo "$BAZEL_INFO_OUTPUT" | grep -q 'XXX[^'$'\r'']*XXX' \
    || fail "${PRODUCT_NAME} info output spaced as $BAZEL_INFO_OUTPUT"
}

function test_query_spacing() {
  # Verify that the output of "bazel query" is suitable for consumption by
  # other tools, i.e., contains only result lines, separated only by newlines.
  BAZEL_QUERY_OUTPUT=`bazel query --experimental_ui 'deps(//pkg:true)'`
  echo "$BAZEL_QUERY_OUTPUT" | grep -q -v '^[@/]' \
   && fail "bazel query output is >$BAZEL_QUERY_OUTPUT<" || true
  if ! $is_windows; then
    echo "$BAZEL_QUERY_OUTPUT" | grep -q $'\r' \
     && fail "bazel query output is >$BAZEL_QUERY_OUTPUT<" || true
  fi
}

function test_query_progress() {
  # Verify that some form of progress is reported during bazel query
  bazel query --experimental_ui 'deps(//pkg:true)' 2> "${TEST_log}"
  expect_log 'Loading:.*packages loaded'
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

function test_version_nobuild {
  bazel version --experimental_ui --curses=yes 2>$TEST_log \
   || fail "bazel version failed"
  expect_not_log "action"
  expect_not_log "Building"
}

function test_version_nobuild_announce_rc {
  bazel version --experimental_ui --curses=yes --announce_rc 2>$TEST_log \
   || fail "bazel version failed"
  expect_not_log "action"
  expect_not_log "Building"
}

function test_subcommand {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel build --experimental_ui -s pkg:gentext 2>$TEST_log \
    || fail "bazel build failed"
  expect_log "here be dragons"
}

function test_subcommand_notdefault {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel build --experimental_ui pkg:gentext 2>$TEST_log \
    || fail "bazel build failed"
  expect_not_log "dragons"
}

function test_loading_progress {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel test --experimental_ui pkg:true 2>$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # some progress indicator is shown during loading
  expect_log 'Loading.*[0-9,]* packages'
}

function test_failure_scrollback_buffer_curses {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel test --experimental_ui --curses=yes --color=yes \
    --nocache_test_results pkg:false pkg:slow 2>$TEST_log \
    && fail "expected failure"
  # Some line starts with FAIL in red bold.
  expect_log '^'$'\(.*\x1b\[K\)*\x1b\[31m\x1b\[1mFAIL:'
}

function test_terminal_title {
  bazel test --experimental_ui --curses=yes \
    --progress_in_terminal_title pkg:true \
    2>$TEST_log || fail "${PRODUCT_NAME} test failed"
  # The terminal title is changed
  expect_log $'\x1b\]0;.*\x07'
}

function test_failure_scrollback_buffer {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel test --experimental_ui --curses=no --color=yes \
    --nocache_test_results pkg:false pkg:slow 2>$TEST_log \
    && fail "expected failure"
  # Some line starts with FAIL in red bold.
  expect_log '^'$'\x1b\[31m\x1b\[1mFAIL:'
}

function test_streamed {
  bazel test --experimental_ui --curses=yes --color=yes \
    --nocache_test_results --test_output=streamed pkg:output >$TEST_log \
    || fail "expected success"
  expect_log 'foobar'
}

function test_stdout_bundled {
    # Verify that the error message is part of the error event
    bazel build --experimental_ui --experimental_ui_debug_all_events \
          error:failwitherror > "${TEST_log}" 2>&1 \
    && fail "expected failure" || :
    grep -A1 '^ERROR' "${TEST_log}" \
        | grep -q "with STDOUT: Here is the error message" \
        || fail "Error message not bundled"
}

function test_output_deduplicated {
    # Verify that we suscessfully deduplicate identical messages from actions
    bazel clean --expunge
    bazel version
    bazel build --experimental_ui --curses=yes --color=yes \
          --experimental_ui_deduplicate \
          pkg/errorAfterWarning:failing >"${TEST_log}" 2>&1 \
        && fail "expected failure" || :
    expect_log_once 'Build Warning'
    expect_log 'This is the error message'
    expect_log 'ERROR.*//pkg/errorAfterWarning:failing'
    expect_log 'deduplicated.*events'
}

function test_debug_deduplicated {
    # Verify that we suscessfully deduplicate identical debug statements
    bazel clean --expunge
    bazel version
    bazel build --experimental_ui --curses=yes --color=yes \
          --experimental_ui_deduplicate \
          pkg/debugMessages/... >"${TEST_log}" 2>&1 || fail "Expected success"
    expect_log_once 'static debug message'
    expect_log 'deduplicated.*events'
}

function test_output_limit {
    # Verify that output limting works
    bazel clean --expunge
    bazel version
    # The two actions produce about 100k of output each. As we set an output
    # limit to 50k total, we expect it to be truncated reasonably so that we
    # can see the end of the output of both actions, while still staying in
    # the limit.
    # However, that limit only applies to the output produces by the bazel
    # server; any startup message generated by the client is on top of that.
    # So we add another 1k output for what the client has to tell.
    bazel build --experimental_ui --curses=yes --color=yes \
          --experimental_ui_limit_console_output=51200 \
          pkg:withOutputA pkg:withOutputB >$TEST_log 2>&1 \
    || fail "expected success"
    expect_log 'Ending A'
    expect_log 'Ending B'
    output_length=`cat $TEST_log | wc -c`
    [ "${output_length}" -le 52224 ] \
        || fail "Output too large, is ${output_length}"
}

function test_status_despite_output_limit {
    # Verify that even if we limit the output very strictly, we
    # still find the test summary.
    bazel clean --expunge
    bazel version
    bazel test --experimental_ui --curses=yes --color=yes \
          --experimental_ui_limit_console_output=500 \
          pkg:truedependingonoutput >$TEST_log 2>&1 \
    || fail "expected success"
    expect_log "//pkg:truedependingonoutput.*PASSED"

    # Also sanity check that the limit was applied, again, allowing
    # 2k for any startup messages etc generated by the client.
    output_length=`cat $TEST_log | wc -c`
    [ "${output_length}" -le 2724 ] \
        || fail "Output too large, is ${output_length}"
}

function test_error_message_despite_output_limit {
    # Verify that, even if we limit the output very strictly, we
    # still the the final error message.
    bazel clean --expunge
    bazel version
    bazel build --experimental_ui --curses=yes --color=yes \
          --experimental_ui_limit_console_output=10240 \
          --noexperimental_ui_deduplicate \
          pkg/errorAfterWarning:failing >"${TEST_log}" 2>&1 \
        && fail "expected failure" || :
    expect_log 'This is the error message'
    expect_log 'ERROR.*//pkg/errorAfterWarning:failing'

    # Also sanity check that the limit was applied, again, allowing
    # 2k for any startup messages etc generated by the client.
    output_length=`cat $TEST_log | wc -c`
    [[ "${output_length}" -le 11264 ]] \
        || fail "Output too large, is ${output_length}"

    # Also expect a note that messages were dropped on the console
    expect_log "dropped.*console"
}

function run_test_attempt_to_print_relative_paths_failing_action() {
    local ui="$1"

    # On the BazelCI Windows environment, `pwd` returns a string that uses a
    # lowercase drive letter and unix-style path separators (e.g. '/c/') with
    # a lowercase drive letter. But internally in Bazel, Path#toString
    # unconditionally uses an uppercase drive letter (see
    # WindowsOsPathPolicy#normalize). I want these tests to check for exact
    # string contents (that's entire goal of the flag being tested), but I
    # don't want them to be brittle across different Windows enviromments, so
    # I've disabled them for now.
    # TODO(nharmata): Fix this.
    [[ "$is_windows" == "true" ]] && return 0

    bazel clean || fail "${PRODUCT_NAME} clean failed"

    bazel build \
        "$ui" \
        --attempt_to_print_relative_paths=false \
        error:failwitherror > "${TEST_log}" 2>&1 && fail "expected failure"
    expect_log "^ERROR: $(pwd)/error/BUILD:1:1: Executing genrule"

    bazel build \
        "$ui" \
        --attempt_to_print_relative_paths=true \
        error:failwitherror > "${TEST_log}" 2>&1 && fail "expected failure"
    expect_log "^ERROR: error/BUILD:1:1: Executing genrule"
    expect_not_log "$(pwd)/error/BUILD"
}

function test_experimental_ui_attempt_to_print_relative_paths_failing_action() {
  run_test_attempt_to_print_relative_paths_failing_action "--experimental_ui"
}

function test_noexperimental_ui_attempt_to_print_relative_paths_failing_action() {
  run_test_attempt_to_print_relative_paths_failing_action "--noexperimental_ui"
}

function run_test_attempt_to_print_relative_paths_pkg_error() {
    local ui="$1"

    # See the note in the test case above for why this is disabled on Windows.
    # TODO(nharmata): Fix this.
    [[ "$is_windows" == "true" ]] && return 0

    bazel clean || fail "${PRODUCT_NAME} clean failed"

    bazel build \
        "$ui" \
        --attempt_to_print_relative_paths=false \
        pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
    expect_log "^ERROR: $(pwd)/bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"

    bazel build \
        "$ui" \
        --attempt_to_print_relative_paths=true \
        pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
    expect_log "^ERROR: bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
    expect_not_log "$(pwd)/bzl/bzl.bzl"
}

function test_experimental_ui_attempt_to_print_relative_paths_pkg_error() {
  run_test_attempt_to_print_relative_paths_pkg_error "--experimental_ui"
}

function test_noexperimental_ui_attempt_to_print_relative_paths_pkg_error() {
  run_test_attempt_to_print_relative_paths_pkg_error "--noexperimental_ui"
}

function test_fancy_symbol_encoding() {
    bazel build //fancyOutput:withFancyOutput > "${TEST_log}" 2>&1 \
        || fail "expected success"
    expect_log $'\xF0\x9F\x8D\x83'
}

run_suite "Integration tests for ${PRODUCT_NAME}'s experimental UI"
