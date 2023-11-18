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
# An end-to-end test that Bazel's UI produces reasonable output.

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
  cmd = "\$(location :do_output.sh) A && touch \"\$@\"",
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

function create_pkg() {
  local -r pkg=$1
  mkdir -p $pkg
  cat > $pkg/true.sh <<EOF
#!/bin/sh
exit 0
EOF
  chmod 755 $pkg/true.sh
  cat > $pkg/BUILD <<EOF
sh_test(
  name = "true",
  srcs = ["true.sh"],
)
EOF
}

#### TESTS #############################################################

function test_basic_progress() {
  bazel test --curses=yes --color=yes pkg:true 2>$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # some progress indicator is shown
  expect_log '\[[0-9,]* / [0-9,]*\]'
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
  # As precisely one target is specified, it should be reported during
  # analysis phase.
  expect_log 'Analy.*pkg:true'
}

function test_line_wrapping() {
  local -r pkg=$FUNCNAME
  create_pkg $pkg
  bazel test --curses=yes --color=yes --terminal_columns=5 $pkg:true 2>$TEST_log || fail "bazel test failed"
  # curses are used to delete at least one line
  expect_log $'\x1b\[1A\x1b\[K'
  # something is written in green
  expect_log $'\x1b\[32m'
  # lines are wrapped, hence at least one line should end with backslash
  expect_log '\\'$'\r''$\|\\$'
}

function test_noshow_progress() {
  bazel test --noshow_progress --curses=yes --color=yes \
    pkg:true 2>$TEST_log || fail "${PRODUCT_NAME} test failed"
  # Info messages should still go through
  expect_log 'Elapsed time'
  # no progress indicator is shown
  expect_not_log '\[[0-9,]* / [0-9,]*\]'
}

function test_basic_progress_no_curses() {
  bazel test --curses=no --color=yes pkg:true 2>$TEST_log \
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
  bazel test --curses=no --color=yes --terminal_columns=9 \
    pkg:true 2>$TEST_log || fail "${PRODUCT_NAME} test failed"
  # expect a long-ish status line
  expect_log '\[[0-9,]* / [0-9,]*\]......'
}

function test_pass() {
  bazel test --curses=yes --color=yes pkg:true >$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # PASS is written in green on the same line as the test target
  expect_log 'pkg:true.*'$'\x1b\[32m''.*PASS'
}

function test_fail() {
  bazel test --curses=yes --color=yes pkg:false >$TEST_log \
    && fail "expected failure"
  # FAIL is written in red bold on the same line as the test target
  expect_log 'pkg:false.*'$'\x1b\[31m\x1b\[1m''.*FAIL'
}

function test_timestamp() {
  bazel test --show_timestamps pkg:true 2>$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # expect something that looks like HH:mm:ss
  expect_log '[0-2][0-9]:[0-5][0-9]:[0-6][0-9]'
}

function test_skymeld_ui() {
  bazel build --experimental_merged_skyframe_analysis_execution pkg:true &> "$TEST_log" \
    || fail "${PRODUCT_NAME} test failed."
  expect_log 'Build completed successfully'
}

# Regression test for b/244163231.
function test_skymeld_ui_with_starlark_flags() {
  local -r pkg=$FUNCNAME
  create_pkg $pkg
  mkdir -p "${pkg}/flags"

  cat > "${pkg}/flags/flags.bzl" <<EOF
def _impl(ctx):
  pass

string_flag = rule(
    implementation = _impl,
    build_setting = config.string(flag = True),
)
EOF

  cat > "${pkg}/flags/BUILD" <<EOF
load('//${pkg}/flags:flags.bzl', 'string_flag')

string_flag(
    name = "flag",
    build_setting_default = "a",
)
EOF

  bazel build --experimental_merged_skyframe_analysis_execution \
      --//$pkg/flags:flag=a \
      $pkg:true &> "$TEST_log" || fail "${PRODUCT_NAME} test failed."
  expect_log 'Build completed successfully'
}

# Regression test for b/244163231.
function test_skymeld_ui_works_with_timestamps() {
  bazel build --experimental_merged_skyframe_analysis_execution --show_timestamps \
    pkg:true &> "$TEST_log" \
    || fail "${PRODUCT_NAME} test failed."
  expect_log 'Build completed successfully'
}

function test_info_spacing() {
  # Verify that the output of "bazel info" is suitable for backtick escapes,
  # in particular free carriage-return characters.
  BAZEL_INFO_OUTPUT=XXX`bazel info workspace`XXX
  echo "$BAZEL_INFO_OUTPUT" | grep -q 'XXX[^'$'\r'']*XXX' \
    || fail "${PRODUCT_NAME} info output spaced as $BAZEL_INFO_OUTPUT"
}

function test_query_spacing() {
  # Verify that the output of "bazel query" is suitable for consumption by
  # other tools, i.e., contains only result lines, separated only by newlines.
  BAZEL_QUERY_OUTPUT=`bazel query 'deps(//pkg:true)'`
  echo "$BAZEL_QUERY_OUTPUT" | grep -q -v '^[@/]' \
   && fail "bazel query output is >$BAZEL_QUERY_OUTPUT<" || true
  if ! $is_windows; then
    echo "$BAZEL_QUERY_OUTPUT" | grep -q $'\r' \
     && fail "bazel query output is >$BAZEL_QUERY_OUTPUT<" || true
  fi
}

function test_query_progress() {
  # Verify that some form of progress is reported during bazel query
  bazel query 'deps(//pkg:true)' 2> "${TEST_log}"
  expect_log 'Loading:.*packages loaded'
}

function test_clean_nobuild {
  bazel clean 2>$TEST_log \
   || fail "bazel shutdown failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_clean_color_nobuild {
  bazel clean --color=yes 2>$TEST_log \
   || fail "bazel shutdown failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_help_nobuild {
  bazel help 2>$TEST_log \
   || fail "bazel help failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_help_color_nobuild {
  bazel help --color=yes 2>$TEST_log \
   || fail "bazel help failed"
  expect_not_log "actions running"
  expect_not_log "Building"
}

function test_version_nobuild {
  bazel version --curses=yes 2>$TEST_log \
   || fail "bazel version failed"
  expect_not_log "action"
  expect_not_log "Building"
}

function test_version_nobuild_announce_rc {
  bazel version --curses=yes --announce_rc 2>$TEST_log \
   || fail "bazel version failed"
  expect_not_log "action"
  expect_not_log "Building"
}

function test_subcommand {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel build -s pkg:gentext 2>$TEST_log \
    || fail "bazel build failed"
  expect_log "here be dragons"
}

function test_subcommand_notdefault {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel build pkg:gentext 2>$TEST_log \
    || fail "bazel build failed"
  expect_not_log "dragons"
}

function test_loading_progress {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel test pkg:true 2>$TEST_log \
    || fail "${PRODUCT_NAME} test failed"
  # some progress indicator is shown during loading
  expect_log 'Loading.*[0-9,]* packages'
}

function test_failure_scrollback_buffer_curses {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel test --curses=yes --color=yes \
    --nocache_test_results pkg:false pkg:slow 2>$TEST_log \
    && fail "expected failure"
  # Some line starts with FAIL in red bold.
  expect_log '^'$'\(.*\x1b\[K\)*\x1b\[31m\x1b\[1mFAIL:'
}

function test_terminal_title {
  bazel test --curses=yes \
    --progress_in_terminal_title pkg:true \
    2>$TEST_log || fail "${PRODUCT_NAME} test failed"
  # The terminal title is changed
  expect_log $'\x1b\]0;.*\x07'
}

function test_failure_scrollback_buffer {
  bazel clean || fail "${PRODUCT_NAME} clean failed"
  bazel test --curses=no --color=yes \
    --nocache_test_results pkg:false pkg:slow 2>$TEST_log \
    && fail "expected failure"
  # Some line starts with FAIL in red bold.
  expect_log '^'$'\x1b\[31m\x1b\[1mFAIL:'
}

function test_streamed {
  bazel test --curses=yes --color=yes \
    --nocache_test_results --test_output=streamed pkg:output >$TEST_log \
    || fail "expected success"
  expect_log 'foobar'
}

function test_stdout_bundled {
    # Verify that the error message is part of the error event
    bazel build --experimental_ui_debug_all_events \
          error:failwitherror > "${TEST_log}" 2>&1 \
    && fail "expected failure" || :
    grep -A1 '^ERROR' "${TEST_log}" \
        | grep -q "with STDOUT: Here is the error message" \
        || fail "Error message not bundled"
}

function test_experimental_ui_attempt_to_print_relative_paths_failing_action() {
  # On the BazelCI Windows environment, `pwd` returns a string that uses a
  # lowercase drive letter and unix-style path separators (e.g. '/c/') with
  # a lowercase drive letter. But internally in Bazel, Path#toString
  # unconditionally uses an uppercase drive letter (see
  # WindowsOsPathPolicy#normalize). I want these tests to check for exact
  # string contents (that's the entire goal of the flag being tested), but I
  # don't want them to be brittle across different Windows environments, so
  # I've disabled them for now.
  # TODO(nharmata): Fix this.
  [[ "$is_windows" == "true" ]] && return 0

  bazel clean || fail "${PRODUCT_NAME} clean failed"

  bazel build --attempt_to_print_relative_paths=false \
      error:failwitherror > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_log "^ERROR: $(pwd)/error/BUILD:1:8: Executing genrule //error:failwitherror failed: "

  bazel build --attempt_to_print_relative_paths=true \
      error:failwitherror > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_log "^ERROR: error/BUILD:1:8: Executing genrule //error:failwitherror failed: "
  expect_not_log "$(pwd)/error/BUILD"
}

function test_experimental_ui_attempt_to_print_relative_paths_pkg_error() {
  # See the note in the test case above for why this is disabled on Windows.
  # TODO(nharmata): Fix this.
  [[ "$is_windows" == "true" ]] && return 0

  bazel clean || fail "${PRODUCT_NAME} clean failed"

  bazel build --attempt_to_print_relative_paths=false \
      pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_log "^ERROR: $(pwd)/bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"

  bazel build --attempt_to_print_relative_paths=true \
      pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_log "^ERROR: bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_not_log "$(pwd)/bzl/bzl.bzl"
}

function test_fancy_symbol_encoding() {
    bazel build //fancyOutput:withFancyOutput > "${TEST_log}" 2>&1 \
        || fail "expected success"
    expect_log $'\xF0\x9F\x8D\x83'
}

function test_ui_events_filters() {
  bazel clean || fail "${PRODUCT_NAME} clean failed"

  bazel build pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_log "^ERROR: .*/bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_log "^WARNING: Target pattern parsing failed."
  expect_log "^INFO: Elapsed time"

  bazel build --ui_event_filters=-error pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_not_log "^ERROR: .*bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_log "^WARNING: Target pattern parsing failed."
  expect_log "^INFO: Elapsed time"

  bazel build --ui_event_filters=info pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_not_log "^ERROR: .*/bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_not_log "^WARNING: Target pattern parsing failed."
  expect_log "^INFO: Elapsed time"

  bazel build --ui_event_filters= pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_not_log "^ERROR: .*/bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_not_log "^WARNING: Target pattern parsing failed."
  expect_not_log "^INFO: Elapsed time"

  bazel build --ui_event_filters=-error --ui_event_filters=+error \
      pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_log "^ERROR: .*bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_log "^WARNING: Target pattern parsing failed."
  expect_log "^INFO: Elapsed time"

  bazel build --ui_event_filters= --ui_event_filters=+info pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_not_log "^ERROR: .*/bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_not_log "^WARNING: Target pattern parsing failed."
  expect_log "^INFO: Elapsed time"

  bazel build --ui_event_filters=warning --ui_event_filters=info --ui_event_filters=+error \
      pkgloadingerror:all > "${TEST_log}" 2>&1 && fail "expected failure"
  expect_log "^ERROR: .*/bzl/bzl.bzl:1:5: name 'invalidsyntax' is not defined"
  expect_not_log "^WARNING: Target pattern parsing failed."
  expect_log "^INFO: Elapsed time"
}

function test_max_stdouterr_bytes_capping_behavior() {
  mkdir -p outs
  cat >outs/BUILD <<'EOF'
genrule(
  name = "short-stdout",
  outs = ["short-stdout.txt"],
  cmd = "echo 'abc'; touch $@",
)
genrule(
  name = "short-stderr",
  outs = ["short-stderr.txt"],
  cmd = "echo 'abc' 1>&2; touch $@",
)
genrule(
  name = "long-stdout",
  outs = ["long-stdout.txt"],
  cmd = "echo 'long line of text'; touch $@",
)
genrule(
  name = "long-stderr",
  outs = ["long-stderr.txt"],
  cmd = "echo 'long line of text' 1>&2; touch $@",
)
genrule(
  name = "short-stdout-long-stderr",
  outs = ["short-stdout-long-stderr.txt"],
  cmd = "echo 'abc'; echo 'long line of text' 1>&2; touch $@",
)
genrule(
  name = "long-stdout-short-stderr",
  outs = ["long-stdout-short-stderr.txt"],
  cmd = "echo 'abc' 1>&2; echo 'long line of text'; touch $@",
)
EOF

  for n in stdout stderr; do
    bazel build --experimental_ui_max_stdouterr_bytes=5 "//outs:short-$n" \
        >"${TEST_log}" 2>&1 || fail "build failed"
    expect_log 'abc'
    expect_not_log 'exceeds maximum size'

    bazel build --experimental_ui_max_stdouterr_bytes=5 "//outs:long-$n" \
        >"${TEST_log}" 2>&1 || fail "build failed"
    expect_not_log 'long line of text'
    expect_log 'exceeds maximum size'
  done

  bazel build --experimental_ui_max_stdouterr_bytes=5 \
      //outs:short-stdout-long-stderr \
      >"${TEST_log}" 2>&1 || fail "build failed"
  expect_log 'abc'
  expect_log 'stderr .*/actions/stderr-.* exceeds maximum size'

  bazel build --experimental_ui_max_stdouterr_bytes=5 \
      //outs:long-stdout-short-stderr \
      >"${TEST_log}" 2>&1 || fail "build failed"
  expect_log 'abc'
  expect_log 'stdout .*/actions/stdout-.* exceeds maximum size'
}

function test_max_stdouterr_bytes_is_for_individual_outputs() {
  mkdir -p outs
  cat >>outs/BUILD <<'EOF'
[genrule(
    name = "out-%d" % i,
    outs = [("out-%d.txt") % i],
    cmd = ("echo '>>>%d<<<'; touch $@") % i,
) for i in range(1000)]
EOF
  # Set --experimental_ui_max_stdouterr_bytes to a low-enough number that
  # tolerates a single stdout and run many concurrent jobs at once to ensure
  # that this limit applies to individual outputs even if we are trying to read
  # all files concurrently.
  bazel build --experimental_ui_max_stdouterr_bytes=20 --jobs=200 //outs/... \
      >"${TEST_log}" 2>&1 || fail "build failed"
  for i in $(seq 0 999); do
    expect_log ">>>${i}<<<"
  done
}

function test_interleaved_errors_and_progress() {
  # Background process necessary to interrupt Bazel doesn't go well on Windows.
  [[ "$is_windows" == true ]] && return
  mkdir -p foo
  cat > foo/BUILD <<'EOF'
genrule(name = 'sleep', outs = ['sleep.out'], cmd = 'sleep 10000')
genrule(name = 'fail',
        outs = ['fail.out'],
        srcs = [':multiline.sh'],
        cmd = '$(location :multiline.sh)'
)
EOF
  cat > foo/multiline.sh <<'EOF'
echo "This
is
a
multiline error message
before
failure"
false
EOF
  chmod +x foo/multiline.sh
  bazel build -k //foo:all --curses=yes >& "$TEST_log" &
  pid="$!"
  while ! grep -q "multiline error message" "$TEST_log" ; do
    sleep 1
  done
  while ! grep -q 'Executing genrule //foo:sleep' "$TEST_log" ; do
    sleep 1
  done
  kill -SIGINT "$pid"
  wait "$pid" || exit_code="$?"
  [[ "$exit_code" == 8 ]] || fail "Should have been interrupted: $exit_code"
  tr -s <"$TEST_log" '\n' '@' |
      grep -q 'Executing genrule //foo:fail failed:[^@]*@This@is@a@multiline error message@before@failure@.*Executing genrule //foo:sleep;' \
      || fail "Unified genrule error message not found"
  # Make sure server is still usable.
  bazel info server_pid >& "$TEST_log" || fail "Couldn't use server"
}

function test_progress_bar_after_stderr() {
  mkdir -p foo
  cat > foo/BUILD <<'EOF'
genrule(name = 'fail', outs = ['fail.out'], cmd = 'false')
sh_test(name = 'foo', data = [':fail'], srcs = ['foo.sh'])
EOF
  touch foo/foo.sh
  chmod +x foo/foo.sh
  # Build event file needed so UI considers build to continue after failure.
  ! bazel test --build_event_json_file=bep.json --curses=yes --color=yes \
      //foo:foo &> "$TEST_log" || fail "Expected failure"
  # Expect to see exactly one failure message.
  expect_log_n '\[31m\[1mERROR: \[0mBuild did NOT complete successfully' 1
}

function test_bazel_run_error_visible() {
  mkdir -p foo
  cat > foo/BUILD <<'EOF'
sh_test(
  name = 'foo',
  srcs = ['foo.sh'],
  shard_count = 2,
)
EOF
  touch foo/foo.sh
  chmod +x foo/foo.sh
  bazel run --curses=yes //foo &> "$TEST_log" && "Expected failure"
  expect_log "ERROR: 'run' only works with tests with one shard"
  # If we would print this again after the run failed, we would overwrite the
  # error message above.
  expect_log_n "INFO: Build completed successfully, [456] total actions" 1
}

run_suite "Integration tests for ${PRODUCT_NAME}'s UI"
