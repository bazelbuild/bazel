#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2018-08-14, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

#### TESTS #############################################################

function test_passing_test_is_reported_correctly() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"
  cat >$pkg/BUILD <<'EOF'
sh_test(
    name = "success",
    size = "small",
    srcs = ["success.sh"],
)
EOF
  cat >$pkg/success.sh <<'EOF'
#!/bin/sh

echo "success.sh is successful"
exit 0
EOF
  chmod +x $pkg/success.sh

  bazel test --nocache_test_results //$pkg:success &>$TEST_log \
      || fail "expected success"
  expect_not_log "success.sh is successful"
  expect_log "^Executed 1 out of 1 test: 1 test passes."
}

function test_failing_test_is_reported_correctly() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"
  cat >$pkg/BUILD <<'EOF'
sh_test(
    name = "fail",
    size = "small",
    srcs = ["fail.sh"],
)
EOF
  cat >$pkg/fail.sh <<'EOF'
#!/bin/sh

echo "fail.sh is failing"
exit 42
EOF
  chmod +x $pkg/fail.sh

  bazel test --nocache_test_results //$pkg:fail &>$TEST_log \
      && fail "expected failure" || true
  expect_log "^//$pkg:fail[[:space:]]\+FAILED in [[:digit:]]\+[\.,][[:digit:]]\+s"
  expect_log "^Executed 1 out of 1 test: 1 fails"
}

function test_build_fail_terse_summary() {
    local -r pkg=$FUNCNAME
    mkdir -p $pkg || fail "mkdir -p $pkg failed"
    cat > $pkg/BUILD <<'EOF'
genrule(
  name = "testsrc",
  outs = ["test.sh"],
  cmd = "false",
)
sh_test(
  name = "failstobuild1",
  srcs = ["test.sh"],
)
sh_test(
  name = "failstobuild2",
  srcs = ["test.sh"],
)
genrule(
  name = "slowtestsrc",
  outs = ["slowtest.sh"],
  cmd = "sleep 20 && echo '#!/bin/sh' > $@ && echo 'true' >> $@ && chmod 755 $@",
)
sh_test(
  name = "willbeskipped",
  srcs = ["slowtest.sh"],
)
EOF
    bazel test --test_summary=terse //$pkg/... &>$TEST_log \
      && fail "expected failure" || :
    expect_not_log 'NO STATUS'
    expect_log 'testsrc'
    expect_log 'were skipped'
}

# Regression test for b/67463263: Tests that spawn subprocesses must not block
# if those subprocesses never finish. If this test fails, because the "my_test"
# test is timing out, it means that Bazel is waiting for the "sleep" to finish,
# which it shouldn't.
function test_process_spawned_by_test_doesnt_block_test_from_completing() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"

  cat > $pkg/BUILD <<'EOF'
java_test(
    name = "my_test",
    main_class = "MyTest",
    srcs = ["MyTest.java"],
    timeout = "short",
    use_testrunner = 0,
)
EOF
  cat > $pkg/MyTest.java <<'EOF'
public class MyTest {
  public static void main(String[] args) throws Exception {
    new ProcessBuilder("sleep", "300").inheritIO().start();
  }
}
EOF

  bazel test //$pkg:my_test &> $TEST_log || fail "expected test to pass"
}

function test_test_suite_non_expansion() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg || fail "mkdir -p $pkg failed"
  cat > $pkg/BUILD <<'EOF'
sh_test(name = 'test_a',
        srcs = [':a.sh'],
)

sh_test(name = 'test_b',
        srcs = [':b.sh'],
)

test_suite(name = 'suite',
)
EOF
  cat > $pkg/a.sh <<'EOF'
#!/bin/sh
exit 0
EOF

  cat > $pkg/b.sh <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x $pkg/a.sh $pkg/b.sh
  bazel test --noexpand_test_suites //$pkg:suite &> $TEST_log || fail "expected test to pass"
  expect_log "//$pkg:test_a"
  expect_log "//$pkg:test_b"
}

function test_print_relative_test_log_paths() {
  # The symlink resolution done by PathPrettyPrinter doesn't seem to work on
  # Windows.
  # TODO(nharmata): Fix this.
  [[ "$is_windows" == "true" ]] && return 0

  local -r pkg="$FUNCNAME"
  mkdir -p "$pkg" || fail "mkdir -p $pkg failed"
  cat > "$pkg"/BUILD <<'EOF'
sh_test(name = 'fail', srcs = ['fail.sh'])
EOF
  cat > "$pkg"/fail.sh <<'EOF'
#!/bin/sh
exit 1
EOF
  chmod +x "$pkg"/fail.sh

  local testlogs_dir=$(bazel info ${PRODUCT_NAME}-testlogs 2> /dev/null)

  bazel test --print_relative_test_log_paths=false //"$pkg":fail &> $TEST_log \
    && fail "expected failure"
  expect_log "^  $testlogs_dir/$pkg/fail/test.log$"

  bazel test --print_relative_test_log_paths=true //"$pkg":fail &> $TEST_log \
    && fail "expected failure"
  expect_log "^  ${PRODUCT_NAME}-testlogs/$pkg/fail/test.log$"
}

run_suite "test tests"
