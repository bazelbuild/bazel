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

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up_jobcount() {
  tmp=$(mktemp -d ${TEST_TMPDIR}/testjobs.XXXXXXXX)

  # We use hardlinks to this file as a communication mechanism between
  # test runs.
  touch ${tmp}/counter

  add_rules_shell "MODULE.bazel"
  mkdir -p dir

  cat <<EOF > dir/test.sh
#!/bin/sh
# hard link
z=\$(mktemp -u ${tmp}/tmp.XXXXXXXX)
ln ${tmp}/counter \${z}

# Make sure other test runs have started too.
sleep 1
nlink=\$(ls -l ${tmp}/counter | awk '{print \$2}')

# 4 links = 3 jobs + ${tmp}/counter
if [ "\$nlink" -gt 4 ] ; then
  echo found "\$nlink" hard links to file, want 4 max.
  exit 1
fi

# Ensure that we don't remove before other runs have inspected the file.
sleep 1
rm \${z}

EOF

  chmod +x dir/test.sh

  cat <<EOF > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
  name = "test",
  srcs = [ "test.sh" ],
  size = "small",
)
EOF
}

# We have to use --spawn_strategy=standalone, because the test actions
# communicate with each other via a hard-linked file.
function test_3_cpus() {
  set_up_jobcount
  # 3 CPUs, so no more than 3 tests in parallel.
  bazel test --spawn_strategy=standalone --test_output=errors \
    --local_test_jobs=0 --local_cpu_resources=3 \
    --runs_per_test=10 //dir:test
}

function test_3_local_jobs() {
  set_up_jobcount
  # 3 local test jobs, so no more than 3 tests in parallel.
  bazel test --spawn_strategy=standalone --test_output=errors \
    --local_test_jobs=3 --local_cpu_resources=10 \
    --runs_per_test=10 //dir:test
}

function test_tmpdir() {
  add_rules_shell "MODULE.bazel"
  mkdir -p foo
  cat > foo/bar_test.sh <<'EOF'
#!/bin/sh
set -e
echo TEST_TMPDIR=$TEST_TMPDIR
touch "$TEST_TMPDIR/foo"
EOF
  chmod +x foo/bar_test.sh
  cat > foo/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "bar_test",
    srcs = ["bar_test.sh"],
)
EOF
  bazel test --test_output=all //foo:bar_test >& $TEST_log || \
    fail "Running sh_test failed"
  expect_log "TEST_TMPDIR=/.*"

  bazel test --nocache_test_results --test_output=all --test_tmpdir=$TEST_TMPDIR //foo:bar_test \
    >& $TEST_log || fail "Running sh_test failed"
  expect_log "TEST_TMPDIR=$TEST_TMPDIR"
}

function test_env_vars() {
  add_rules_shell "MODULE.bazel"
  mkdir -p foo
  cat > foo/testenv.sh <<'EOF'
#!/bin/sh
echo "pwd: $PWD"
echo "src: $TEST_SRCDIR"
echo "ws: $TEST_WORKSPACE"
EOF
  chmod +x foo/testenv.sh
  cat > foo/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "foo",
    srcs = ["testenv.sh"],
)
EOF

  bazel test --test_output=all //foo &> $TEST_log || fail "Test failed"
  expect_log "pwd: .*/foo.runfiles/_main$"
  expect_log "src: .*/foo.runfiles$"
  expect_log "ws: _main$"
}

function test_env_vars_override() {
  add_rules_shell "MODULE.bazel"
  mkdir -p foo
  cat > foo/testenv.sh <<'EOF'
#!/bin/sh
echo "foo: $FOO"
echo "bar: $BAR"
echo "baz: $BAZ"
echo "test_size: $TEST_SIZE"
echo "ws: $TEST_WORKSPACE"
EOF
  chmod +x foo/testenv.sh
  cat > foo/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
    name = "foo",
    srcs = ["testenv.sh"],
    size = "small",
    env = {
      "FOO": "frombuild",
      "TEST_SIZE": "ignored",
    },
    env_inherit = [
      "BAZ"
    ],
)
EOF

 # The next line ensures that the test passes in IPv6-only networks on macOS.
  if is_darwin; then
    export JAVA_TOOL_OPTIONS="-Djava.net.preferIPv6Addresses=true"
    export STARTUP_OPTS="--host_jvm_args=-Djava.net.preferIPv6Addresses=true"
  else
    export STARTUP_OPTS=""
  fi

  # Test BAR is set from --action_env
  BAZ=fromaction bazel --ignore_all_rc_files $STARTUP_OPTS test --test_output=all \
    --action_env=BAR=fromcli --action_env=BAZ \
    //foo &> $TEST_log || fail "Test failed"
  expect_log "foo: frombuild"
  expect_log "bar: fromcli"
  expect_log "baz: fromaction"
  expect_log "test_size: small"
  expect_log "ws: _main$"

  # Test FOO from the BUILD file wins
  # Test BAR is set from --test_env
  # Test BAZ is set from --test_env
  BAZ=fromtest bazel --ignore_all_rc_files $STARTUP_OPTS test --test_output=all \
    --action_env=FOO=fromcli --test_env=FOO=fromcli --test_env=BAR=fromcli \
    --test_env=BAZ //foo &> $TEST_log || fail "Test failed"
  expect_log "foo: frombuild"
  expect_log "bar: fromcli"
  expect_log "baz: fromtest"
  expect_log "test_size: small"
  expect_log "ws: _main$"
}

function test_runfiles_java_runfiles_merges_env_vars() {
  runfiles_merges_runfiles_env_vars JAVA_RUNFILES PYTHON_RUNFILES
}

function test_runfiles_python_runfiles_merges_env_vars() {
  runfiles_merges_runfiles_env_vars PYTHON_RUNFILES JAVA_RUNFILES
}

# Usage: runfiles_merges_runfiles_env_vars overridden unchanged
function runfiles_merges_runfiles_env_vars() {
  add_rules_shell "MODULE.bazel"
  local -r overridden=$1
  local -r unchanged=$2
  mkdir -p foo
  cat > foo/foo.sh <<'EOF'
#!/bin/sh
echo "JAVA_RUNFILES: ${JAVA_RUNFILES}"
echo "PYTHON_RUNFILES: ${PYTHON_RUNFILES}"
EOF
  chmod +x foo/foo.sh
  cat > foo/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name = "foo", srcs = ["foo.sh"])
EOF

  bazel test --test_env="${overridden}"=override --test_output=all \
      //foo >& "${TEST_log}" || fail "Test failed"

  expect_log "${overridden}: /.*/execroot/_main/override"
  expect_log "${unchanged}: /.*/execroot/_main/bazel-out/[^/]\+-fastbuild/bin/foo/foo.runfiles"
}

function test_run_under_external_label_with_options() {
  mkdir -p testing run || fail "mkdir testing run failed"
  cat <<EOF > run/BUILD
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
sh_binary(
  name='under', srcs=['under.sh'],
  visibility=["//visibility:public"],
)
EOF

  touch run/REPO.bazel

  cat <<EOF > run/under.sh
#!/bin/sh
echo running under @run//:under "\$*"
EOF
  chmod u+x run/under.sh

  cat <<EOF > testing/passing_test.sh
#!/bin/sh
exit 0
EOF
  chmod u+x testing/passing_test.sh

  cat <<EOF > testing/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
  name = "passing_test" ,
  srcs = [ "passing_test.sh" ])
EOF

  cat <<EOF > MODULE.bazel
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
    name = "run",
    path = "./run",
)
EOF
  add_rules_shell "MODULE.bazel"

  bazel test //testing:passing_test --run_under='@run//:under -c' \
    --test_output=all >& $TEST_log || fail "Expected success"

  expect_log 'running under @run//:under -c testing/passing_test'
  expect_log 'passing_test *PASSED'
  expect_log '1 test passes.$'
}

function test_run_under_label_with_options() {
  add_rules_shell "MODULE.bazel"
  mkdir -p testing run || fail "mkdir testing run failed"
  cat <<EOF > run/BUILD
load("@rules_shell//shell:sh_binary.bzl", "sh_binary")
sh_binary(
  name='under', srcs=['under.sh'],
  visibility=["//visibility:public"],
)
EOF

  cat <<EOF > run/under.sh
#!/bin/sh
echo running under //run:under "\$*"
EOF
  chmod u+x run/under.sh

  cat <<EOF > testing/passing_test.sh
#!/bin/sh
exit 0
EOF
  chmod u+x testing/passing_test.sh

  cat <<EOF > testing/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
  name = "passing_test" ,
  srcs = [ "passing_test.sh" ])
EOF

  bazel test //testing:passing_test --run_under='//run:under -c' \
    --test_output=all >& $TEST_log || fail "Expected success"
  expect_log 'running under //run:under -c testing/passing_test'
  expect_log 'passing_test *PASSED'
  expect_log '1 test passes.$'
}

# This test uses "--ignore_all_rc_files" since outside .bazelrc files can pollute
# this environment. Just "--bazelrc=/dev/null" is not sufficient to fix.
function test_run_under_path() {
  add_rules_shell "MODULE.bazel"
  mkdir -p testing || fail "mkdir testing failed"
  cat > testing/BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name='t1', srcs=['t1.sh'])
EOF
  cat <<EOF > testing/t1.sh
#!/bin/sh
exit 0
EOF
  chmod u+x testing/t1.sh

  mkdir -p scripts
  cat <<EOF > scripts/hello
#!/bin/sh
echo "hello script!!!" "\$@"
EOF
  chmod u+x scripts/hello

  # We don't just use the local PATH, but use the test's PATH, which is more restrictive.
  PATH=$PATH:$PWD/scripts bazel --ignore_all_rc_files test //testing:t1 -s --run_under=hello \
    --test_output=all --incompatible_strict_action_env=true >& $TEST_log && fail "Expected failure"

  # With --action_env=PATH, the local PATH is forwarded to the test.
  PATH=$PATH:$PWD/scripts bazel test //testing:t1 -s --run_under=hello \
    --test_output=all >& $TEST_log || fail "Expected success"
  expect_log 'hello script!!! testing/t1'

  # We need to forward the PATH to make it work.
  PATH=$PATH:$PWD/scripts bazel test //testing:t1 -s --run_under=hello \
    --test_env=PATH --test_output=all >& $TEST_log || fail "Expected success"
  expect_log 'hello script!!! testing/t1'

  # Make sure it still works if --run_under includes an arg.
  PATH=$PATH:$PWD/scripts bazel test //testing:t1 \
    -s --run_under='hello "some_arg   with"          space' \
    --test_env=PATH --test_output=all >& $TEST_log || fail "Expected success"
  expect_log 'hello script!!! some_arg   with space testing/t1'

  # Make sure absolute path works also
  bazel test //testing:t1 --run_under=$PWD/scripts/hello \
    -s --test_output=all >& $TEST_log || fail "Expected success"
  expect_log 'hello script!!! testing/t1'
}

function test_run_under_external_file_with_options() {
  # Set up the external repo.
  local run_repo=$TEST_TMPDIR/run
  mkdir -p $run_repo || fail "mkdir run_repo failed"
  touch $run_repo/REPO.bazel

  cat <<EOF > $run_repo/BUILD
exports_files(["under.sh"])
EOF
  cat <<EOF > $run_repo/under.sh
#!/bin/sh
echo running under @run//:under "\$*"
EOF
  chmod u+x $run_repo/under.sh


  # Set up the main repo.
  cat <<EOF > MODULE.bazel
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
    name = "run",
    path = "../run",
)
EOF
  add_rules_shell "MODULE.bazel"

  mkdir -p testing || fail "mkdir testing failed"

  cat <<EOF > testing/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
  name = "passing_test" ,
  srcs = [ "passing_test.sh" ])
EOF
  cat <<EOF > testing/passing_test.sh
#!/bin/sh
exit 0
EOF
  chmod u+x testing/passing_test.sh


  bazel test //testing:passing_test -s --run_under='@run//:under.sh -c' \
    --test_output=all >& $TEST_log || fail "Expected success"

  expect_log 'running under @run//:under -c testing/passing_test'
  expect_log 'passing_test *PASSED'
  expect_log '1 test passes.$'
}

function test_test_timeout() {
  add_rules_shell "MODULE.bazel"
  mkdir -p dir

  cat <<EOF > dir/test.sh
#!/bin/sh
sleep 3
exit 0

EOF

  chmod +x dir/test.sh

  cat <<EOF > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "test",
    timeout = "short",
    srcs = [ "test.sh" ],
    size = "small",
  )
EOF

  bazel test --test_timeout=2 //dir:test &> $TEST_log && fail "should have timed out"
  expect_log "TIMEOUT"
  bazel test --test_timeout=20 //dir:test &> $TEST_log || fail "expected success"
}

# Makes sure that runs_per_test_detects_flakes detects FLAKY if any of the 5
# attempts passes (which should cover all cases of being picky about the
# first/last/etc ones only being counted).
# We do this using an un-sandboxed test which keeps track of how many runs there
# have been using files which are undeclared inputs/outputs.
function test_runs_per_test_detects_flakes() {
  # Directory for counters
  local COUNTER_DIR="${TEST_TMPDIR}/counter_dir"
  mkdir -p "${COUNTER_DIR}"
  add_rules_shell "MODULE.bazel"

  for (( i = 1 ; i <= 5 ; i++ )); do

    # This file holds the number of the next run
    echo 1 > "${COUNTER_DIR}/$i"
    cat <<EOF > test$i.sh
#!/bin/sh
i=\$(cat "${COUNTER_DIR}/$i")

# increment the hidden state
echo \$((i + 1)) > "${COUNTER_DIR}/$i"

# succeed exactly once.
exit \$((i != $i))
}
EOF
    chmod +x test$i.sh
    cat <<EOF > BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(name = "test$i", srcs = [ "test$i.sh" ])
EOF
    bazel test --spawn_strategy=standalone \
        --jobs=1 \
        --runs_per_test=5 \
        --runs_per_test_detects_flakes \
        //:test$i &> $TEST_log || fail "should have succeeded"
    expect_log "FLAKY, failed in 4 out of 5"
  done
}

function test_runs_per_test_detects_flakes_cancel_concurrent() {
  # Directory for counters
  local COUNTER_DIR="${TEST_TMPDIR}/counter_dir"
  mkdir -p "${COUNTER_DIR}"
  add_rules_shell "MODULE.bazel"

  # This file holds the number of the next run
  echo 1 > "${COUNTER_DIR}/counter"
  cat <<EOF > test.sh
#!/bin/sh
i=\$(cat "${COUNTER_DIR}/counter")

echo "Run \$i"

# increment the hidden state
echo \$((i + 1)) > "${COUNTER_DIR}/counter"

# succeed in the first two runs, fail in the third one
exit \$((i <= 2 ? 0 : 1))
}
EOF
  chmod +x test.sh
  cat <<EOF > BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(name = "test", srcs = [ "test.sh" ])
EOF
  bazel test --spawn_strategy=standalone \
      --jobs=1 \
      --runs_per_test=5 \
      --runs_per_test_detects_flakes \
      --experimental_cancel_concurrent_tests=on_failed \
      //:test &> $TEST_log || fail "should have succeeded"
  expect_log_n "^FAIL: //:test" 1
  expect_log_n "^CANCELLED: //:test" 2
  expect_log "FLAKY, failed in 1 out of 3"
}

# Tests that the test.xml is extracted from the sandbox correctly.
function test_xml_is_present() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"
  cat <<'EOF' > dir/test.sh
#!/bin/sh
echo HELLO > $XML_OUTPUT_FILE
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "test",
    srcs = [ "test.sh" ],
  )
EOF

  bazel test -s --test_output=streamed //dir:test &> $TEST_log || fail "expected success"

  xml_log=bazel-testlogs/dir/test/test.xml
  [ -s $xml_log ] || fail "$xml_log was not present after test"
}

function write_test_xml_timeout_files() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"

  cat <<'EOF' > dir/test.sh
#!/usr/bin/env bash
echo "xmltest"
echo -n "before "
# Invalid XML character
perl -e 'print "\x1b"'
# Invalid UTF-8 characters
perl -e 'print "\xc0\x00\xa0\xa1"'
echo " after"
# ]]> needs escaping
echo "<!CDATA[]]>"
sleep 10
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "test",
    srcs = [ "test.sh" ],
  )
EOF
}

function test_xml_is_present_when_timingout() {
  write_test_xml_timeout_files
  bazel test -s --test_timeout=1 --nocache_test_results \
     //dir:test &> $TEST_log && fail "should have failed" || true

  xml_log=bazel-testlogs/dir/test/test.xml
  [[ -s "${xml_log}" ]] || fail "${xml_log} was not present after test"
  cat "${xml_log}" > $TEST_log
  # The new script does not convert exit codes to signals.
  expect_log '"exited with error code 142"'
  expect_log '<system-out>'
  # The entire test log is inlined into the xml file, which includes a header
  # generated by test-setup.sh; the header starts with
  # "exec ${PAGER:-/usr/bin/less}".
  expect_log '<!\[CDATA\[exec ${PAGER:-/usr/bin/less}'
  expect_log 'before ????? after'
  # This is different from above, since we're using a SIGTERM trap to output
  # timing information.
  expect_log '<!CDATA\[\]\]>\]\]<!\[CDATA\[>'
  expect_log '</system-out>'
}

# Tests that the test.xml and test.log are correct and the test does not
# hang when the test launches a subprocess.
function test_subprocess_non_timeout() {
  add_rules_shell "MODULE.bazel"
  mkdir -p dir

  cat <<'EOF' > dir/test.sh
echo "Pretending to sleep..."
sleep 600 &
echo "Finished!" >&2
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "test",
    timeout = "short",
    srcs = [ "test.sh" ],
  )
EOF

  bazel test --test_output=streamed --test_timeout=2 \
     //dir:test &> $TEST_log || fail "expected success"

  xml_log=bazel-testlogs/dir/test/test.xml
  expect_log 'Pretending to sleep...'
  expect_log 'Finished!'
  [ -s "${xml_log}" ] || fail "${xml_log} was not present after test"
  cp "${xml_log}" $TEST_log
  expect_log_once "testcase"
  expect_log 'Pretending to sleep...'
  expect_log 'Finished!'
}

# Check that fallback xml output is correctly generated for sharded tests.
function test_xml_fallback_for_sharded_test() {
  add_rules_shell "MODULE.bazel"
  mkdir -p dir

  cat <<'EOF' > dir/test.sh
#!/bin/sh
touch "$TEST_SHARD_STATUS_FILE"
exit $((TEST_SHARD_INDEX == 1))
EOF

  chmod +x dir/test.sh

  cat <<EOF > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
  name = "test",
  srcs = [ "test.sh" ],
  shard_count = 2,
)
EOF

  bazel test //dir:test && fail "should have failed" || true

  cp bazel-testlogs/dir/test/shard_1_of_2/test.xml $TEST_log
  expect_log "errors=\"0\""
  expect_log_once "testcase"
  expect_log "name=\"dir/test_shard_1/2\""
  cp bazel-testlogs/dir/test/shard_2_of_2/test.xml $TEST_log
  expect_log "errors=\"1\""
  expect_log_once "testcase"
  expect_log "name=\"dir/test_shard_2/2\""
}

# Simple test that we actually enforce testonly, see #1923.
function test_testonly_is_enforced() {
  mkdir -p testonly
  cat <<'EOF' >testonly/BUILD
genrule(
    name = "testonly",
    srcs = [],
    cmd = "echo testonly | tee $@",
    outs = ["testonly.txt"],
    testonly = 1,
)
genrule(
    name = "not-testonly",
    srcs = [":testonly"],
    cmd = "echo should fail | tee $@",
    outs = ["not-testonly.txt"],
)
EOF
    bazel build //testonly &>$TEST_log || fail "Building //testonly failed"
    bazel build //testonly:not-testonly &>$TEST_log && fail "Should have failed" || true
    expect_log "'//testonly:not-testonly' depends on testonly target '//testonly:testonly'"
}

function test_always_xml_output() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"

  cat <<EOF > dir/success.sh
#!/bin/sh
exit 0
EOF
  cat <<EOF > dir/fail.sh
#!/bin/sh
exit 1
EOF

  chmod +x dir/{success,fail}.sh

  cat <<EOF > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "success",
    srcs = [ "success.sh" ],
)
sh_test(
    name = "fail",
    srcs = [ "fail.sh" ],
)
EOF

  bazel test //dir:all &> $TEST_log && fail "should have failed" || true
  [ -f "bazel-testlogs/dir/success/test.xml" ] \
    || fail "No xml file for //dir:success"
  [ -f "bazel-testlogs/dir/fail/test.xml" ] \
    || fail "No xml file for //dir:fail"

  cat bazel-testlogs/dir/success/test.xml >$TEST_log
  expect_log "errors=\"0\""
  expect_log_once "testcase"
  expect_log_once "duration=\"[0-9]\+\""
  expect_log "name=\"dir/success\""
  cat bazel-testlogs/dir/fail/test.xml >$TEST_log
  expect_log "errors=\"1\""
  expect_log_once "testcase"
  expect_log_once "duration=\"[0-9]\+\""
  expect_log "name=\"dir/fail\""

  bazel test //dir:all --run_under=exec &> $TEST_log && fail "should have failed" || true
  cp bazel-testlogs/dir/success/test.xml $TEST_log
  expect_log "errors=\"0\""
  expect_log_once "testcase"
  expect_log_once "duration=\"[0-9]\+\""
  expect_log "name=\"dir/success\""
  cp bazel-testlogs/dir/fail/test.xml $TEST_log
  expect_log "errors=\"1\""
  expect_log_once "testcase"
  expect_log_once "duration=\"[0-9]\+\""
  expect_log "name=\"dir/fail\""
}

function test_detailed_test_summary_for_failed_test() {
  copy_examples
  add_rules_java "MODULE.bazel"
  setup_javatest_support

  local java_native_tests=//examples/java-native/src/test/java/com/example/myproject

  bazel test --test_summary=detailed "${java_native_tests}:fail" >& $TEST_log \
    && fail "Test $* succeed while expecting failure" \
    || true
  expect_log 'FAILED.*com\.example\.myproject\.Fail\.testFail'
}

function test_detailed_test_summary_for_passed_test() {
  copy_examples
  add_rules_java "MODULE.bazel"
  setup_javatest_support

  local java_native_tests=//examples/java-native/src/test/java/com/example/myproject

  bazel test --test_summary=detailed "${java_native_tests}:hello" >& $TEST_log \
    || fail "expected success"
  expect_log 'PASSED.*com\.example\.myproject\.TestHello\.testNoArgument'
  expect_log 'PASSED.*com\.example\.myproject\.TestHello\.testWithArgument'
}

# This test uses "--ignore_all_rc_files" since outside .bazelrc files can pollute
# this environment. Just "--bazelrc=/dev/null" is not sufficient to fix.
function test_flaky_test() {
  add_rules_shell "MODULE.bazel"
  cat >BUILD <<EOF
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(name = "flaky", flaky = True, srcs = ["flaky.sh"])
sh_test(name = "pass", flaky = True, srcs = ["true.sh"])
sh_test(name = "fail", flaky = True, srcs = ["false.sh"])
EOF
  FLAKE_FILE="${TEST_TMPDIR}/flake"
  rm -f "${FLAKE_FILE}"
  cat >flaky.sh <<EOF
#!/bin/sh
if ! [ -f "${FLAKE_FILE}" ]; then
  echo 1 > "${FLAKE_FILE}"
  echo "fail"
  exit 1
fi
echo "pass"
EOF
  cat >true.sh <<EOF
#!/bin/sh
echo "pass"
exit 0
EOF
  cat >false.sh <<EOF
#!/bin/sh
echo "fail"
exit 1
EOF
  chmod +x true.sh flaky.sh false.sh

  # The next line ensures that the test passes in IPv6-only networks on macOS.
  if is_darwin; then
    export JAVA_TOOL_OPTIONS="-Djava.net.preferIPv6Addresses=true"
    export STARTUP_OPTS="--host_jvm_args=-Djava.net.preferIPv6Addresses=true"
  else
    export STARTUP_OPTS=""
  fi

  # We do not use sandboxing so we can trick to be deterministically flaky
  bazel --ignore_all_rc_files $STARTUP_OPTS test --experimental_ui_debug_all_events \
      --spawn_strategy=standalone //:flaky &> $TEST_log \
      || fail "//:flaky should have passed with flaky support"
  [ -f "${FLAKE_FILE}" ] || fail "Flaky test should have created the flake-file!"

  expect_log_once "FAIL.*: //:flaky (.*/flaky/test_attempts/attempt_1.log)"
  expect_log_once "PASS.*: //:flaky"
  expect_log_once "FLAKY: //:flaky"
  cat bazel-testlogs/flaky/test_attempts/attempt_1.log &> $TEST_log
  assert_equals "fail" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"
  assert_equals 1 $(ls bazel-testlogs/flaky/test_attempts/*.log | wc -l)
  cat bazel-testlogs/flaky/test.log &> $TEST_log
  assert_equals "pass" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"

  bazel --ignore_all_rc_files $STARTUP_OPTS test --experimental_ui_debug_all_events //:pass \
      &> $TEST_log || fail "//:pass should have passed"
  expect_log_once "PASS.*: //:pass"
  expect_log_once "PASSED"
  [ ! -d bazel-test_logs/pass/test_attempts ] \
    || fail "Got test attempts while expected non for non-flaky tests"
  cat bazel-testlogs/flaky/test.log &> $TEST_log
  assert_equals "pass" "$(tail -1 bazel-testlogs/flaky/test.log)"

  bazel --ignore_all_rc_files $STARTUP_OPTS test --experimental_ui_debug_all_events //:fail \
      &> $TEST_log && fail "//:fail should have failed" \
      || true
  expect_log_n "FAIL.*: //:fail (.*/fail/test_attempts/attempt_..log)" 2
  expect_log_once "FAIL.*: //:fail (.*/fail/test.log)"
  expect_log_once "FAILED: //:fail"
  expect_log_n ".*/fail/test.log$" 2
  cat bazel-testlogs/fail/test_attempts/attempt_1.log &> $TEST_log
  assert_equals "fail" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"
  assert_equals 2 $(ls bazel-testlogs/fail/test_attempts/*.log | wc -l)
  cat bazel-testlogs/fail/test.log &> $TEST_log
  assert_equals "fail" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"
}

function setup_undeclared_outputs_test() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"

  cat <<'EOF' > dir/test.sh
#!/usr/bin/env bash
mkdir -p "$TEST_UNDECLARED_OUTPUTS_DIR/deeply/nested"
echo "some text" > "$TEST_UNDECLARED_OUTPUTS_DIR/text.txt"
echo "<!DOCTYPE html>" > "$TEST_UNDECLARED_OUTPUTS_DIR/deeply/nested/index.html"
if [[ -z "$FLAKY_FILE" ]]; then
  # Non-flaky execution.
  exit 0
elif [[ -e "$FLAKY_FILE" ]]; then
  # Flaky execution, second run.
  exit 0
else
  # Flaky execution, first run.
  touch "$FLAKY_FILE"
  exit 1
fi
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
  name = "test",
  srcs = ["test.sh"],
)
EOF
}

function test_undeclared_outputs_are_zipped() {
  setup_undeclared_outputs_test

  local -r outputs_manifest=bazel-testlogs/dir/test/test.outputs_manifest/MANIFEST
  local -r outputs_dir=bazel-testlogs/dir/test/test.outputs
  local -r outputs_zip=$outputs_dir/outputs.zip
  local -r output_text=$outputs_dir/text.txt
  local -r output_html=$outputs_dir/deeply/nested/index.html

  bazel test -s --zip_undeclared_test_outputs //dir:test &> $TEST_log || fail "expected success"

  assert_exists "$outputs_zip"
  assert_not_exists "$output_text"
  assert_not_exists "$output_html"

  unzip -q "$outputs_zip" -d unzipped_outputs || fail "failed to unzip $outputs_zip"
  assert_equals "$(cat unzipped_outputs/text.txt)" "some text"
  assert_equals "$(cat unzipped_outputs/deeply/nested/index.html)" "<!DOCTYPE html>"

  assert_exists "$outputs_manifest"
  cat > expected_manifest <<EOF
deeply/nested/index.html	16	text/html
text.txt	10	text/plain
EOF
  assert_equals "$(cat "$outputs_manifest")" "$(cat expected_manifest)"
}

function test_undeclared_outputs_are_zipped_multiple_attempts() {
  setup_undeclared_outputs_test

  local -r outputs_dir=bazel-testlogs/dir/test/test_attempts/attempt_1.outputs
  local -r outputs_zip=$outputs_dir/outputs.zip
  local -r output_text=$outputs_dir/text.txt
  local -r output_html=$outputs_dir/deeply/nested/index.html

  local -r flaky_file=$TEST_TMPDIR/flaky
  rm -f "$flaky_file"

  bazel test -s --zip_undeclared_test_outputs --spawn_strategy=local \
      --flaky_test_attempts=2 --test_env=FLAKY_FILE="$flaky_file" \
      //dir:test &> $TEST_log || fail "expected success"

  assert_exists "$outputs_zip"
  assert_not_exists "$output_text"
  assert_not_exists "$output_html"
}

function test_undeclared_outputs_are_not_zipped() {
  setup_undeclared_outputs_test

  local -r outputs_manifest=bazel-testlogs/dir/test/test.outputs_manifest/MANIFEST
  local -r outputs_dir=bazel-testlogs/dir/test/test.outputs
  local -r outputs_zip=$outputs_dir/outputs.zip
  local -r output_text=$outputs_dir/text.txt
  local -r output_html=$outputs_dir/deeply/nested/index.html

  bazel test -s //dir:test &> $TEST_log || fail "expected success"

  assert_not_exists "$outputs_zip"
  assert_exists "$output_text"
  assert_exists "$output_html"

  assert_equals "$(cat "$output_text")" "some text"
  assert_equals "$(cat "$output_html")" "<!DOCTYPE html>"

  assert_exists "$outputs_manifest"
  cat > expected_manifest <<EOF
deeply/nested/index.html	16	text/html
text.txt	10	text/plain
EOF
  assert_equals "$(cat "$outputs_manifest")" "$(cat expected_manifest)"
}

function test_undeclared_outputs_are_not_zipped_multiple_attempts() {
  setup_undeclared_outputs_test

  local -r outputs_dir=bazel-testlogs/dir/test/test_attempts/attempt_1.outputs
  local -r outputs_zip=$outputs_dir/outputs.zip
  local -r output_text=$outputs_dir/text.txt
  local -r output_html=$outputs_dir/deeply/nested/index.html

  local -r flaky_file=$TEST_TMPDIR/flaky
  rm -f "$flaky_file"

  bazel test -s --spawn_strategy=local \
      --flaky_test_attempts=2 --test_env=FLAKY_FILE="$flaky_file" \
      //dir:test &> $TEST_log || fail "expected success"

  assert_not_exists "$outputs_zip"
  assert_exists "$output_text"
  assert_exists "$output_html"
}

function test_undeclared_outputs_zipped_then_unzipped() {
  setup_undeclared_outputs_test

  local -r outputs_dir=bazel-testlogs/dir/test/test.outputs
  local -r outputs_zip=$outputs_dir/outputs.zip
  local -r output_text=$outputs_dir/text.txt
  local -r output_html=$outputs_dir/deeply/nested/index.html

  bazel test -s --zip_undeclared_test_outputs //dir:test &> $TEST_log || fail "expected success"

  assert_exists "$outputs_zip"
  assert_not_exists "$output_text"
  assert_not_exists "$output_html"

  bazel test -s //dir:test &> $TEST_log || fail "expected success"

  assert_not_exists "$outputs_zip"
  assert_exists "$output_text"
  assert_exists "$output_html"
}

function test_undeclared_outputs_unzipped_then_zipped() {
  setup_undeclared_outputs_test

  local -r outputs_dir=bazel-testlogs/dir/test/test.outputs
  local -r outputs_zip=$outputs_dir/outputs.zip
  local -r output_text=$outputs_dir/text.txt
  local -r output_html=$outputs_dir/deeply/nested/index.html

  bazel test -s //dir:test &> $TEST_log || fail "expected success"

  assert_not_exists "$outputs_zip"
  assert_exists "$output_text"
  assert_exists "$output_html"

  bazel test -s --zip_undeclared_test_outputs //dir:test &> $TEST_log || fail "expected success"

  assert_exists "$outputs_zip"
  assert_not_exists "$output_text"
  assert_not_exists "$output_html"
}

function test_undeclared_outputs_annotations_are_added() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"

  cat <<'EOF' > dir/test.sh
#!/bin/sh
echo "an annotation" > "$TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR/1.part"
echo "another annotation" > "$TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR/2.part"
echo "pass"
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "test",
    srcs = [ "test.sh" ],
  )
EOF

  bazel test -s //dir:test &> $TEST_log || fail "expected success"

  # Newlines are useful around diffs. This helps us get them in bash strings.
  N=$'\n'

  # Check that the undeclared outputs manifest exists and that it has the
  # correct contents.
  annotations=bazel-testlogs/dir/test/test.outputs_manifest/ANNOTATIONS
  [ -s $annotations ] || fail "$annotations was not present after test"
  cat > expected_annotations <<EOF
an annotation
another annotation
EOF
diff expected_annotations "$annotations" > d || fail "$annotations differs from expected:$N$(cat d)$N"
}

function test_no_zip_annotation_manifest_when_no_undeclared_outputs() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"

  cat <<'EOF' > dir/test.sh
#!/bin/sh
echo "pass"
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = "test",
    srcs = [ "test.sh" ],
  )
EOF

  bazel test -s //dir:test &> $TEST_log || fail "expected success"

  # Check that the undeclared outputs directory doesn't exist.
  outputs_zip=bazel-testlogs/dir/test/test.outputs/outputs.zip
  [ ! -e $outputs_zip ] || fail "$outputs_zip was present after test"

  # Check that the undeclared outputs manifest directory doesn't exist.
  outputs_manifest=bazel-testlogs/dir/test/test.outputs_manifest/MANIFEST
  [ ! -d $outputs_manifest ] || fail "$outputs_manifest was present after test"
}

function test_test_with_nobuild_runfile_manifests() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"

  touch dir/test.sh
  chmod u+x dir/test.sh
  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
    name = 'test',
    srcs = ['test.sh'],
)
EOF
  bazel test --nobuild_runfile_manifests //dir:test >& $TEST_log && fail "should have failed"
  expect_log "cannot run local tests with --nobuild_runfile_manifests"
}

function test_test_with_reserved_env_variable() {
  mkdir -p dir
  add_rules_shell "MODULE.bazel"

  touch dir/test.sh
  chmod u+x dir/test.sh
  cat <<'EOF' > dir/BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
    name = 'test',
    srcs = ['test.sh'],
    env = {
      "TEST_NAME": "foo"
    },
)
EOF
  bazel test //dir:test >& $TEST_log && fail "should have failed"
  expect_log "cannot set env variable TEST_NAME=foo because TEST_NAME is reserved"
}

function test_run_from_external_repo_sibling_repository_layout() {
  cat <<EOF > MODULE.bazel
local_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository")
local_repository(
    name = "a",
    path = "./a",
)
EOF
  add_rules_python "MODULE.bazel"

  mkdir -p a
  touch a/REPO.bazel
  cat <<'EOF' > a/BUILD
load("@rules_python//python:py_test.bzl", "py_test")
py_test(
    name = 'x',
    srcs = ['x.py'],
)
EOF
  touch a/x.py

  bazel test --experimental_sibling_repository_layout @a//:x &> $TEST_log \
      || fail "expected success"

  cp $(testlogs_dir +local_repository+a)/x/test.xml $TEST_log
  expect_log "<testsuite name=\"+local_repository+a/x\""
  expect_log "<testcase name=\"+local_repository+a/x\""
}

function test_xml_output_format() {
  add_rules_python "MODULE.bazel"

  cat <<'EOF' > BUILD
load("@rules_python//python:py_test.bzl", "py_test")
py_test(
    name = 'x',
    srcs = ['x.py'],
)
EOF
  touch x.py

  bazel test //:x &> $TEST_log \
      || fail "expected success"

  cat bazel-testlogs/x/test.xml > $TEST_log
  expect_log "<testsuite name=\"x\""
  expect_log "<testcase name=\"x\""
}

function test_shard_status_file_checked() {
  add_rules_shell "MODULE.bazel"
  cat <<'EOF' > BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")

sh_test(
    name = 'x',
    srcs = ['x.sh'],
    shard_count = 2,
)
EOF
  touch x.sh
  chmod +x x.sh

  bazel test \
      --incompatible_check_sharding_support \
      //:x  &> $TEST_log && fail "expected failure"
  expect_log "Sharding requested, but the test runner did not advertise support for it by touching TEST_SHARD_STATUS_FILE."

  echo 'touch "$TEST_SHARD_STATUS_FILE"' > x.sh
  bazel test \
      --incompatible_check_sharding_support \
      //:x  &> $TEST_log || fail "expected success"
}

function test_premature_exit_file_checked() {
  add_rules_shell "MODULE.bazel"
  cat <<'EOF' > BUILD
load("@rules_shell//shell:sh_test.bzl", "sh_test")
sh_test(
    name = 'x',
    srcs = ['x.sh'],
)
EOF
  cat <<'EOF' > x.sh
#!/bin/sh
touch "$TEST_PREMATURE_EXIT_FILE"
echo "fake pass"
exit 0
EOF
  chmod +x x.sh

  bazel test //:x --test_output=errors &> $TEST_log && fail "expected failure"
  expect_log "-- Test exited prematurely (TEST_PREMATURE_EXIT_FILE exists) --"
}

function test_test_env_change_preserves_cache() {
  local -r pkg="pkg${LINENO}"
  mkdir -p "${pkg}"
  cat > "$pkg/BUILD" <<'EOF'
load(":defs.bzl", "my_rule")
my_rule(
  name = "my_rule",
)
EOF
  cat > "$pkg/defs.bzl" <<'EOF'
def _my_rule_impl(ctx):
  print("my_rule is being analyzed")
  out = ctx.actions.declare_file(ctx.label.name)
  ctx.actions.write(out, "hi")
  return [DefaultInfo(files = depset([out]))]
my_rule = rule(_my_rule_impl)
EOF

  bazel build "${pkg}:my_rule" >$TEST_log 2>&1 \
   || fail "expected build to pass"
  expect_log "my_rule is being analyzed"

  bazel build --test_env=FOO=bar "${pkg}:my_rule" >$TEST_log 2>&1 \
   || fail "expected build to pass"
  expect_not_log "my_rule is being analyzed"
}

run_suite "bazel test tests"
