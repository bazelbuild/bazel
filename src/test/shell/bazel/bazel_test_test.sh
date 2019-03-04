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
    --local_test_jobs=0 --local_resources=10000,3,100 \
    --runs_per_test=10 //dir:test
}

function test_3_local_jobs() {
  set_up_jobcount
  # 3 local test jobs, so no more than 3 tests in parallel.
  bazel test --spawn_strategy=standalone --test_output=errors \
    --local_test_jobs=3 --local_resources=10000,10,100 \
    --runs_per_test=10 //dir:test
}

# TODO(#2228): Re-enable when the tmpdir creation is fixed.
function DISABLED_test_tmpdir() {
  mkdir -p foo
  cat > foo/bar_test.sh <<'EOF'
#!/bin/sh
echo TEST_TMPDIR=$TEST_TMPDIR
EOF
  chmod +x foo/bar_test.sh
  cat > foo/BUILD <<EOF
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

  # If we run `bazel test //src/test/shell/bazel:bazel_test_test` on Linux, it
  # will be sandboxed and this "inner test" creating /foo/bar will actually
  # succeed. If we run it on OS X (or in general without sandboxing enabled),
  # it will fail to create /foo/bar, since obviously we don't have write
  # permissions.
  if bazel test --nocache_test_results --test_output=all \
    --test_tmpdir=/foo/bar //foo:bar_test >& $TEST_log; then
    # We are in a sandbox.
    expect_log "TEST_TMPDIR=/foo/bar"
  else
    # We are not sandboxed.
    expect_log "Could not create TEST_TMPDIR"
  fi
}

function test_env_vars() {
  cat > WORKSPACE <<EOF
workspace(name = "bar")
EOF
  mkdir -p foo
  cat > foo/testenv.sh <<'EOF'
#!/bin/sh
echo "pwd: $PWD"
echo "src: $TEST_SRCDIR"
echo "ws: $TEST_WORKSPACE"
echo "foo: $FOO"
EOF
  chmod +x foo/testenv.sh
  cat > foo/BUILD <<EOF
sh_test(
    name = "foo",
    srcs = ["testenv.sh"],
    envs = {"FOO", "bar"},
)
EOF

  bazel test --test_output=all //foo &> $TEST_log || fail "Test failed"
  expect_log "pwd: .*/foo.runfiles/bar$"
  expect_log "src: .*/foo.runfiles$"
  expect_log "ws: bar$"
  expect_log "foo: bar"
}

function test_run_under_label_with_options() {
  mkdir -p testing run || fail "mkdir testing run failed"
  cat <<EOF > run/BUILD
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

# This test uses "--nomaster_bazelrc" since outside .bazelrc files can pollute
# this environment and cause --experimental_ui to be turned on, which causes
# this test to fail. Just "--bazelrc=/dev/null" is not sufficient to fix.
function test_run_under_path() {
  mkdir -p testing || fail "mkdir testing failed"
  echo "sh_test(name='t1', srcs=['t1.sh'])" > testing/BUILD
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
  PATH=$PATH:$PWD/scripts bazel --nomaster_bazelrc test //testing:t1 -s --run_under=hello \
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

function test_test_timeout() {
  mkdir -p dir

  cat <<EOF > dir/test.sh
#!/bin/sh
sleep 3
exit 0

EOF

  chmod +x dir/test.sh

  cat <<EOF > dir/BUILD
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
sh_test(name = "test$i", srcs = [ "test$i.sh" ])
EOF
    bazel test --spawn_strategy=standalone --jobs=1 \
        --runs_per_test=5 --runs_per_test_detects_flakes \
        //:test$i &> $TEST_log || fail "should have succeeded"
    expect_log "FLAKY"
  done
}

# Tests that the test.xml is extracted from the sandbox correctly.
function test_xml_is_present() {
  mkdir -p dir

  cat <<'EOF' > dir/test.sh
#!/bin/sh
echo HELLO > $XML_OUTPUT_FILE
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
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

  cat <<'EOF' > dir/test.sh
#!/bin/bash
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
sh_test(
    name = "test",
    srcs = [ "test.sh" ],
  )
EOF
}

function test_xml_is_present_when_timingout() {
  write_test_xml_timeout_files
  bazel test -s --test_timeout=1 --nocache_test_results \
     --noexperimental_split_xml_generation \
     //dir:test &> $TEST_log && fail "should have failed" || true

  xml_log=bazel-testlogs/dir/test/test.xml
  [[ -s "${xml_log}" ]] || fail "${xml_log} was not present after test"
  cat "${xml_log}" > $TEST_log
  expect_log '"Timed out"'
  expect_log '<system-out>'
  # "xmltest" is the first line of output from the test.sh script.
  expect_log '<!\[CDATA\[xmltest'
  expect_log 'before ????? after'
  expect_log '<!CDATA\[\]\]>\]\]<!\[CDATA\[>\]\]>'
  expect_log '</system-out>'
}

function test_xml_is_present_when_timingout_split_xml() {
  write_test_xml_timeout_files
  bazel test -s --test_timeout=1 --nocache_test_results \
     --experimental_split_xml_generation \
     //dir:test &> $TEST_log && fail "should have failed" || true

  xml_log=bazel-testlogs/dir/test/test.xml
  [[ -s "${xml_log}" ]] || fail "${xml_log} was not present after test"
  cat "${xml_log}" > $TEST_log
  # The new script does not convert exit codes to signals.
  expect_log '"exited with error code 142"'
  expect_log '<system-out>'
  # When using --noexperimental_split_xml_generation, the output of the
  # subprocesses goes into the xml file, while
  # --experimental_split_xml_generation inlines the entire test log into
  # the xml file, which includes a header generated by test-setup.sh;
  # the header starts with "exec ${PAGER:-/usr/bin/less}".
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
  mkdir -p dir

  cat <<'EOF' > dir/test.sh
echo "Pretending to sleep..."
sleep 600 &
echo "Finished!" >&2
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
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
  mkdir -p dir

  cat <<EOF > dir/test.sh
#!/bin/sh
exit \$((TEST_SHARD_INDEX == 1))
EOF

  chmod +x dir/test.sh

  cat <<EOF > dir/BUILD
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

function test_detailed_test_summary() {
  copy_examples
  cat > WORKSPACE <<EOF
workspace(name = "io_bazel")
EOF
  setup_javatest_support

  local java_native_tests=//examples/java-native/src/test/java/com/example/myproject

  bazel test --test_summary=detailed "${java_native_tests}:fail" >& $TEST_log \
    && fail "Test $* succeed while expecting failure" \
    || true
  expect_log 'FAILED.*com\.example\.myproject\.Fail\.testFail'
}

# This test uses "--nomaster_bazelrc" since outside .bazelrc files can pollute
# this environment and cause --experimental_ui to be turned on, which causes
# this test to fail. Just "--bazelrc=/dev/null" is not sufficient to fix.
function test_flaky_test() {
  cat >BUILD <<EOF
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

  # We do not use sandboxing so we can trick to be deterministically flaky
  # TODO(b/37617303): make test UI-independent
  bazel --nomaster_bazelrc test --noexperimental_ui --spawn_strategy=standalone //:flaky &> $TEST_log \
      || fail "//:flaky should have passed with flaky support"
  [ -f "${FLAKE_FILE}" ] || fail "Flaky test should have created the flake-file!"

  expect_log_once "FAIL: //:flaky (.*/flaky/test_attempts/attempt_1.log)"
  expect_log_once "PASS: //:flaky"
  expect_log_once "FLAKY"
  cat bazel-testlogs/flaky/test_attempts/attempt_1.log &> $TEST_log
  assert_equals "fail" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"
  assert_equals 1 $(ls bazel-testlogs/flaky/test_attempts/*.log | wc -l)
  cat bazel-testlogs/flaky/test.log &> $TEST_log
  assert_equals "pass" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"

  # TODO(b/37617303): make test UI-independent
  bazel --nomaster_bazelrc test --noexperimental_ui //:pass &> $TEST_log \
      || fail "//:pass should have passed"
  expect_log_once "PASS: //:pass"
  expect_log_once PASSED
  [ ! -d bazel-test_logs/pass/test_attempts ] \
    || fail "Got test attempts while expected non for non-flaky tests"
  cat bazel-testlogs/flaky/test.log &> $TEST_log
  assert_equals "pass" "$(tail -1 bazel-testlogs/flaky/test.log)"

  # TODO(b/37617303): make test UI-independent
  bazel --nomaster_bazelrc test --noexperimental_ui //:fail &> $TEST_log \
      && fail "//:fail should have failed" \
      || true
  expect_log_n "FAIL: //:fail (.*/fail/test_attempts/attempt_..log)" 2
  expect_log_once "FAIL: //:fail (.*/fail/test.log)"
  expect_log_once "FAILED"
  cat bazel-testlogs/fail/test_attempts/attempt_1.log &> $TEST_log
  assert_equals "fail" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"
  assert_equals 2 $(ls bazel-testlogs/fail/test_attempts/*.log | wc -l)
  cat bazel-testlogs/fail/test.log &> $TEST_log
  assert_equals "fail" "$(awk "NR == $(wc -l < $TEST_log)" $TEST_log)"
}

function test_undeclared_outputs_are_zipped_and_manifest_exists() {
  mkdir -p dir

  cat <<'EOF' > dir/test.sh
#!/bin/sh
echo "some text" > "$TEST_UNDECLARED_OUTPUTS_DIR/text.txt"
echo "<!DOCTYPE html>" > "$TEST_UNDECLARED_OUTPUTS_DIR/fake.html"
echo "pass"
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
sh_test(
    name = "test",
    srcs = [ "test.sh" ],
  )
EOF

  bazel test -s //dir:test &> $TEST_log || fail "expected success"

  # Newlines are useful around diffs. This helps us get them in bash strings.
  N=$'\n'

  # Check that the undeclared outputs zip file exists.
  outputs_zip=bazel-testlogs/dir/test/test.outputs/outputs.zip
  [ -s $outputs_zip ] || fail "$outputs_zip was not present after test"

  # Check the contents of the zip file.
  unzip -q "$outputs_zip" -d unzipped_outputs || fail "failed to unzip $outputs_zip"
  cat > expected_text <<EOF
some text
EOF
diff "unzipped_outputs/text.txt" expected_text > d || fail "unzipped_outputs/text.txt differs from expected:$N$(cat d)$N"
  cat > expected_html <<EOF
<!DOCTYPE html>
EOF
diff expected_html "unzipped_outputs/fake.html" > d || fail "unzipped_outputs/fake.html differs from expected:$N$(cat d)$N"

  # Check that the undeclared outputs manifest exists and that it has the
  # correct contents.
  outputs_manifest=bazel-testlogs/dir/test/test.outputs_manifest/MANIFEST
  [ -s $outputs_manifest ] || fail "$outputs_manifest was not present after test"
  cat > expected_manifest <<EOF
fake.html	16	text/html
text.txt	10	text/plain
EOF
diff expected_manifest "$outputs_manifest" > d || fail "$outputs_manifest differs from expected:$N$(cat d)$N"
}

function test_undeclared_outputs_annotations_are_added() {
  mkdir -p dir

  cat <<'EOF' > dir/test.sh
#!/bin/sh
echo "an annotation" > "$TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR/1.part"
echo "another annotation" > "$TEST_UNDECLARED_OUTPUTS_ANNOTATIONS_DIR/2.part"
echo "pass"
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
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

  cat <<'EOF' > dir/test.sh
#!/bin/sh
echo "pass"
exit 0
EOF

  chmod +x dir/test.sh

  cat <<'EOF' > dir/BUILD
sh_test(
    name = "test",
    srcs = [ "test.sh" ],
  )
EOF

  bazel test -s //dir:test &> $TEST_log || fail "expected success"

  # Check that the undeclared outputs directory doesn't exist.
  outputs_dir=bazel-testlogs/dir/test/test.outputs/
  [ ! -d $outputs_dir ] || fail "$outputs_dir was present after test"

  # Check that the undeclared outputs manifest directory doesn't exist.
  outputs_manifest_dir=bazel-testlogs/dir/test/test.outputs_manifest/
  [ ! -d $outputs_manifest_dir ] || fail "$outputs_manifest_dir was present after test"
}

function test_test_with_nobuild_runfile_manifests() {
  mkdir -p dir

  touch dir/test.sh
  chmod u+x dir/test.sh
  cat <<'EOF' > dir/BUILD
sh_test(
    name = 'test',
    srcs = ['test.sh'],
)
EOF
  bazel test --nobuild_runfile_manifests //dir:test >& $TEST_log && fail "should have failed"
  expect_log "cannot run local tests with --nobuild_runfile_manifests"
}

run_suite "bazel test tests"
