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

set -eu

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


function test_passing_test_is_reported_correctly() {
  mkdir -p tests
  cat >tests/BUILD <<'EOF'
sh_test(
    name = "success",
    size = "small",
    srcs = ["success.sh"],
)
EOF
  cat >tests/success.sh <<'EOF'
#!/bin/sh

echo "success.sh is successful"
exit 0
EOF
  chmod +x tests/success.sh

  bazel test --nocache_test_results //tests:success &>$TEST_log \
      || fail "expected success"
  expect_not_log "success.sh is successful"
  expect_log "^Executed 1 out of 1 test: 1 test passes."
}

function test_failing_test_is_reported_correctly() {
  mkdir -p tests
  cat >tests/BUILD <<'EOF'
sh_test(
    name = "fail",
    size = "small",
    srcs = ["fail.sh"],
)
EOF
  cat >tests/fail.sh <<'EOF'
#!/bin/sh

echo "fail.sh is failing"
exit 42
EOF
  chmod +x tests/fail.sh

  bazel test --nocache_test_results //tests:fail &>$TEST_log \
      && fail "expected failure" || true
  expect_log "^//tests:fail[[:space:]]\+FAILED in [[:digit:]]\+\.[[:digit:]]\+s"
  expect_log "^Executed 1 out of 1 test: 1 fails"
}

function test_build_fail_terse_summary() {
    mkdir -p tests
    cat > tests/BUILD <<'EOF'
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
    bazel test --test_summary=terse //tests/... &>$TEST_log \
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
  mkdir -p dir

  cat > dir/BUILD <<'EOF'
java_test(
    name = "my_test",
    main_class = "MyTest",
    srcs = ["MyTest.java"],
    timeout = "short",
    use_testrunner = 0,
)
EOF
  cat > dir/MyTest.java <<'EOF'
public class MyTest {
  public static void main(String[] args) throws Exception {
    new ProcessBuilder("sleep", "300").inheritIO().start();
  }
}
EOF

  bazel test //dir:my_test &> $TEST_log || fail "expected test to pass"
}

function test_test_suite_non_expansion() {
  mkdir -p dir
  cat > dir/BUILD <<'EOF'
sh_test(name = 'test_a',
        srcs = [':a.sh'],
)

sh_test(name = 'test_b',
        srcs = [':b.sh'],
)

test_suite(name = 'suite',
)
EOF
  cat > dir/a.sh <<'EOF'
#!/bin/sh
exit 0
EOF

  cat > dir/b.sh <<'EOF'
#!/bin/sh
exit 0
EOF
  chmod +x dir/a.sh dir/b.sh
  bazel test --noexpand_test_suites //dir:suite &> $TEST_log || fail "expected test to pass"
  expect_log '//dir:test_a'
  expect_log '//dir:test_b'
}
run_suite "test tests"
