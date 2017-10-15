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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

add_to_bazelrc "test --nocache_test_results"

# End of preamble.

function create_test_files() {
  # We use this directory as a communication mechanism between test runs. Each
  # test adds a unique file to the directory and then removes it.
  mkdir -p $TEST_TMPDIR/testfiles

  mkdir dir

  cat <<EOF > dir/test.sh
#!/bin/sh

z=\$(mktemp $TEST_TMPDIR/testfiles/tmp.XXXXXXXX)

# Try to ensure other test runs have started too.
sleep 1

numtestfiles=\$(ls -1 $TEST_TMPDIR/testfiles/ | wc -l)

# The tests below are configured to prevent more than 3 tests from running at
# once. This block returns an error code from this script if it observes more
# than 3 files in the testfiles/ directory.
if [[ "\${numtestfiles}" -gt 3 ]] ; then
  echo "Found \${numtestfiles} test files, but there should be 3 at max."
  exit 1
fi

# Try to ensure that we don't remove the test file before other runs have a
# chance to inspect the file.
sleep 1

rm \${z}

EOF

  chmod +x dir/test.sh

  cat <<EOF > dir/BUILD
sh_test(
  name = "test",
  srcs = [ "test.sh" ],
  size = "small",
  tags = [ "local" ],
)
EOF
}

function test_local_test_jobs_constrains_test_execution() {
  create_test_files
  # 3 local test jobs, so no more than 3 tests in parallel.
  bazel test --local_test_jobs=3 --local_resources=10000,10,100 --runs_per_test=10 \
      //dir:test >& $TEST_log || fail "Expected success"
}

function test_no_local_test_jobs_causes_local_resources_to_constrain_test_execution() {
  create_test_files
  # unlimited local test jobs, so local resources enforces 3 tests in parallel.
  bazel test --local_resources=10000,3,100 --runs_per_test=10 \
      //dir:test >& $TEST_log || fail "Expected success"
}

function test_local_test_jobs_exceeds_jobs_causes_warning() {
  create_test_files
  # 10 local test jobs, but only 3 jobs, so warning is printed, and only 3 tests run concurrently
  bazel test --jobs=3 --local_test_jobs=10 --local_resources=10000,10,100 --runs_per_test=10 \
      //dir:test >& $TEST_log || fail "Expected success"

  expect_log 'High value for --local_test_jobs'
}

function test_negative_local_test_jobs_causes_warning() {
  create_test_files
  bazel test --local_test_jobs=-1 --local_resources=10000,10,100 --runs_per_test=10 \
      //dir:test >& $TEST_log && fail "Expected test failure"

  expect_log 'Invalid parameter for --local_test_jobs'
}

run_suite "Tests for --local_test_jobs option."
