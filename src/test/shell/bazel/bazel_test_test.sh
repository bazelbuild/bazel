#!/bin/bash
#
# Copyright 2015 Google Inc. All rights reserved.
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
# Test parallel jobs limitation
#

set -eu

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function set_up() {
  tmp=${TEST_TMPDIR}/testjobs
  # Cleanup the tempory directory
  rm -fr ${tmp}
  mkdir -p ${tmp}

  # We use hardlinks to this file as a communication mechanism between
  # test runs.
  touch ${tmp}/counter

  mkdir -p dir

  cat <<EOF > dir/test.sh
#!/bin/bash
# hard link
z=\$(mktemp -u ${tmp}/tmp.XXXXXXXX)
ln ${tmp}/counter \${z}

# Make sure other test runs have started too.
sleep 1
nlink=\$(ls -l ${tmp}/counter | awk '{print \$2}')

# 4 links = 3 jobs + ${tmp}/counter
if [[ "\$nlink" -gt 4 ]] ; then
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

function test_3_cpus() {
  # 3 CPUs, so no more than 3 tests in parallel.
  bazel test --test_output=errors --local_resources=10000,3,100  --runs_per_test=10 //dir:test
}

function test_3_local_jobs() {
  # 3 local test jobs, so no more than 3 tests in parallel.
  bazel test --test_output=errors --local_test_jobs=3 --local_resources=10000,10,100 --runs_per_test=10 //dir:test
}

function test_unlimited_local_jobs() {
  # unlimited local test jobs, so local resources enforces 3 tests in parallel.
  bazel test --test_output=errors --local_resources=10000,3,100 --runs_per_test=10 //dir:test
}

function test_tmpdir() {
  mkdir -p foo
  cat > foo/bar_test.sh <<EOF
#!/bin/bash
echo "I'm a test"
EOF
  chmod +x foo/bar_test.sh
  cat > foo/BUILD <<EOF
sh_test(
    name = "bar_test",
    srcs = ["bar_test.sh"],
)
EOF
  bazel test --test_output=all -s //foo:bar_test >& $TEST_log || \
    fail "Running sh_test failed"
  expect_log "TEST_TMPDIR=/.*"

  cat > foo/bar_test.sh <<EOF
#!/bin/bash
echo "I'm a different test"
EOF
  bazel test --test_output=all --test_tmpdir=$TEST_TMPDIR -s //foo:bar_test \
    >& $TEST_log || fail "Running sh_test failed"
  expect_log "TEST_TMPDIR=$TEST_TMPDIR"
}

run_suite "test tests"
