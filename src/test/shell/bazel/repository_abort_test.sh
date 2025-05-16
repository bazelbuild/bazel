#!/usr/bin/env bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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

test_other_fail() {
  # Test that if one external repository fails in a non-keep-going build
  # the other is correctly reported as aborted rather than failed.
  cat > MODULE.bazel <<'EOF'
slowrepo = use_repo_rule("//:slowrepo.bzl", "slowrepo")
failfast = use_repo_rule("//:fastfailure.bzl", "failfast")

slowrepo(name="slow")
failfast(name="failfast")
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "slow",
  srcs = ["@slow//:data.txt"],
  outs = ["slow.txt"],
  cmd = "cp $< $@",
)

genrule(
  name = "fail",
  srcs = ["@failfast//:data.txt"],
  outs = ["fail.txt"],
  cmd = "cp $< $@",
)
EOF
  # Define a repository where fetching takes really long, so
  # that the failure in fetching the other repo certainly comes
  # before the fetch is done; hence we expect the fetch of this
  # repository to be aborted.
  cat > slowrepo.bzl <<'EOF'
def _impl(ctx):
  st = ctx.execute(["/bin/sh", "-c", "sleep 10 && echo 42 > data.txt"])
  ctx.file("BUILD", "exports_files(['data.txt'])")

slowrepo = repository_rule(
  implementation = _impl,
  attrs = {},
)
EOF
  # Add a repository that just fails, but gives bazel enough time
  # to initiate the fetch of the other repository first. We expect
  # this repository to fail.
  cat > fastfailure.bzl <<'EOF'
def _impl(ctx):
  ctx.execute(["sleep", "2"])
  fail("This is the failure message")

failfast = repository_rule(
  implementation = _impl,
  attrs = {},
)
EOF

  bazel build //:slow //:fail > "${TEST_log}" 2>&1 \
      && fail "should not build" || :

  expect_log 'error.*failfast'
  expect_log 'This is the failure message'
  expect_not_log 'slowrepo'
  expect_not_log '@slow'

  bazel build //:slow > "${TEST_log}" 2>&1 \
    || fail "Should build now"

  cat `bazel info bazel-genfiles 2>/dev/null`/slow.txt | grep -q 42 \
      || fail "Not the expected output"
}

test_timeout() {
  # Verify that timeouts can still be handled by repository rules
  cat > MODULE.bazel <<'EOF'
timeoutrepo = use_repo_rule("//:handletimeout.bzl", "timeoutrepo")

timeoutrepo(name="ext")
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@ext//:data.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
  cat > handletimeout.bzl <<'EOF'
def _impl(ctx):
  ctx.file("BUILD", "exports_files(['data.txt'])")
  st = ctx.execute(["sleep", "90"], timeout=1)
  print ("Exit code: %s" % (st.return_code,))
  if st.return_code == 0:
    fail("Expected to be informed about the timeout")
  else:
    ctx.file("data.txt", "42")

timeoutrepo = repository_rule(
  implementation = _impl,
  attrs = {},
)
EOF

  bazel build //:it || fail "Expected success"
}

run_suite "repository rules abort behavior test"
