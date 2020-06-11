#!/bin/bash
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
#
# Test //external mechanisms
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }

test_children_waited() {
    # Verify that a successful ctx.execute waits for its child processes.

    # As we might have delayed side-effects, switch to a throw-away directory.
    WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
    cd "${WRKDIR}"

    cat > repo.bzl <<'EOF'
def _impl(ctx):
  ctx.file("BUILD", "exports_files(['data.txt'])")
  ctx.execute(["/bin/sh", "-c", "echo foo > data.txt"])
  ctx.execute(["/bin/sh", "-c", "(sleep 30; echo bar > data.txt) &"])
  ctx.execute(["/bin/sh", "-c", "echo baz > data.txt"])

changing_repo = repository_rule(_impl)
EOF
    cat > WORKSPACE <<'EOF'
load("//:repo.bzl", "changing_repo")

changing_repo(name="change")
EOF
    cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@change//:data.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
    bazel build //:it
    cp `bazel info bazel-genfiles`/it.txt output
    cat output
    grep baz output || fail "repo rule faild to wait for child process"
}

test_interrupted_children_waited() {
    # Verify that an interrupted ctx.execute waits for its child processes.

    # As we might have delayed side-effects, switch to a throw-away directory.
    WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
    cd "${WRKDIR}"

    cat > repo.bzl <<EOF
def _impl(ctx):
  ctx.file("BUILD", "exports_files(['data.txt'])")
  ctx.execute(["/bin/sh", "-c", "echo foo > data.txt"])
  ctx.execute(["/bin/sh", "-c", "(sleep 30; echo after > ${WRKDIR}/side.txt) & wait \$!"])

changing_repo = repository_rule(_impl)
EOF
    cat > WORKSPACE <<'EOF'
load("//:repo.bzl", "changing_repo")

changing_repo(name="change")
EOF
    cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@change//:data.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF
    echo before > side.txt
    bazel build //:it > "${TEST_log}" 2>&1 &
    bazel_pid="$!"
    sleep 15 && kill "${bazel_pid}"
    cp side.txt side1.txt
    sleep 30
    cp side.txt side2.txt
    echo; echo before wait:; echo
    cat side1.txt
    echo; echo after wait:; echo
    cat side2.txt
    echo
    diff side1.txt side2.txt || fail "found delayed side effects"
}

run_suite "Starlark execute tests"
