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
    grep baz output || fail "repo rule failed to wait for child process"
}

test_interrupted_children_waited() {
    # Verify that an interrupted ctx.execute waits for its child processes.

    # As we might have delayed side-effects, switch to a throw-away directory.
    WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
    cd "${WRKDIR}"

    mkfifo fifo

    cat > waiter.sh <<EOF
#!/bin/sh

cd "${WRKDIR}"

echo \$\$ > repo_pid
echo started > status
cat fifo > /dev/null  # Signal that repo_pid is written
sleep 30  # This gets interrupted early if the test case passes
echo finished > status
EOF
    chmod +x waiter.sh

    cat > repo.bzl <<EOF
def _impl(ctx):
  ctx.file("BUILD", "exports_files(['data.txt'])")
  ctx.execute(["${WRKDIR}/waiter.sh"])

waiting_repo = repository_rule(_impl)
EOF

    cat > WORKSPACE <<'EOF'
load("//:repo.bzl", "waiting_repo")

waiting_repo(name="wait")
EOF

    cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@wait//:data.txt"],
  outs = ["it.txt"],
  cmd = "exit 1",
)
EOF
    bazel build --nobuild //:it > "$TEST_log" 2>&1 &
    bazel_pid="$!"
    echo start > fifo  # Wait until repo_pid is written
    repo_pid="$(cat repo_pid)"

    kill "${bazel_pid}"
    wait "${bazel_pid}" && fail "Bazel should have been interrupted"
    if kill -0 "${repo_pid}"; then
      kill -9 "${repo_pid}"  # Let's not leave it alive
      fail "repo process still running"
    fi

    grep -sq started "${WRKDIR}/status" || fail "repo ran to completion"
}

run_suite "Starlark execute tests"
