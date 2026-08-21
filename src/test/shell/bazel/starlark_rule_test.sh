#!/usr/bin/env bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
# Tests building with rules defined in Starlark.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Test a basic Starlark rule which touches an output file
function test_basic_output() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":starlark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt"
)
EOF

  cat << 'EOF' >> test/starlark.bzl
def _test_impl(ctx):
  ctx.actions.run_shell(outputs = [ctx.outputs.out],
                        command = "touch " + ctx.outputs.out.path)
  files_to_build = depset([ctx.outputs.out])
  return DefaultInfo(
      files = files_to_build,
  )

test_rule = rule(
    implementation=_test_impl,
    attrs = {
        "out": attr.output(mandatory = True),
    },
)
EOF

  bazel build //test:test &> $TEST_log \
      || fail "should have generated output successfully"
}

# Test a basic Starlark rule which is valid except the action fails on execution.
function test_execution_failure() {
  mkdir -p test
  cat << EOF >> test/BUILD
load(":starlark.bzl", "test_rule")

test_rule(
    name = "test",
    out = "output.txt"
)
EOF

  cat << 'EOF' >> test/starlark.bzl
def _test_impl(ctx):
  ctx.actions.run_shell(outputs = [ctx.outputs.out],
                        command = "not_a_command")
  files_to_build = depset([ctx.outputs.out])
  return DefaultInfo(
      files = files_to_build,
  )

test_rule = rule(
    implementation=_test_impl,
    attrs = {
        "out": attr.output(mandatory = True),
    },
)
EOF

  ! bazel build //test:test &> $TEST_log \
      || fail "Should have resulted in an execution error"

  expect_log "error executing Action command.*not_a_command"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/13189
function test_execute_tool_from_root_package() {
  cat >BUILD <<'EOF'
load("foo.bzl", "foo")

foo(
    name = "x",
    out = "x.out",
    tool = "bin.sh",
)

genrule(
    name = "y",
    outs = ["y.out"],
    cmd = "$(location bin.sh) $@",
    tools = ["bin.sh"],
)
EOF

  cat >foo.bzl << 'EOF'
def _impl(ctx):
    ctx.actions.run(
        outputs = [ctx.outputs.out],
        executable = ctx.executable.tool,
        arguments = [ctx.outputs.out.path],
    )
    return [DefaultInfo(files = depset([ctx.outputs.out]))]

foo = rule(
    implementation = _impl,
    attrs = {
        "tool": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "out": attr.output(mandatory = True),
    },
)
EOF

  cat >bin.sh <<'EOF'
#!/usr/bin/env bash
echo hello $0 > $1
EOF
  chmod +x bin.sh

  # //:x would fail without the bugfix of https://github.com/bazelbuild/bazel/issues/13189
  bazel build //:x &> $TEST_log || fail "Expected success"
  bazel build //:y &> $TEST_log || fail "Expected success"
  rm BUILD bin.sh foo.bzl
}

# Writes a package with a rule that captures the stdout of a tool via the
# `stdout` parameter of ctx.actions.run and a rule that consumes the captured
# output in a downstream action.
function setup_stdout_capture() {
  mkdir -p pkg
  cat > pkg/defs.bzl <<'EOF'
def _capture_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.name + ".out")
    ctx.actions.run(
        outputs = [],
        executable = ctx.executable.tool,
        arguments = [ctx.attr.text],
        stdout = out,
        mnemonic = "Capture",
    )
    return [DefaultInfo(files = depset([out]))]

capture = rule(
    implementation = _capture_impl,
    attrs = {
        "text": attr.string(mandatory = True),
        "tool": attr.label(
            allow_single_file = True,
            executable = True,
            cfg = "exec",
        ),
    },
)

def _consume_impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.name + ".copy")
    ctx.actions.run_shell(
        inputs = ctx.files.src,
        outputs = [out],
        command = "cp \"$1\" \"$2\"",
        arguments = [ctx.files.src[0].path, out.path],
        mnemonic = "Consume",
    )
    return [DefaultInfo(files = depset([out]))]

consume = rule(
    implementation = _consume_impl,
    attrs = {"src": attr.label(allow_files = True, mandatory = True)},
)
EOF

  cat > pkg/echo_tool.sh <<'EOF'
#!/usr/bin/env bash
printf '%s' "$1"
EOF
  chmod +x pkg/echo_tool.sh

  cat > pkg/BUILD <<'EOF'
load(":defs.bzl", "capture", "consume")

capture(
    name = "captured",
    text = "hello-from-stdout",
    tool = "echo_tool.sh",
)

consume(
    name = "consumed",
    src = ":captured",
)
EOF
}

function do_test_stdout_capture() {
  setup_stdout_capture
  bazel build //pkg:captured "$@" &> $TEST_log || fail "build failed"
  assert_equals "hello-from-stdout" "$(cat bazel-bin/pkg/captured.out)"
  # The captured stdout is not reported as regular action output.
  expect_not_log "hello-from-stdout"
}

function test_actions_run_stdout_capture() {
  do_test_stdout_capture
}

function test_actions_run_stdout_capture_local() {
  do_test_stdout_capture --spawn_strategy=local
}

function test_actions_run_stdout_capture_sandboxed() {
  do_test_stdout_capture --spawn_strategy=sandboxed
}

function test_actions_run_stdout_output_consumed_by_downstream_action() {
  setup_stdout_capture
  bazel build //pkg:consumed &> $TEST_log || fail "build failed"
  assert_equals "hello-from-stdout" "$(cat bazel-bin/pkg/consumed.copy)"
}

run_suite "Starlark rule definition tests"
