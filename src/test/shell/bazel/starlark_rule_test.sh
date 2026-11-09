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

function setup_map_each_file_digests() {
  mkdir -p digests
  echo same > digests/a.txt
  echo same > digests/b.txt
  echo different > digests/c.txt

  cat > digests/defs.bzl <<'EOF'
_DIGEST_PREFIX = "digest="

def _digest(file, expander):
    digest = expander.digest(file)
    if digest == "<DIGEST>":
        return "digest=<DIGEST>"
    return _DIGEST_PREFIX + digest

def _bare_digest(file, expander):
    return expander.digest(file)

def _expand(file, expander):
    return [child.path for child in expander.expand(file)]

def _first_digest(files, expander):
    return expander.digest(files[0])

def _second_digest(files, expander):
    return expander.digest(files[1])

def _manifest_impl(ctx):
    output = ctx.actions.declare_file(ctx.label.name + ".out")
    srcs = ctx.files.srcs
    mode = ctx.attr.mode
    args = ctx.actions.args()
    manifest_args = [args]
    if mode == "conflict":
        conflicting_args = ctx.actions.args()
        if srcs[0].is_directory:
            # Without tree metadata, analysis skips both callbacks.
            args.add_all(srcs, map_each = _digest)
            conflicting_args.add_all(srcs, map_each = _bare_digest)
        else:
            # Both callbacks return <DIGEST> during analysis.
            pairs = depset([(srcs[0], srcs[1]), (srcs[1], srcs[0])])
            args.add_all(pairs, map_each = _first_digest)
            conflicting_args.add_all(pairs, map_each = _second_digest)
        manifest_args.append(conflicting_args)
    elif ctx.attr.join_srcs:
        args.add_joined(ctx.attr.join_srcs[DefaultInfo].files, join_with = ",", map_each = _digest)
        args.use_param_file("%s", use_always = True)
        args.set_param_file_format("multiline")
    elif mode == "prune_expand":
        args.add_all(srcs, map_each = _expand, expand_directories = False)
    else:
        args.add_all(srcs, map_each = _digest)

    unused_inputs = None
    outputs = [output]
    if mode in ("prune", "prune_expand"):
        unused_inputs = ctx.actions.declare_file(ctx.label.name + ".unused")
        outputs.append(unused_inputs)
    command = """
output="$1"
unused="$2"
shift 2
printf '%s\\n' "$@" > "$output"
if [ -n "$unused" ]; then
  : > "$unused"
fi
"""
    if ctx.attr.join_srcs:
        command = "cat \"$3\" > \"$1\""
    arguments = ["-c", command, "", output.path, unused_inputs.path if unused_inputs else ""]
    for action_args in manifest_args:
        ctx.actions.run(
            inputs = srcs,
            outputs = outputs,
            executable = "/bin/sh",
            arguments = arguments + [action_args],
            unused_inputs_list = unused_inputs,
        )
    return DefaultInfo(files = depset([output]))

manifest = rule(
    implementation = _manifest_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "join_srcs": attr.label(),
        "mode": attr.string(default = "digest", values = ["digest", "conflict", "prune", "prune_expand"]),
    },
)

def _tree_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name + ".dir")
    ctx.actions.run_shell(
        inputs = ctx.files.srcs,
        outputs = [output],
        arguments = [output.path] + [f.path for f in ctx.files.srcs],
        command = """
output="$1"
shift
mkdir -p "$output"
cp "$@" "$output"
""",
    )
    return DefaultInfo(files = depset([output]))

tree = rule(
    implementation = _tree_impl,
    attrs = {"srcs": attr.label_list(allow_files = True)},
)
EOF

  cat > digests/BUILD <<'EOF'
load(":defs.bzl", "manifest", "tree")

SRCS = ["a.txt", "b.txt", "c.txt"]
filegroup(name = "sources", srcs = SRCS)
tree(name = "directory", srcs = SRCS)

manifest(name = "source", srcs = SRCS)
manifest(name = "tree", srcs = [":directory"])
manifest(name = "joined", srcs = [":sources"], join_srcs = ":sources")
# Build :joined first. Both actions use the :sources depset in their Args,
# and only :joined declares those files as action inputs.
manifest(
    name = "undeclared",
    srcs = [":joined"],
    join_srcs = ":sources",
)
manifest(name = "conflicting", srcs = ["a.txt", "c.txt"], mode = "conflict")
manifest(name = "conflicting_tree", srcs = [":directory"], mode = "conflict")
manifest(name = "pruning", srcs = ["a.txt"], mode = "prune")
manifest(name = "pruning_expand", srcs = [":directory"], mode = "prune_expand")
EOF
}

function assert_digest_outputs() {
  local output_dir="${PRODUCT_NAME}-bin/digests"
  assert_equals "$(printf '%s\n' "$@")" "$(cat "$output_dir/source.out")"
  assert_equals "$1,$2,$3" "$(cat "$output_dir/joined.out")"
}

function test_map_each_file_digests() {
  setup_map_each_file_digests

  bazel aquery --output=textproto --include_commandline //digests:source \
    &> "$TEST_log" || fail "aquery failed"
  expect_log_n 'arguments: "digest=<DIGEST>"$' 3

  bazel build //digests:source //digests:joined //digests:tree \
    &> "$TEST_log" || fail "build failed"

  local output_dir="${PRODUCT_NAME}-bin/digests"
  local digest_a digest_b digest_c
  {
    read -r digest_a
    read -r digest_b
    read -r digest_c
  } < "$output_dir/source.out"
  [[ -n "$digest_a" && -n "$digest_b" && -n "$digest_c" ]] || fail "empty digest"
  assert_equals "$digest_a" "$digest_b"
  assert_not_equals "$digest_a" "$digest_c"
  assert_digest_outputs "$digest_a" "$digest_b" "$digest_c"
  assert_equals "$(cat "$output_dir/source.out")" "$(cat "$output_dir/tree.out")"

  echo changed > digests/a.txt
  bazel build //digests:source //digests:joined \
    &> "$TEST_log" || fail "build failed after changing a source file"

  local changed_digest
  read -r changed_digest < "$output_dir/source.out"
  [[ -n "$changed_digest" ]] || fail "empty digest after changing a source file"
  assert_not_equals "$digest_a" "$changed_digest"
  assert_digest_outputs "$changed_digest" "$digest_b" "$digest_c"

  # Change the expanded arguments without changing any input files or aquery arguments.
  sed 's/^_DIGEST_PREFIX = "digest="$/_DIGEST_PREFIX = "updated="/' \
    digests/defs.bzl > digests/defs.bzl.new
  mv digests/defs.bzl.new digests/defs.bzl
  bazel build //digests:source //digests:joined \
    &> "$TEST_log" || fail "build failed after changing map_each"
  assert_digest_outputs "updated=${changed_digest#digest=}" \
    "updated=${digest_b#digest=}" "updated=${digest_c#digest=}"
}

function test_map_each_file_digests_rejects_undeclared_source() {
  setup_map_each_file_digests
  bazel build //digests:undeclared &> "$TEST_log" && fail "expected build failure"
  expect_log 'Only declared regular file inputs have a digest: <source file digests/a\.txt>$'
}

function test_map_each_file_digests_reports_action_conflict() {
  setup_map_each_file_digests
  local target
  for target in conflicting conflicting_tree; do
    bazel build --nobuild "//digests:$target" &> "$TEST_log" \
      && fail "expected an action conflict for $target"
    expect_log "$target\.out' is generated by these conflicting actions"
  done
}

function test_map_each_file_digests_rejects_unused_inputs_list() {
  setup_map_each_file_digests
  bazel build //digests:pruning &> "$TEST_log" && fail "expected build failure"
  expect_log 'DirectoryExpander\.digest() is unavailable for actions that discover inputs, including unused_inputs_list$'
}

function test_map_each_directory_expansion_with_unused_inputs_list() {
  setup_map_each_file_digests
  bazel build //digests:pruning_expand &> "$TEST_log" || fail "build failed"
  assert_equals $'a.txt\nb.txt\nc.txt' \
    "$(sed 's@.*/@@' "${PRODUCT_NAME}-bin/digests/pruning_expand.out")"
}

run_suite "Starlark rule definition tests"
