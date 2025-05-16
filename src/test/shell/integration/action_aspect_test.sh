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
#
# Tests Starlark API pertaining to action inspection via aspect.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

add_to_bazelrc "build --package_path=%workspace%"

function test_directory_args_inspection() {
  mkdir -p package
  cat > package/makes_tree_artifacts.sh <<EOF
#!/usr/bin/env bash
my_dir=\$1

touch \$my_dir/a.txt
touch \$my_dir/b.txt
touch \$my_dir/c.txt
EOF
  chmod 755 package/makes_tree_artifacts.sh

  cat > package/write.sh <<EOF
#!/usr/bin/env bash
output_file=\$1
shift;

echo "\$@" > \$output_file
EOF
  chmod 755 package/write.sh

  cat > package/lib.bzl <<EOF
def _tree_art_impl(ctx):
    my_dir = ctx.actions.declare_directory('dir')
    ctx.actions.run(
        executable = ctx.executable._makes_tree,
        outputs = [my_dir],
        arguments = [my_dir.path])

    digest = ctx.actions.declare_file('digest')
    digest_args = ctx.actions.args()
    digest_args.add(digest.path)
    digest_args.add_all([my_dir])

    ctx.actions.run(executable = ctx.executable._write,
        inputs = [my_dir],
        outputs = [digest],
        arguments = [digest_args])
    return [DefaultInfo(files=depset([digest]))]

def _actions_test_impl(target, ctx):
    action = target.actions[1] # digest action
    aspect_out = ctx.actions.declare_file('aspect_out')
    ctx.actions.run_shell(inputs = action.inputs,
                          outputs = [aspect_out],
                          command = "echo \$@ > " + aspect_out.path,
                          arguments = action.args)
    return [OutputGroupInfo(out=[aspect_out])]

tree_art_rule = rule(implementation = _tree_art_impl,
    attrs = {
        "_makes_tree" : attr.label(allow_single_file = True,
            cfg = "exec",
            executable = True,
            default = "//package:makes_tree_artifacts.sh"),
        "_write" : attr.label(allow_single_file = True,
            cfg = "exec",
            executable = True,
            default = "//package:write.sh")})

actions_test_aspect = aspect(implementation = _actions_test_impl)
EOF

  cat > package/BUILD <<EOF
load(":lib.bzl", "tree_art_rule")

tree_art_rule(name = "x")
EOF

  bazel build package:x || fail "Unexpected build failure"

  cat "${PRODUCT_NAME}-bin/package/digest" | grep "a.txt.*b.txt.*c.txt" \
      || fail "rule digest does not contain tree artifact args"

  bazel build package:x --aspects=//package:lib.bzl%actions_test_aspect \
      --output_groups=out

  cat "${PRODUCT_NAME}-bin/package/aspect_out" | grep "a.txt.*b.txt.*c.txt" \
      || fail "aspect Args do not contain tree artifact args"
}

function test_directory_args_inspection_param_file() {
  mkdir -p package
  cat > package/makes_tree_artifacts.sh <<EOF
#!/usr/bin/env bash
my_dir=\$1

touch \$my_dir/a.txt
touch \$my_dir/b.txt
touch \$my_dir/c.txt
EOF
  chmod 755 package/makes_tree_artifacts.sh

  cat > package/write.sh <<EOF
#!/usr/bin/env bash

param_file=\$1
shift;

output_file="\$(head -n 1 \$param_file)"

tail \$param_file -n +2 | tr '\n' ' ' > "\$output_file"
EOF
  chmod 755 package/write.sh

  cat > package/lib.bzl <<EOF
def _tree_art_impl(ctx):
    my_dir = ctx.actions.declare_directory('dir')
    ctx.actions.run(
        executable = ctx.executable._makes_tree,
        outputs = [my_dir],
        arguments = [my_dir.path])

    digest = ctx.actions.declare_file('digest')
    digest_args = ctx.actions.args()
    digest_args.add(digest.path)
    digest_args.add_all([my_dir])
    digest_args.use_param_file("%s", use_always=True)

    ctx.actions.run(executable = ctx.executable._write,
        inputs = [my_dir],
        outputs = [digest],
        arguments = [digest_args])
    return [DefaultInfo(files=depset([digest]))]

def _actions_test_impl(target, ctx):
    action = target.actions[1] # digest action
    raw_args_out = ctx.actions.declare_file('raw_args_out')
    param_file_out = ctx.actions.declare_file('param_file_out')
    ctx.actions.run_shell(inputs = action.inputs,
                          outputs = [raw_args_out],
                          command = "echo \$@ > " + raw_args_out.path,
                          arguments = action.args)
    ctx.actions.run_shell(inputs = action.inputs,
                          outputs = [param_file_out],
                          command = "cat \$2 > " + param_file_out.path,
                          arguments = action.args)
    return [OutputGroupInfo(out=[raw_args_out, param_file_out])]

tree_art_rule = rule(implementation = _tree_art_impl,
    attrs = {
        "_makes_tree" : attr.label(allow_single_file = True,
            cfg = "exec",
            executable = True,
            default = "//package:makes_tree_artifacts.sh"),
        "_write" : attr.label(allow_single_file = True,
            cfg = "exec",
            executable = True,
            default = "//package:write.sh")})

actions_test_aspect = aspect(implementation = _actions_test_impl)
EOF

  cat > package/BUILD <<EOF
load(":lib.bzl", "tree_art_rule")

tree_art_rule(name = "x")
EOF

  bazel build package:x || fail "Unexpected build failure"

  cat "${PRODUCT_NAME}-bin/package/digest" | grep "a.txt.*b.txt.*c.txt" \
      || fail "rule digest does not contain tree artifact args"

  bazel build package:x --aspects=//package:lib.bzl%actions_test_aspect \
      --output_groups=out

  cat "${PRODUCT_NAME}-bin/package/raw_args_out" | grep ".params" \
      || fail "aspect Args does not contain a params file"

  cat "${PRODUCT_NAME}-bin/package/param_file_out" | tr '\n' ' ' \
      | grep "a.txt.*b.txt.*c.txt" \
      || fail "aspect params file does not contain tree artifact args"
}

# Test that a starlark action shadowing another action is not rerun after blaze
# shutdown.
function test_starlark_action_with_shadowed_action_not_rerun_after_shutdown() {
  local package="a1"

  create_starlark_action_with_shadowed_action_cache_test_files "${package}"

  bazel build "${package}:a" \
      --aspects="//${package}:lib.bzl%actions_test_aspect" \
      --output_groups=out || \
      fail "bazel build should've succeeded"

  cp "bazel-bin/${package}/run_timestamp" "${package}/run_1_timestamp"

  # Test that the Starlark action is not rerun after bazel shutdown if
  # the inputs did not change
  bazel shutdown

  bazel build "${package}:a" \
      --aspects="//${package}:lib.bzl%actions_test_aspect" \
      --output_groups=out || \
      fail "bazel build should've succeeded"

  cp "bazel-bin/${package}/run_timestamp" "${package}/run_2_timestamp"

  diff "${package}/run_1_timestamp" "${package}/run_2_timestamp" \
      || fail "Starlark action should not rerun after bazel shutdown"
}

# Test that a starlark action shadowing another action is rerun if the inputs
# of the shadowed_action change.
function test_starlark_action_rerun_after_shadowed_action_inputs_change() {
  local package="a2"

  create_starlark_action_with_shadowed_action_cache_test_files "${package}"

  bazel build "${package}:a" \
      --aspects="//${package}:lib.bzl%actions_test_aspect" \
      --output_groups=out || \
      fail "bazel build should've succeeded"

  cp "bazel-bin/${package}/run_timestamp" "${package}/run_1_timestamp"

  # Test that the Starlark action would rerun if the inputs of the
  # shadowed action changed
  rm -f "${package}/x.h"
  echo "inline int x() { return 0; }" > "${package}/x.h"

  bazel build "${package}:a" \
      --aspects="//${package}:lib.bzl%actions_test_aspect" \
      --output_groups=out || \
      fail "bazel build should've succeeded"

  cp "bazel-bin/${package}/run_timestamp" "${package}/run_2_timestamp"

  diff "${package}/run_1_timestamp" "${package}/run_2_timestamp" \
      && fail "Starlark action should rerun after shadowed action inputs change" \
      || :
}

function create_starlark_action_with_shadowed_action_cache_test_files() {
  local package="$1"

  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF
def _actions_test_impl(target, ctx):
    compile_action = None

    for action in target.actions:
      if action.mnemonic == "CppCompile":
        compile_action = action

    if not compile_action:
      fail("Couldn't find compile action")

    aspect_out = ctx.actions.declare_file("run_timestamp")
    ctx.actions.run_shell(
        shadowed_action = compile_action,
        mnemonic = "AspectAction",
        outputs = [aspect_out],
        # record the timestamp in output file to validate action rerun
        command = "cat ${package}/x.h > %s && date '+%%s' >> %s" % (
            aspect_out.path,
            aspect_out.path,
        ),
    )

    return [OutputGroupInfo(out = [aspect_out])]

actions_test_aspect = aspect(implementation = _actions_test_impl)
EOF

  echo "inline int x() { return 42; }" > "${package}/x.h"
  cat > "${package}/a.cc" <<EOF
#include "${package}/x.h"

int a() { return x(); }
EOF
  cat > "${package}/BUILD" <<EOF
cc_library(
  name = "x",
  hdrs  = ["x.h"],
)

cc_library(
  name = "a",
  srcs = ["a.cc"],
  deps = [":x"],
)
EOF
}

function test_aspect_requires_aspect_no_action_conflict() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/write.sh" <<EOF
#!/usr/bin/env bash
output_file=\$1
unused_file=\$2

echo 'output' > \$output_file
echo 'unused' > \$unused_file
EOF

  chmod 755 "${package}/write.sh"

  cat > "${package}/defs.bzl" <<EOF
def _aspect_b_impl(target, ctx):
  file = ctx.actions.declare_file('{}_aspect_b_file.txt'.format(target.label.name))
  unused_file = ctx.actions.declare_file('{}.unused'.format(target.label.name))
  args = ctx.actions.args()
  args.add(file.path)
  args.add(unused_file.path)
  ctx.actions.run(
    executable = ctx.executable._write,
    outputs = [file, unused_file],
    arguments = [args],
    # Adding unused_inputs_list just to make this action not shareable
    unused_inputs_list = unused_file,
  )
  return [OutputGroupInfo(aspect_b_out = [file])]

aspect_b = aspect(
  implementation = _aspect_b_impl,
  attrs = {
    "_write": attr.label(
                allow_single_file = True,
                cfg = "exec",
                executable = True,
                default = "//${package}:write.sh")
  }
)

def _aspect_a_impl(target, ctx):
  print(['aspect_a can see file ' +
         f.path.split('/')[-1] for f in target[OutputGroupInfo].aspect_b_out.to_list()])
  files = target[OutputGroupInfo].aspect_b_out.to_list()
  return [OutputGroupInfo(aspect_a_out = files)]

aspect_a = aspect(
  implementation = _aspect_a_impl,
  requires = [aspect_b],
  attr_aspects = ['dep_2'],
)

def _rule_1_impl(ctx):
  files = []
  if ctx.attr.dep_1:
    files += ctx.attr.dep_1[OutputGroupInfo].rule_2_out.to_list()
  if ctx.attr.dep_2:
    files += ctx.attr.dep_2[OutputGroupInfo].aspect_a_out.to_list()
  return [DefaultInfo(files=depset(files))]

rule_1 = rule(
  implementation = _rule_1_impl,
  attrs = {
      'dep_1': attr.label(),
      'dep_2': attr.label(aspects = [aspect_a]),
  }
)

def _rule_2_impl(ctx):
  files = []
  if ctx.attr.dep:
    print([ctx.label.name +
            ' can see file ' +
            f.path.split('/')[-1] for f in ctx.attr.dep[OutputGroupInfo].aspect_b_out.to_list()])
    files += ctx.attr.dep[OutputGroupInfo].aspect_b_out.to_list()
  return [OutputGroupInfo(rule_2_out = files)]

rule_2 = rule(
  implementation = _rule_2_impl,
  attrs = {
    'dep': attr.label(aspects = [aspect_b]),
  }
)
EOF

  cat > "${package}/BUILD" <<EOF
load('//${package}:defs.bzl', 'rule_1', 'rule_2')
exports_files(["write.sh"])
rule_1(
  name = 'main_target',
  dep_1 = ':dep_target_1',
  dep_2 = ':dep_target_2',
)
rule_2(
  name = 'dep_target_1',
  dep = ':dep_target_2',
)
rule_2(name = 'dep_target_2')
EOF

  bazel build "//${package}:main_target" &> $TEST_log || fail "Build failed"

  expect_log "aspect_a can see file dep_target_2_aspect_b_file.txt"
  expect_log "dep_target_1 can see file dep_target_2_aspect_b_file.txt"
}

run_suite "Tests Starlark API pertaining to action inspection via aspect"
