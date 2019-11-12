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

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

add_to_bazelrc "build --package_path=%workspace%"

function test_directory_args_inspection() {
  mkdir -p package
  cat > package/makes_tree_artifacts.sh <<EOF
#!/bin/bash
my_dir=\$1

touch \$my_dir/a.txt
touch \$my_dir/b.txt
touch \$my_dir/c.txt
EOF
  chmod 755 package/makes_tree_artifacts.sh

  cat > package/write.sh <<EOF
#!/bin/bash
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
            cfg = "host",
            executable = True,
            default = "//package:makes_tree_artifacts.sh"),
        "_write" : attr.label(allow_single_file = True,
            cfg = "host",
            executable = True,
            default = "//package:write.sh")})

actions_test_aspect = aspect(implementation = _actions_test_impl)
EOF

  cat > package/BUILD <<EOF
load(":lib.bzl", "tree_art_rule")

tree_art_rule(name = "x")
EOF

  bazel build package:x --experimental_action_args \
      || fail "Unexpected build failure"

  cat "${PRODUCT_NAME}-bin/package/digest" | grep "a.txt.*b.txt.*c.txt" \
      || fail "rule digest does not contain tree artifact args"

  bazel build package:x --aspects=//package:lib.bzl%actions_test_aspect \
      --output_groups=out --experimental_action_args

  cat "${PRODUCT_NAME}-bin/package/aspect_out" | grep "a.txt.*b.txt.*c.txt" \
      || fail "aspect Args do not contain tree artifact args"
}

function test_directory_args_inspection_param_file() {
  mkdir -p package
  cat > package/makes_tree_artifacts.sh <<EOF
#!/bin/bash
my_dir=\$1

touch \$my_dir/a.txt
touch \$my_dir/b.txt
touch \$my_dir/c.txt
EOF
  chmod 755 package/makes_tree_artifacts.sh

  cat > package/write.sh <<EOF
#!/bin/bash

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
            cfg = "host",
            executable = True,
            default = "//package:makes_tree_artifacts.sh"),
        "_write" : attr.label(allow_single_file = True,
            cfg = "host",
            executable = True,
            default = "//package:write.sh")})

actions_test_aspect = aspect(implementation = _actions_test_impl)
EOF

  cat > package/BUILD <<EOF
load(":lib.bzl", "tree_art_rule")

tree_art_rule(name = "x")
EOF

  bazel build package:x --experimental_action_args \
      || fail "Unexpected build failure"

  cat "${PRODUCT_NAME}-bin/package/digest" | grep "a.txt.*b.txt.*c.txt" \
      || fail "rule digest does not contain tree artifact args"

  bazel build package:x --aspects=//package:lib.bzl%actions_test_aspect \
      --output_groups=out --experimental_action_args

  cat "${PRODUCT_NAME}-bin/package/raw_args_out" | grep ".params" \
      || fail "aspect Args does not contain a params file"

  cat "${PRODUCT_NAME}-bin/package/param_file_out" | tr '\n' ' ' \
      | grep "a.txt.*b.txt.*c.txt" \
      || fail "aspect params file does not contain tree artifact args"
}

run_suite "Tests Starlark API pertaining to action inspection via aspect"
