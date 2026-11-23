#!/usr/bin/env bash
#
# Copyright 2026 The Bazel Authors. All rights reserved.
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
# End-to-end tests for Starlark custom args (ctx.actions.args).

# --- begin runfiles.bash initialization v3 ---
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function set_up() {
  # Proactively delete all data which is potentially created by the tests.
  # The correctness of this test suite relies on the list of directories being
  # complete.
  rm -rf foo
  # Make sure there are no leftover build artifacts.
  bazel clean >& "${TEST_log}" || fail "Expected success"
}

function tear_down() {
  bazel --noblock_for_lock shutdown 2> "${TEST_log}" || \
    fail "Failed to shut down ${PRODUCT_NAME} -- did a test leave a lingering command?"
}

function test_benign_change_action_cache_hit() {
  mkdir -p foo
  cat > "foo/def.bzl" <<'EOF'
def _transform(file):
  return file.path.replace("/dir/", "/dir2/")

def _r(ctx):
    input_dir = ctx.actions.declare_directory(ctx.attr.prefix + ctx.attr.suffix)
    ctx.actions.run_shell(
      outputs = [input_dir],
      command = "touch " + input_dir.path + '/file'
    )

    # Create files in 'dir2' based on transforming file names from 'dir'
    dir = ctx.actions.declare_directory("dir2")
    args = ctx.actions.args()
    args.add_all([input_dir], map_each=_transform, expand_directories=True)
    ctx.actions.run_shell(
        mnemonic = "touchOutputs",
        inputs = [input_dir],
        outputs = [dir],
        command = 'touch "$@"',
        arguments = [args],
    )

    return [DefaultInfo(files = depset([dir]))]

r = rule(
  implementation = _r,
  attrs = { "prefix" : attr.string(), "suffix" : attr.string()}
)
EOF

  cat > "foo/BUILD" <<'EOF'
load(":def.bzl", "r")
r(name="a", prefix="d", suffix="ir")
EOF

  bazel build --experimental_ui_debug_all_events --color=no --curses=no \
    //foo:a >& "${TEST_log}" || fail "Expected success"
  expect_log "FINISH .* touchOutputs foo/dir2"

  cat > "foo/BUILD" <<'EOF'
load(":def.bzl", "r")
r(name="a", prefix="di", suffix="r")
EOF

  bazel shutdown >& "${TEST_log}" || fail "Expected success"
  bazel build --experimental_ui_debug_all_events --color=no --curses=no \
    //foo:a >& "${TEST_log}" || fail "Expected success"
  expect_not_log "FINISH .* touchOutputs foo/dir2"
}

function test_expand_directories_map_each_constant_for_directory() {
  mkdir -p foo
  cat > "foo/directory.bzl" <<'EOF'
def on_directory(dir):
    return "constant value"
EOF
  _test_expand_directories_map_each_no_false_sharing
}

function test_expand_directories_map_each_fails_on_directory() {
  mkdir -p foo
  cat > foo/directory.bzl <<'EOF'
def on_directory(dir):
    fail("Should never be called for " + dir.path)
EOF
  _test_expand_directories_map_each_no_false_sharing
}

# Test scenario making sure that a change to the logic of function passed as
# `map_each` for the arguments results with no sharing through the action cache.
# This test is parameterized by the behavior on calling the `map_each` function
# on a directory entry. The test case will call `on_directory` function from
# `foo/directory.bzl` -- this file must be created before calling this helper.
function _test_expand_directories_map_each_no_false_sharing() {
  cat > foo/def.bzl <<'EOF'
load(":transform.bzl", "transform_file_name")
load(":directory.bzl", "on_directory")

def _transform(file):
    if file.path.endswith('dir'):
        return on_directory(file)
    return file.dirname + '2/' + transform_file_name(file.basename)

def _r(ctx):
    # Create a 'dir' directory tree artifact with 'file' in it.
    input_dir = ctx.actions.declare_directory("dir")
    ctx.actions.run_shell(
      outputs = [input_dir],
      command = "touch " + input_dir.path + '/file'
    )

    # Create files in 'dir2' based on transforming file names from 'dir'
    dir = ctx.actions.declare_directory("dir2")
    args = ctx.actions.args()
    args.add_all([input_dir], map_each=_transform, expand_directories=True)
    ctx.actions.run_shell(
        inputs = [input_dir],
        outputs = [dir],
        command = 'touch "$@"',
        arguments = [args],
    )

    return [DefaultInfo(files = depset([dir]))]

r = rule(implementation = _r)
EOF

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "r")
r(name = "a")
EOF

  cat > foo/transform.bzl <<'EOF'
def transform_file_name(name):
  return name
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"

  cat > foo/transform.bzl <<'EOF'
def transform_file_name(name):
  return name + "_suffix"
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "file_suffix" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"
}

function test_expand_directories_map_each_failure_change() {
  mkdir -p foo
  cat > foo/def.bzl <<'EOF'
load(":failure.bzl", "failure")

def _transform(file):
    if file.path.endswith('dir'):
        return fail("Directory")
    return failure()

def _r(ctx):
    # Create a '_dir' directory tree artifact with 'file' in it.
    input_dir = ctx.actions.declare_directory("dir")
    ctx.actions.run_shell(
      outputs = [input_dir],
      command = "touch %s/file" % input_dir.path
    )

    # Dummy output -- we will fail to create the command line anyway.
    dummy_file = ctx.actions.declare_file("dummy_file")
    args = ctx.actions.args()
    args.add_all([input_dir], map_each=_transform, expand_directories=True)
    ctx.actions.run_shell(
        inputs = [input_dir],
        outputs = [dummy_file],
        command = 'echo "$@" > %s' % dummy_file.path,
        arguments = [args],
    )

    return [DefaultInfo(files = depset([dummy_file]))]

r = rule(implementation = _r)
EOF

  cat >> foo/BUILD << EOF
load(":def.bzl", "r")
r(name = "a")
EOF

  cat >> foo/failure.bzl <<'EOF'
def failure():
    fail("First failure")
EOF

  bazel build foo:a >& "${TEST_log}" && fail "Expected build failure"
  expect_log "First failure"

  cat > foo/failure.bzl <<'EOF'
def failure():
    fail("Second failure")
EOF

  bazel build foo:a >& "${TEST_log}" && fail "Expected build failure"
  expect_log "Second failure"
}

function test_expand_directories_same_name_different_module() {
  mkdir -p foo
  cat > foo/def.bzl <<'EOF'
load(":other_rewrite.bzl", other_rewrite="rewrite_path")

def rewrite_path(file):
    return file.path.replace('/dir/', '/dir2/')

def _touch_outs_impl(ctx):
    input_dir = ctx.actions.declare_directory("dir")
    ctx.actions.run_shell(
        outputs = [input_dir],
        command = "touch %s/file" % input_dir.path)
    dir = ctx.actions.declare_directory("dir2")
    args = ctx.actions.args()
    rewrite = other_rewrite if ctx.attr.use_other_rewrite else rewrite_path
    args.add_all([input_dir], map_each=rewrite, expand_directories=True)
    ctx.actions.run_shell(
        arguments = [args],
        inputs = [input_dir],
        outputs = [dir],
        command = 'touch "$@"',
    )

    return [DefaultInfo(files=depset([dir]))]

touch_outputs = rule(
    implementation = _touch_outs_impl,
    attrs = { "use_other_rewrite" : attr.bool(default=False) })
EOF

  cat > foo/other_rewrite.bzl <<'EOF'
def rewrite_path(file):
    return file.path.replace('/dir/', '/dir2/prefix_')
EOF

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "touch_outputs")
touch_outputs(name="a")
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "touch_outputs")
touch_outputs(name="a", use_other_rewrite=True)
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "prefix_file" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"
}

function test_map_each_identical_function_and_module_contents_different_behavior() {
  mkdir -p foo
  cat > foo/f1.bzl <<'EOF'
def transform(fun):
  return "identical" if fun == transform else "different"
EOF

  cp foo/f1.bzl foo/f2.bzl

  cat > foo/def.bzl <<'EOF'
load(":f1.bzl", f1="transform")
load(":f2.bzl", f2="transform")

def _r(ctx):
  transform = f2 if ctx.attr.use_f2 else f1
  file = ctx.actions.declare_file("file")
  args = ctx.actions.args()
  args.add_all([f1], map_each=transform)
  ctx.actions.run_shell(
    outputs = [file],
    command = "echo $@ > " + file.path,
    arguments = [args],
  )
  return [DefaultInfo(files=depset([file]))]

r = rule(implementation=_r, attrs={ "use_f2" : attr.bool(default=False) })
EOF

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "r")
r(name="a")
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "identical" "$(cat "${PRODUCT_NAME}-bin/foo/file")"

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "r")
r(name="a", use_f2=True)
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "different" "$(cat "${PRODUCT_NAME}-bin/foo/file")"
}

function test_map_each_different_builtin_functions() {
  mkdir -p foo
  cat > foo/def.bzl <<'EOF'
def _r(ctx):
  transform = type if ctx.attr.use_type else str
  file = ctx.actions.declare_file("file")
  args = ctx.actions.args()
  args.add_all(["text"], map_each=transform)
  ctx.actions.run_shell(
    outputs = [file],
    command = "echo $@ > " + file.path,
    arguments = [args],
  )
  return [DefaultInfo(files=depset([file]))]

r = rule(implementation=_r, attrs={ "use_type" : attr.bool(default=False) })
EOF

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "r")
r(name="a")
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "text" "$(cat "${PRODUCT_NAME}-bin/foo/file")"

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "r")
r(name="a", use_type=True)
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "string" "$(cat "${PRODUCT_NAME}-bin/foo/file")"
}

function test_expand_directories_dir_becomes_a_file() {
  mkdir -p foo
  cat > foo/def.bzl <<'EOF'
def _rewrite_path(file):
    return file.path.replace('a_dir', 'b_dir')

def _touch_outs_impl(ctx):
    input_dir = ctx.attr.dep[DefaultInfo].files.to_list()[0]
    dir = ctx.actions.declare_directory("%s_dir" % ctx.label.name)
    args = ctx.actions.args()
    args.add_all([input_dir], map_each=_rewrite_path, expand_directories=True)
    ctx.actions.run_shell(
        arguments = [args],
        inputs = [input_dir],
        outputs = [dir],
        command = 'touch "$@"',
    )

    return [DefaultInfo(files=depset([dir]))]

touch_outs = rule(
    implementation=_touch_outs_impl,
    attrs = {'dep' : attr.label()}
)
EOF

  cat > foo/BUILD <<'EOF'
load(":create.bzl", "create")
load(":def.bzl", "touch_outs")

create(name="a")
touch_outs(name="b", dep=":a")
EOF

  cat > foo/create.bzl <<'EOF'
def _create(ctx):
    dir = ctx.actions.declare_directory("%s_dir" % ctx.label.name)
    ctx.actions.run_shell(outputs=[dir], command="touch %s/file" % dir.path)
    return [DefaultInfo(files=depset([dir]))]

create = rule(implementation=_create)
EOF

  bazel build //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(ls -1 "${PRODUCT_NAME}-bin/foo/b_dir")"

  cat > foo/create.bzl <<'EOF'
def _create(ctx):
    file = ctx.actions.declare_file("%s_dir" % ctx.label.name)
    ctx.actions.run_shell(outputs=[file], command="touch %s" % file.path)
    return [DefaultInfo(files=depset([file]))]

create = rule(implementation=_create)
EOF

  bazel build //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "" "$(ls -1 "${PRODUCT_NAME}-bin/foo/b_dir")"
}

function test_args_expand_directories_aspect_over_native_rule() {
  add_rules_shell "MODULE.bazel"
  mkdir -p foo
  _write_aspect_declaring_action_def > foo/def.bzl

  cat > foo/BUILD <<'EOF'
load("@rules_shell//shell:sh_library.bzl", "sh_library")
load(":def.bzl", "touch_outs")

sh_library(name="a")
touch_outs(name="b", deps=[":a"])
EOF

  cat > foo/transform.bzl <<'EOF'
def transform_file_name(name):
  return name
EOF

  bazel build //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"

  cat > foo/transform.bzl <<'EOF'
def transform_file_name(name):
  return name + "_suffix"
EOF

  bazel build //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "file_suffix" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"
}

# Regression test for b/153904017.
function test_expand_directories_starlark_rule_aspect() {
  mkdir -p foo
  _write_aspect_declaring_action_def > foo/def.bzl

  cat > foo/noop.bzl <<'EOF'
def _noop(ctx):
  return []

noop = rule(implementation=_noop)
EOF

  cat > foo/BUILD <<'EOF'
load(":def.bzl", "touch_outs")
load(":noop.bzl", "noop")

noop(name="a")
touch_outs(name="b", deps=[":a"])
EOF

  cat > foo/transform.bzl <<'EOF'
def transform_file_name(name):
  return name
EOF

  bazel build //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"

  cat > foo/transform.bzl <<'EOF'
def transform_file_name(name):
  return name + "_suffix"
EOF

  bazel build //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "file_suffix" "$(ls -1 "${PRODUCT_NAME}-bin/foo/dir2")"
}

function _write_aspect_declaring_action_def {
  cat <<'EOF'
load(":transform.bzl", "transform_file_name")

ExtraInfo = provider(fields = {'dir' : 'dir'})

def _transform_if_not_dir(file):
    if file.path.endswith('dir'):
        return 'constant'
    return file.dirname + '2/' + transform_file_name(file.basename)

def _touch_outs_impl(target, ctx):
    input_dir = ctx.actions.declare_directory("dir")
    ctx.actions.run_shell(
        outputs = [input_dir],
        command = "touch " + input_dir.path + '/file'
    )

    dir = ctx.actions.declare_directory("dir2")
    args = ctx.actions.args()
    args.add_all(
      [input_dir],
      map_each=_transform_if_not_dir,
      expand_directories=True
    )
    ctx.actions.run_shell(
        arguments = [args],
        inputs = [input_dir],
        outputs = [dir],
        command = 'touch "$@"',
    )

    return [ExtraInfo(dir=dir)]

_touch_outs_aspect = aspect(
    implementation = _touch_outs_impl,
    attr_aspects = ['outs'],
)


def _propagate_outs(ctx):
    return [DefaultInfo(files=depset([ctx.attr.deps[0][ExtraInfo].dir]))]

touch_outs = rule(
    implementation=_propagate_outs,
    attrs = {'deps' : attr.label_list(aspects = [_touch_outs_aspect])}
)
EOF
}

function test_map_each_with_expander_change_map_each() {
  mkdir foo
  cat > foo/def.bzl <<'EOF'
load(':transform.bzl', 'transform')
def _get_file(dir, expander):
  files = expander.expand(dir.dir)
  if not files:
    return ''
  return transform(files[0].basename)

def _r(ctx):
  dir = ctx.actions.declare_directory('dir')
  ctx.actions.run_shell(outputs=[dir], command='touch %s/file' % dir.path)

  file = ctx.actions.declare_file('file')
  s = struct(dir=dir)
  args = ctx.actions.args()
  args.add_all([s], map_each=_get_file)
  ctx.actions.run_shell(
    inputs = [dir],
    outputs = [file],
    command = 'echo "$@" > %s' % file.path,
    arguments = [args]
  )
  return DefaultInfo(files=depset([file]))

r = rule(implementation=_r)
EOF

  cat > foo/BUILD <<'EOF'
load('def.bzl', 'r')
r(name='a')
EOF

  cat > foo/transform.bzl <<'EOF'
def transform(path):
  return path
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(cat "${PRODUCT_NAME}-bin/foo/file")"

  cat > foo/transform.bzl <<'EOF'
def transform(path):
  return path + '_suffix'
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "file_suffix" "$(cat "${PRODUCT_NAME}-bin/foo/file")"
}

function test_map_each_nested_set_changed_directory_expansion() {
  mkdir foo
  write_rule_with_directory_in_nestedset > foo/def.bzl

  cat > foo/BUILD <<'EOF'
load(':def.bzl', 'r')
r(name='a')
EOF

  cat > foo/transform.bzl <<'EOF'
def transform(path):
  return path
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(cat "${PRODUCT_NAME}-bin/foo/file")"

  cat > foo/transform.bzl <<'EOF'
def transform(path):
  return path + '_suffix'
EOF

  bazel build //foo:a >& "${TEST_log}" || fail "Expected success"
  assert_equals "file_suffix" "$(cat "${PRODUCT_NAME}-bin/foo/file")"
}

function test_map_each_nested_set_changed_directory_expansion_shared_action() {
  mkdir foo
  write_rule_with_directory_in_nestedset > foo/def.bzl

  cat > foo/BUILD <<'EOF'
load(':def.bzl', 'r')
r(name='a')
r(name='b')
EOF

  cat > foo/transform.bzl <<'EOF'
def transform(path):
  return path
EOF

  bazel build //foo:a //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "file" "$(cat "${PRODUCT_NAME}-bin/foo/file")"

  cat > foo/transform.bzl <<'EOF'
def transform(path):
  return path + '_suffix'
EOF

  bazel build //foo:a //foo:b >& "${TEST_log}" || fail "Expected success"
  assert_equals "file_suffix" "$(cat "${PRODUCT_NAME}-bin/foo/file")"
}

function write_rule_with_directory_in_nestedset() {
  cat <<'EOF'
load(':transform.bzl', 'transform')

def _f(file):
  if file.is_directory:
    return 'dir'
  return transform(file.basename)

def _r(ctx):
  dir = ctx.actions.declare_directory('dir')
  dir2 = ctx.actions.declare_directory('dir2')
  ctx.actions.run_shell(
    outputs = [dir, dir2],
    command = 'touch %s/file' % dir.path
  )

  file = ctx.actions.declare_file('file')
  args = ctx.actions.args()
  # 2 elements in the nestedset to avoid the case of inlined singleton.
  args.add_all(depset([dir, dir2]), map_each=_f)
  ctx.actions.run_shell(
    inputs = [dir, dir2],
    outputs = [file],
    command = 'echo "$@" > %s' % file.path,
    arguments = [args]
  )
  return DefaultInfo(files=depset([file]))

r = rule(implementation=_r, attrs = {'files' : attr.string_list()})
EOF
}

function test_expand_directory_missing_from_inputs() {
  mkdir foo
  cat > foo/create_dir.bzl <<'EOF'
def _create_dir(ctx):
  dir = ctx.actions.declare_directory("dir_" + ctx.label.name)
  ctx.actions.run_shell(outputs=[dir], command="touch %s/file" % dir.path)
  return DefaultInfo(files=depset([dir]))
create_dir = rule(implementation=_create_dir)
EOF
  define_rule_listing_files_missing_input > foo/def.bzl
  cat > foo/BUILD <<'EOF'
load(":def.bzl", "list_files_skip_dep_from_inputs")
load(":create_dir.bzl", "create_dir")
create_dir(name="a")
list_files_skip_dep_from_inputs(name="b", dep=":a", expand_directories=True)
EOF

  local exit_code=0
  bazel build //foo:b &> "${TEST_log}" || exit_code=$?

  assert_equals 1 "${exit_code}"
  expect_log "Failed to expand directory <generated file foo/dir_a>"
}

function test_noexpand_directory_missing_from_inputs() {
  mkdir foo
  cat > foo/create_dir.bzl <<'EOF'
def _create_dir(ctx):
  dir = ctx.actions.declare_directory("dir_" + ctx.label.name)
  ctx.actions.run_shell(outputs=[dir], command="touch %s/file" % dir.path)
  return DefaultInfo(files=depset([dir]))
create_dir = rule(implementation=_create_dir)
EOF
  define_rule_listing_files_missing_input > foo/def.bzl
  cat > foo/BUILD <<'EOF'
load(":def.bzl", "list_files_skip_dep_from_inputs")
load(":create_dir.bzl", "create_dir")
create_dir(name="a")
list_files_skip_dep_from_inputs(name="b", dep=":a", expand_directories=False)
EOF

  bazel build //foo:b &> "${TEST_log}" || fail "Expected build to succeed"

  assert_contains \
      "^${PRODUCT_NAME}-out/[^/][^/]*/bin/foo/dir_a$" \
      "${PRODUCT_NAME}-bin/foo/list"
}

function define_rule_listing_files_missing_input() {
  cat <<'EOF'
def _list_files_skip_dep_from_inputs(ctx):
  file = ctx.actions.declare_file("list")
  args = ctx.actions.args()
  args.add_joined(
    ctx.files.dep,
    join_with = ", ",
    expand_directories=ctx.attr.expand_directories
  )
  ctx.actions.run_shell(
    outputs = [file],
    command = "echo $1 > %s" % file.path,
    arguments = [args]
  )
  return DefaultInfo(files=depset([file]))

list_files_skip_dep_from_inputs = rule(
  implementation = _list_files_skip_dep_from_inputs,
  attrs = {
    "dep": attr.label(allow_files=True),
    "expand_directories": attr.bool(mandatory=True)
  }
)
EOF
}

run_suite "starlark custom args tests"
