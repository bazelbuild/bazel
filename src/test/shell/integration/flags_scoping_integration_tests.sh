#!/usr/bin/env bash
#
# Copyright 2024 The Bazel Authors. All rights reserved.
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
# An end-to-end test for flagsets.

# --- begin runfiles.bash initialization ---
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

write_project_scl_definition

function declare_common_provider() {
  local -r pkg="$1"
  mkdir -p "${pkg}"
  cat > "${pkg}/provider.bzl" <<EOF
BuildSettingInfo = provider(fields = ["value"])
EOF
}

function set_up_flags() {
  local -r pkg="$1"

  # Define common starlark flags.
  #mkdir -p "${pkg}"
  declare_common_provider "${pkg}"
  cat > "${pkg}"/flag.bzl <<EOF
load("//${pkg}:provider.bzl", "BuildSettingInfo")

def _sample_flag_impl(ctx):
    print ("Flag value for %s: %s" % (ctx.label.name, ctx.build_setting_value))
    return BuildSettingInfo(value = ctx.build_setting_value)

sample_flag = rule(
    implementation = _sample_flag_impl,
    build_setting = config.string(flag = True),
    attrs = {
      "scope": attr.string(default = "universal"),
    }
)
EOF
}

function create_transitions() {
  local pkg="${1}"
  local project_flag1_location="${2}"
  echo "project_flag1_location: ${project_flag1_location}"
  local project_flag2_location="${3}"
  echo "project_flag2_location: ${project_flag2_location}"
  local universal_flag_location="${4}"
  echo "universal_flag_location: ${universal_flag_location}"

  cat > "${pkg}/def.bzl" <<EOF
project_flag1_location = "${project_flag1_location}"
project_flag2_location = "${project_flag2_location}"
universal_flag_location = "${universal_flag_location}"

def _transition_impl(settings, attr):
    return {
      "//%s:project_flag" % project_flag1_location: "changed_value_project_flag1",
      "//%s:project_flag" % project_flag2_location: "changed_value_project_flag2",
      "//%s:universal_flag" % universal_flag_location: "changed_value_universal_flag"
    }

example_transition = transition(
    implementation = _transition_impl,
    inputs = [],
    outputs = [
      "//%s:project_flag" % project_flag1_location,
      "//%s:project_flag" % project_flag2_location,
      "//%s:universal_flag" % universal_flag_location
    ]
)

def _out_of_scope_transition_impl(settings, attr):
    return {
      "//out_of_scope:project_flag_baseline" : "baseline"
    }

out_of_scope_transition = transition(
    implementation = _out_of_scope_transition_impl,
    inputs = [],
    outputs = [
      "//out_of_scope:project_flag_baseline"
    ]
)

def _rule_impl(ctx):
    print("dummy")

transition_attached = rule(
    implementation = _rule_impl,
    cfg = example_transition,
)

def _aspect_impl(target, ctx):
    print("dummy")
    return []

simple_aspect = aspect(
    implementation = _aspect_impl,
    attr_aspects = [],
)

transition_on_dep = rule(
    implementation = _rule_impl,
    attrs = {
        "dep": attr.label(aspects = [simple_aspect], cfg = out_of_scope_transition),
    },
)

EOF
}

function test_flags_scoping_remove_out_of_scope_flags() {
  echo "testing test_flags_scoped_properly"
  local -r pkg="$FUNCNAME"
  local -r in_scope_flags_pkg="in_scope_flags"
  local -r out_of_scope_flags_pkg="out_of_scope_flags"
  local -r universal_flags_pkg="universal_flags"

  mkdir -p "${pkg}"
  mkdir -p "${in_scope_flags_pkg}"
  mkdir -p "${out_of_scope_flags_pkg}"
  mkdir -p "${universal_flags_pkg}"

  set_up_flags "${pkg}" "${in_scope_flags_pkg}" "${out_of_scope_flags_pkg}" "${universal_flags_pkg}"

# define flags that should be applied to the project
  cat > "${in_scope_flags_pkg}/BUILD" <<EOF
load("//${pkg}:flag.bzl", "sample_flag")

sample_flag(
    name = "project_flag",
    scope = "project",
    build_setting_default = "foo",
)
EOF

# define flags that are put of scope for the project
  #mkdir -p "${out_of_scope_flags_pkg}"
  cat > "${out_of_scope_flags_pkg}/BUILD" <<EOF
load("//${pkg}:flag.bzl", "sample_flag")

sample_flag(
    name = "project_flag",
    scope = "project",
    build_setting_default = "foo",
)

sample_flag(
    name = "project_flag_baseline",
    scope = "project",
    build_setting_default = "foo",
)
EOF

# define universal flag that should propagate everywhere
  cat > "${universal_flags_pkg}/BUILD" <<EOF
load("//${pkg}:flag.bzl", "sample_flag")

sample_flag(
    name = "universal_flag",
    build_setting_default = "foo",
)
EOF

  cat > "${in_scope_flags_pkg}/PROJECT.scl" <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create(project_directories = [ "//${pkg}"])
EOF

  cat > "${out_of_scope_flags_pkg}/PROJECT.scl" <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create(project_directories = [ "//${out_of_scope_flags_pkg}"])
EOF

# create transitions with project and universal flags and rules that attach transitions
  create_transitions "${pkg}" "${in_scope_flags_pkg}" "${out_of_scope_flags_pkg}" "${universal_flags_pkg}"

# targets belonging to the project
  cat > "${pkg}/BUILD" <<EOF
load("//${pkg}:def.bzl", "transition_attached")
transition_attached(
    name = "project_target",
)
EOF
  bazel build //${pkg}:project_target --//${out_of_scope_flags_pkg}:project_flag_baseline=baseline || fail "bazel failed"

  for config in $(bazel config | tail -n +2 | cut -d ' ' -f 1); do
    bazel config "${config}" >> $TEST_log
  done

  cat $TEST_log

  expect_log "//in_scope_flags:project_flag: changed_value_project_flag1"
  expect_log "//universal_flags:universal_flag: changed_value_universal_flag"
  expect_log "//out_of_scope_flags:project_flag_baseline: baseline"
  expect_not_log "//out_of_scope_flags:project_flag: changed_value_project_flag2"
}

function test_flags_scoping_works_with_aspect_on_alias_chain() {
  echo "testing test_flags_scoping_works_with_aspect_on_alias_chain"
  local -r pkg="$FUNCNAME"
  local -r in_scope_pkg="in_scope"
  local -r out_of_scope_pkg="out_of_scope"
  local -r universal_pkg="universal"

  mkdir -p "${pkg}"
  mkdir -p "${in_scope_pkg}"
  mkdir -p "${out_of_scope_pkg}"
  mkdir -p "${universal_pkg}"

  set_up_flags "${pkg}" "${in_scope_pkg}" "${out_of_scope_pkg}" "${universal_pkg}"
  create_transitions "${pkg}" "${in_scope_pkg}" "${out_of_scope_pkg}" "${universal_pkg}"

# set up target that is in scope for the project
  cat > "${pkg}/BUILD" <<EOF
load("//${pkg}:def.bzl", "transition_on_dep")

transition_on_dep(
    name = "main",
    dep = ":in_scope_alias",
)

alias(
    name = "in_scope_alias",
    actual = "//${out_of_scope_pkg}:out_of_scope_alias",
)
EOF

  cat > "${out_of_scope_pkg}/PROJECT.scl" <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create(project_directories = ["//${pkg}"])
EOF

  # set up package with out of scope alias chain and actual target
  cat > "${out_of_scope_pkg}/BUILD" <<EOF
load("//${pkg}:def.bzl", "transition_on_dep")
load("//${pkg}:flag.bzl", "sample_flag")

alias(
    name = "out_of_scope_alias",
    actual = ":actual",
    visibility = ["//visibility:public"],
)

transition_on_dep(
    name = "actual",
)

sample_flag(
    name = "project_flag_baseline",
    scope = "project",
    build_setting_default = "foo",
)

EOF

  bazel build //${pkg}:main || fail "bazel failed"
}

function create_test_rule() {
  local pkg="${1}"
  cat > "${pkg}/defs.bzl" <<EOF
load("//${pkg}:provider.bzl", "BuildSettingInfo")

def _print_flag_value_impl(ctx):
    flag_value = ctx.attr._my_flag[BuildSettingInfo].value
    print("%s: flag value is: %s" % (ctx.label, flag_value))

print_flag_value = rule(
    implementation = _print_flag_value_impl,
    attrs = {
        "_my_flag": attr.label(
            default = "//${pkg}:test",
        ),
        "deps": attr.label_list(
            cfg = "exec",
        ),
        "scope": attr.string(
            doc = "The scope indicates where a flag can propagate to",
            default = "universal",
        ),
    },
)

EOF
}

function test_exec_propagation() {
  local -r pkg="$FUNCNAME"
  mkdir -p "${pkg}"

  set_up_flags "${pkg}"
  create_test_rule "${pkg}"
  cat > "${pkg}/BUILD" <<EOF
load("//${pkg}:defs.bzl", "print_flag_value")
load("//${pkg}:flag.bzl", "sample_flag")

print_flag_value(
    name = "print_flag",
    deps = ["dep"],
)

print_flag_value(
    name = "dep",
)

sample_flag(
    name = "test",
    build_setting_default = "default target value",
    scope = "exec:--//${pkg}:host_test",
)

sample_flag(
    name = "host_test",
    build_setting_default = "default_host_value",
)
EOF
  # --flag_alias=host_test=//${pkg}:host_test is needed for this feature to work
  # because that's how we find potential --foo candidates

  # test for when both host and target flags are not specified on the command line
  bazel build //${pkg}:print_flag --flag_alias=host_test=//${pkg}:host_test &> "$TEST_log" || fail "bazel failed"
  expect_log "${pkg}:dep: flag value is: default_host_value"
  expect_log "${pkg}:print_flag: flag value is: default target value"

  # test for when both host and target flags are specified on the command line
  bazel build //${pkg}:print_flag --//${pkg}:test=custom_target_value --//${pkg}:host_test=custom_host_value --flag_alias=host_test=//${pkg}:host_test &> "$TEST_log" || fail "bazel failed"
  expect_log "${pkg}:dep: flag value is: custom_host_value"
  expect_log "${pkg}:print_flag: flag value is: custom_target_value"

  # test for when only target flag is specified on the command line
  bazel build //${pkg}:print_flag --//${pkg}:test=custom_target_value --flag_alias=host_test=//${pkg}:host_test &> "$TEST_log" || fail "bazel failed"
  expect_log "${pkg}:dep: flag value is: default_host_value"
  expect_log "${pkg}:print_flag: flag value is: custom_target_value"

  # test for when only host flag is specified on the command line
  bazel build //${pkg}:print_flag --//${pkg}:host_test=custom_host_value --flag_alias=host_test=//${pkg}:host_test &> "$TEST_log" || fail "bazel failed"
  expect_log "${pkg}:dep: flag value is: custom_host_value"
  expect_log "${pkg}:print_flag: flag value is: default target value"
}

function test_exec_propagation_does_not_create_ST_hash() {
  echo "testing test_exec_propagation_does_not_create_ST_hash"
    local -r pkg="$FUNCNAME"
  mkdir -p "${pkg}"

  set_up_flags "${pkg}"
  create_test_rule "${pkg}"
  cat > "${pkg}/BUILD" <<EOF
load("//${pkg}:defs.bzl", "print_flag_value")
load("//${pkg}:flag.bzl", "sample_flag")

print_flag_value(
    name = "print_flag",
    deps = ["dep"],
)

print_flag_value(
    name = "dep",
)

sample_flag(
    name = "test",
    build_setting_default = "default target value",
    scope = "exec:--//${pkg}:host_test",
)

sample_flag(
    name = "host_test",
    build_setting_default = "default_host_value",
)
EOF
  # --flag_alias=host_test=//${pkg}:host_test is needed for this feature to work
  # because that's how we find potential --foo candidates

  bazel clean && bazel cquery --noimplicit_deps --flag_alias=host_test=//${pkg}:host_test "deps(//${pkg}:print_flag) intersect //${pkg}:*"
  bazel config &> "$TEST_log" || fail "bazel failed"
  expect_not_log "ST-"
  expect_log "-opt-exec"
}

function test_foobar_exec_transition_foo_baz_keeps_flag() {
  echo "testing test_foobar_exec_transition_foo_baz_keeps_flag"
  local -r pkg="foo"

  mkdir -p "${pkg}"

  # define a project-scoped flag
  cat > "${pkg}/flag.bzl" <<EOF
BuildSettingInfo = provider(fields = ["value"])

def _sample_flag_impl(ctx):
    return BuildSettingInfo(value = ctx.build_setting_value)

sample_flag = rule(
    implementation = _sample_flag_impl,
    build_setting = config.string(flag = True),
    attrs = {
      "scope": attr.string(default = "universal"),
    }
)
EOF

  cat > "${pkg}/BUILD" <<EOF
load("//${pkg}:flag.bzl", "sample_flag")

sample_flag(
    name = "project_flag",
    scope = "project",
    build_setting_default = "default",
)
EOF

  # create PROJECT.scl
  cat > "${pkg}/PROJECT.scl" <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create(project_directories = [ "//${pkg}"])
EOF

  # create rules that use the flag
  cat > "${pkg}/rules.bzl" <<EOF
load("//${pkg}:flag.bzl", "BuildSettingInfo")

def _exec_dep_impl(ctx):
    print("Exec dep flag value: " + ctx.attr._flag[BuildSettingInfo].value)

exec_dep_rule = rule(
    implementation = _exec_dep_impl,
    attrs = {
        "_flag": attr.label(default = "//${pkg}:project_flag"),
    }
)

def _main_rule_impl(ctx):
    pass

def _set_flag_transition_impl(settings, attr):
    return {"//${pkg}:project_flag": "custom_value"}

set_flag_transition = transition(
    implementation = _set_flag_transition_impl,
    inputs = [],
    outputs = ["//${pkg}:project_flag"]
)

main_rule = rule(
    implementation = _main_rule_impl,
    cfg = set_flag_transition,
    attrs = {
        "exec_dep": attr.label(cfg = "exec"),
    }
)
EOF

  cat >> "${pkg}/BUILD" <<EOF
load("//${pkg}:rules.bzl", "main_rule", "exec_dep_rule")

exec_dep_rule(
    name = "exec_target",
)

main_rule(
    name = "main_target",
    exec_dep = ":exec_target",
)
EOF

  bazel build //${pkg}:main_target >& $TEST_log || fail "bazel failed"

  expect_log "Exec dep flag value: custom_value"
}

function test_foobar_exec_transition_otherpackage_baz_drops_flag() {
  echo "testing test_foobar_exec_transition_otherpackage_baz_drops_flag"
  local -r pkg="foo"
  local -r other_pkg="otherpackage"

  mkdir -p "${pkg}"
  mkdir -p "${other_pkg}"

  # define a project-scoped flag
  cat > "${pkg}/flag.bzl" <<EOF
BuildSettingInfo = provider(fields = ["value"])

def _sample_flag_impl(ctx):
    return BuildSettingInfo(value = ctx.build_setting_value)

sample_flag = rule(
    implementation = _sample_flag_impl,
    build_setting = config.string(flag = True),
    attrs = {
      "scope": attr.string(default = "universal"),
    }
)
EOF

  cat > "${pkg}/BUILD" <<EOF
load("//${pkg}:flag.bzl", "sample_flag")

package(default_visibility = ["//visibility:public"])

sample_flag(
    name = "project_flag",
    scope = "project",
    build_setting_default = "default",
)
EOF

  # create PROJECT.scl
  cat > "${pkg}/PROJECT.scl" <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create(project_directories = [ "//${pkg}"])
EOF

  # create rules that use the flag
  cat > "${other_pkg}/rules.bzl" <<EOF
load("//${pkg}:flag.bzl", "BuildSettingInfo")

def _exec_dep_impl(ctx):
    print("Exec dep flag value: " + ctx.attr._flag[BuildSettingInfo].value)

exec_dep_rule = rule(
    implementation = _exec_dep_impl,
    attrs = {
        "_flag": attr.label(default = "//${pkg}:project_flag"),
    }
)

def _main_rule_impl(ctx):
    pass

def _set_flag_transition_impl(settings, attr):
    return {"//${pkg}:project_flag": "custom_value"}

set_flag_transition = transition(
    implementation = _set_flag_transition_impl,
    inputs = [],
    outputs = ["//${pkg}:project_flag"]
)

main_rule = rule(
    implementation = _main_rule_impl,
    cfg = set_flag_transition,
    attrs = {
        "exec_dep": attr.label(cfg = "exec"),
    }
)
EOF

  cat > "${other_pkg}/BUILD" <<EOF
load("//${other_pkg}:rules.bzl", "main_rule", "exec_dep_rule")

exec_dep_rule(
    name = "exec_target",
)

main_rule(
    name = "main_target",
    exec_dep = ":exec_target",
)
EOF

  bazel build //${other_pkg}:main_target >& $TEST_log || fail "bazel failed"

  expect_log "Exec dep flag value: default"
}

function test_project_scope_reset_to_declared_default() {
  local -r pkg="$FUNCNAME"
  mkdir -p "${pkg}/python" "${pkg}/native"
  cat > "${pkg}/defs.bzl" <<EOF
ValueInfo = provider(fields = ["value"])

def _flag(ctx):
    return [ValueInfo(value = ctx.build_setting_value)]

version_flag = rule(implementation = _flag, build_setting = config.string(flag = True), attrs = {
    "scope": attr.string(),
    "scope_reset": attr.string(default = "baseline"),
})

def _probe(ctx):
    value = ctx.attr._version[ValueInfo].value
    if value != ctx.attr.expected:
        fail("%s: expected %s, got %s" % (ctx.label, ctx.attr.expected, value))
    if ctx.attr.tool and ctx.attr.tool[ValueInfo].value != value:
        fail("Exec tool lost selected version: %s" % ctx.attr.tool[ValueInfo].value)
    out = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(out, value)
    return [DefaultInfo(files = depset([out])), ValueInfo(value = value)]

_attrs = {
    "expected": attr.string(),
    "dep": attr.label(),
    "tool": attr.label(cfg = "exec"),
    "_version": attr.label(default = "//${pkg}/python:version"),
}
probe = rule(implementation = _probe, attrs = _attrs)

def _transition(_settings, _attr):
    return {"//${pkg}/python:version": "3.14"}

_transition_cfg = transition(implementation = _transition, inputs = [], outputs = ["//${pkg}/python:version"])
transitioned_probe = rule(implementation = _probe, attrs = _attrs | {
    "_allowlist_function_transition": attr.label(default = "@bazel_tools//tools/allowlists/function_transition_allowlist"),
}, cfg = _transition_cfg)
EOF
  cat > "${pkg}/python/BUILD.bazel" <<EOF
load("//${pkg}:defs.bzl", "version_flag", "probe", "transitioned_probe")
package(default_visibility = ["//visibility:public"])
version_flag(name = "version", build_setting_default = "neutral", scope = "project", scope_reset = "default")
probe(name = "tool311", expected = "3.11")
probe(name = "tool312", expected = "3.12")
probe(name = "tool314", expected = "3.14")
probe(name = "binding311", expected = "3.11", dep = "//${pkg}/native:core", tool = ":tool311")
probe(name = "binding312", expected = "3.12", dep = "//${pkg}/native:core", tool = ":tool312")
transitioned_probe(name = "transitioned", expected = "3.14", dep = "//${pkg}/native:core", tool = ":tool314")
EOF
  cat > "${pkg}/native/BUILD.bazel" <<EOF
load("//${pkg}:defs.bzl", "probe")
probe(name = "core", expected = "neutral", visibility = ["//visibility:public"])
EOF
  cat > "${pkg}/BUILD" <<EOF
exports_files(["defs.bzl"])
EOF
  cat > "${pkg}/python/PROJECT.scl" <<EOF
load("//third_party/bazel/src/main/protobuf/project:project_proto.scl", "project_pb2")
project = project_pb2.Project.create(project_directories = ["//${pkg}/python"])
EOF

  # Bindings and owned exec tools retain CLI and transitioned values; native deps use defaults.
  bazel build //${pkg}/python:binding311 //${pkg}/python:transitioned \
      --//${pkg}/python:version=3.11 > "$TEST_log" 2>&1 || fail "3.11 build failed"
  bazel build //${pkg}/python:binding312 //${pkg}/python:transitioned \
      --//${pkg}/python:version=3.12 > "$TEST_log" 2>&1 || fail "3.12 build failed"

  local first_config second_config default_config
  first_config=$(bazel cquery //${pkg}/native:core --//${pkg}/python:version=3.11 \
      2> "$TEST_log") || fail "first query failed"
  second_config=$(bazel cquery //${pkg}/native:core --//${pkg}/python:version=3.12 \
      2> "$TEST_log") || fail "second query failed"
  default_config=$(bazel cquery //${pkg}/native:core --//${pkg}/python:version=neutral \
      2> "$TEST_log") || fail "default query failed"
  assert_equals "$first_config" "$second_config"
  assert_equals "$first_config" "$default_config"
  local shared_config
  shared_config=$(bazel cquery \
      "filter('native:core', deps(set(//${pkg}/python:binding311 //${pkg}/python:transitioned)))" \
      --//${pkg}/python:version=3.11 2> "$TEST_log") || fail "shared graph query failed"
  assert_equals "$first_config" "$shared_config"

  # Equal configurations must also have stable artifact paths across all three baselines.
  local first_path second_path default_path
  first_path=$(bazel cquery //${pkg}/native:core --output=files --//${pkg}/python:version=3.11 \
      2> "$TEST_log") || fail "first path query failed"
  second_path=$(bazel cquery //${pkg}/native:core --output=files --//${pkg}/python:version=3.12 \
      2> "$TEST_log") || fail "second path query failed"
  default_path=$(bazel cquery //${pkg}/native:core --output=files --//${pkg}/python:version=neutral \
      2> "$TEST_log") || fail "default path query failed"
  assert_equals "$first_path" "$second_path"
  assert_equals "$first_path" "$default_path"
}

run_suite "Integration tests for flags scoping"
