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
  declare -r EXE_EXT=".exe"
else
  declare -r EXE_EXT=""
fi

# Tests in this file do not actually start a Python interpreter, but plug in a
# fake stub executable to serve as the "interpreter".

use_fake_python_runtimes_for_testsuite

#### TESTS #############################################################

# Tests that a cycle reached via a command-line aspect does not crash.
# Does not crash deterministically, because if the configured target's cycle is
# reported first, the aspect loading key's cycle is suppressed.
function test_cycle_under_command_line_aspect() {
  mkdir -p test
  cat > test/aspect.bzl << 'EOF' || fail "Couldn't write aspect.bzl"
def _simple_aspect_impl(target, ctx):
    return struct()

simple_aspect = aspect(implementation=_simple_aspect_impl)
EOF
  echo "sh_library(name = 'cycletarget', deps = [':cycletarget'])" \
      > test/BUILD || fail "Couldn't write BUILD file"

  # No flag, use the default from the rule.
  bazel build --nobuild -k //test:cycletarget \
      --aspects 'test/aspect.bzl%simple_aspect' &> $TEST_log \
      && fail "Expected failure"
  local -r exit_code="$?"
  [[ "$exit_code" == 1 ]] || fail "Unexpected exit code: $exit_code"
  expect_log "cycle in dependency graph"
  expect_log "//test:cycletarget \(.*\) \[self-edge\]"
  expect_not_log "IllegalStateException"
}

# When a target is broken, report an error in the target, not the aspect.
# This test would be flaky if errors were non-deterministically reported during
# target and aspect analysis, and would fail outright if aspect failures were
# preferred.
function test_aspect_on_target_with_analysis_failure() {
  mkdir -p test
  cat > test/aspect.bzl << 'EOF' || fail "Couldn't write aspect.bzl"
def _simple_aspect_impl(target, ctx):
    return struct()

simple_aspect = aspect(implementation=_simple_aspect_impl)
EOF
  echo "sh_library(name = 'brokentarget', deps = [':missing'])" \
      > test/BUILD || fail "Couldn't write BUILD file"

  bazel build //test:brokentarget \
      --aspects 'test/aspect.bzl%simple_aspect' &> $TEST_log \
      && fail "Expected failure"
  local -r exit_code="$?"
  [[ "$exit_code" == 1 ]] || fail "Unexpected exit code: $exit_code"
  expect_log "Analysis of target '//test:brokentarget' failed"
  expect_not_log "Analysis of aspects"
}

function test_aspect_required_providers_with_toplevel_aspects() {
  local package="a"
  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF
prov_a = provider()
prov_b = provider()
prov_c = provider()

def _aspect_a_impl(target, ctx):
  print("aspect_a runs on target {}".format(target.label))
  return []

def _aspect_b_impl(target, ctx):
  print("aspect_b runs on target {}".format(target.label))
  return []

aspect_a = aspect(implementation = _aspect_a_impl,
                  attr_aspects = ['deps'],
                  required_providers = [prov_a])
aspect_b = aspect(implementation = _aspect_b_impl,
                  attr_aspects = ['deps'],
                  required_providers = [prov_b, prov_c])

def _rule_with_a_impl(ctx):
  return [prov_a()]

def _rule_with_bc_impl(ctx):
  return [prov_b(), prov_c()]

rule_with_a = rule(implementation = _rule_with_a_impl,
                    attrs = {'deps': attr.label_list()},
                    provides = [prov_a])

rule_with_bc = rule(implementation = _rule_with_bc_impl,
                   attrs = {'deps': attr.label_list()},
                   provides = [prov_b, prov_c])
EOF

  cat > "${package}/BUILD" <<EOF
load('//${package}:lib.bzl', 'rule_with_a', 'rule_with_bc')
rule_with_a(
  name = 'target_with_a',
  deps = [':target_with_a_indeps', ':target_with_bc']
)

rule_with_bc(name = 'target_with_bc')

rule_with_a(name = 'target_with_a_indeps')
EOF

  # without using --incompatible_top_level_aspects_require_providers, aspect_a
  # and aspect_b should attempt to run on top level target: target_with_a and
  # propagate to its dependencies where they will run based on the dependencies
  # advertised providers.
  bazel build "${package}:target_with_a" \
        --aspects="//${package}:lib.bzl%aspect_a" \
        --aspects="//${package}:lib.bzl%aspect_b" &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # Only aspect_a is applied on target_with_a because its "provided" providers
  # do not macth aspect_b required providers.
  expect_log "aspect_a runs on target @//${package}:target_with_a"
  expect_not_log "aspect_b runs on target @//${package}:target_with_a"

  # Only aspect_a can run on target_with_a_indeps
  expect_log "aspect_a runs on target @//${package}:target_with_a_indeps"
  expect_not_log "aspect_b runs on target @//${package}:target_with_a_indeps"

  # Only aspect_b can run on target_with_bc
  expect_not_log "aspect_a runs on target @//${package}:target_with_bc"
  expect_log "aspect_b runs on target @//${package}:target_with_bc"

  # using --incompatible_top_level_aspects_require_providers, the top level
  # target rule's advertised providers will be checked and only aspect_a will be
  # applied on target_with_a and propagated to its dependencies.
  bazel build "${package}:target_with_a" \
        --aspects="//${package}:lib.bzl%aspect_a" \
        --aspects="//${package}:lib.bzl%aspect_b" &>"$TEST_log" \
        --incompatible_top_level_aspects_require_providers \
      || fail "Build failed but should have succeeded"

  # Only aspect_a is applied on target_with_a
  expect_log "aspect_a runs on target @//${package}:target_with_a"
  expect_not_log "aspect_b runs on target @//${package}:target_with_a"

  # Only aspect_a can run on target_with_a_indeps
  expect_log "aspect_a runs on target @//${package}:target_with_a_indeps"
  expect_not_log "aspect_b runs on target @//${package}:target_with_a_indeps"

  # rule_with_bc advertised provides only match the required providers for
  # aspect_b, but aspect_b is not propagated from target_with_a
  expect_not_log "aspect_a runs on target @//${package}:target_with_bc"
  expect_not_log "aspect_b runs on target @//${package}:target_with_bc"
}

function test_aspect_required_providers_default_no_required_providers() {
  local package="a"
  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF
prov_a = provider()
prov_b = provider()

def _my_aspect_impl(target, ctx):
  print("my_aspect runs on target {}".format(target.label))
  return []

my_aspect = aspect(implementation = _my_aspect_impl,
                   attr_aspects = ['deps'])

def _rule_without_providers_impl(ctx):
  pass

def _rule_with_providers_impl(ctx):
  return [prov_a(), prov_b()]

rule_without_providers = rule(implementation = _rule_without_providers_impl,
                              attrs = {'deps': attr.label_list(aspects = [my_aspect])})

rule_with_providers = rule(implementation = _rule_with_providers_impl,
                           attrs = {'deps': attr.label_list()},
                           provides = [prov_a, prov_b])

rule_with_providers_not_advertised = rule(implementation = _rule_with_providers_impl,
                                          attrs = {'deps': attr.label_list()},
                                          provides = [])
EOF

  cat > "${package}/BUILD" <<EOF
load('//${package}:lib.bzl', 'rule_with_providers', 'rule_without_providers',
                             'rule_with_providers_not_advertised')
rule_without_providers(
  name = 'main',
  deps = [':target_without_providers', ':target_with_providers',
          ':target_with_providers_not_advertised']
)

rule_without_providers(name = 'target_without_providers')

rule_with_providers(name = 'target_with_providers')

rule_with_providers(name = 'target_with_providers_indeps')

rule_with_providers_not_advertised(
  name = 'target_with_providers_not_advertised',
  deps = [':target_with_providers_indeps'],
)
EOF

  bazel build "${package}:main" &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # my_aspect does not require any providers so it will be applied to all the
  # dependencies of main target
  expect_log "my_aspect runs on target @//${package}:target_without_providers"
  expect_log "my_aspect runs on target @//${package}:target_with_providers"
  expect_log "my_aspect runs on target @//${package}:target_with_providers_not_advertised"
  expect_log "my_aspect runs on target @//${package}:target_with_providers_indeps"
}

function test_aspect_required_providers_flat_set_of_required_providers() {
  local package="a"
  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF
prov_a = provider()
prov_b = provider()

def _my_aspect_impl(target, ctx):
  print("my_aspect runs on target {}".format(target.label))
  return []

my_aspect = aspect(implementation = _my_aspect_impl,
                   attr_aspects = ['deps'],
                   required_providers = [prov_a, prov_b])

def _rule_without_providers_impl(ctx):
  pass

def _rule_with_a_impl(ctx):
  return [prov_a()]

def _rule_with_ab_impl(ctx):
  return [prov_a(), prov_b()]

rule_without_providers = rule(implementation = _rule_without_providers_impl,
                              attrs = {'deps': attr.label_list(aspects = [my_aspect])})

rule_with_a = rule(implementation = _rule_with_a_impl,
                           attrs = {'deps': attr.label_list()},
                           provides = [prov_a])

rule_with_ab = rule(implementation = _rule_with_ab_impl,
                           attrs = {'deps': attr.label_list()},
                           provides = [prov_a, prov_b])

rule_with_ab_not_advertised = rule(implementation = _rule_with_ab_impl,
                                          attrs = {'deps': attr.label_list()},
                                          provides = [])

EOF

  cat > "${package}/BUILD" <<EOF
load('//${package}:lib.bzl', 'rule_without_providers', 'rule_with_a',
                             'rule_with_ab', 'rule_with_ab_not_advertised')

rule_without_providers(
  name = 'main',
  deps = [':target_without_providers', ':target_with_a', ':target_with_ab',
          ':target_with_ab_not_advertised'],
)

rule_without_providers(name = 'target_without_providers')

rule_with_a(
  name = 'target_with_a',
  deps = [':target_with_ab_indeps_not_reached'],
)

rule_with_ab(
  name = 'target_with_ab',
  deps = [':target_with_ab_indeps_reached']
)

rule_with_ab(name = 'target_with_ab_indeps_not_reached')

rule_with_ab(name = 'target_with_ab_indeps_reached')

rule_with_ab_not_advertised(name = 'target_with_ab_not_advertised')

EOF

  bazel build "${package}:main" &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # my_aspect will only be applied on target_with_ab and
  # target_with_ab_indeps_reached since their rule (rule_with_ab) is the only
  # rule that advertises the aspect required providers.
  expect_log "my_aspect runs on target @//${package}:target_with_ab"
  expect_log "my_aspect runs on target @//${package}:target_with_ab_indeps_reached"
  expect_not_log "/^my_aspect runs on target @//${package}:target_with_a$/"
  expect_not_log "my_aspect runs on target @//${package}:target_without_providers"
  expect_not_log "my_aspect runs on target @//${package}:target_with_ab_not_advertised"

  # my_aspect cannot be propagated to target_with_ab_indeps_not_reached
  # because it was not applied to its parent (target_with_a)
  expect_not_log "my_aspect runs on target @//${package}:target_with_ab_indeps_not_reached"
}

function test_aspect_required_providers_with_list_of_required_providers_lists() {
  local package="a"
  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF
prov_a = provider()
prov_b = provider()
prov_c = provider()

def _my_aspect_impl(target, ctx):
  print("my_aspect runs on target {}".format(target.label))
  return []

my_aspect = aspect(implementation = _my_aspect_impl,
                   attr_aspects = ['deps'],
                   required_providers = [[prov_a, prov_b], [prov_c]])

def _rule_without_providers_impl(ctx):
  pass

rule_without_providers = rule(implementation = _rule_without_providers_impl,
                              attrs = {'deps': attr.label_list(aspects=[my_aspect])},
                              provides = [])

def _rule_with_a_impl(ctx):
  return [prov_a()]

rule_with_a = rule(implementation = _rule_with_a_impl,
                   attrs = {'deps': attr.label_list()},
                   provides = [prov_a])

def _rule_with_c_impl(ctx):
  return [prov_c()]

rule_with_c = rule(implementation = _rule_with_c_impl,
                   attrs = {'deps': attr.label_list()},
                   provides = [prov_c])

def _rule_with_ab_impl(ctx):
  return [prov_a(), prov_b()]

rule_with_ab = rule(implementation = _rule_with_ab_impl,
                    attrs = {'deps': attr.label_list()},
                    provides = [prov_a, prov_b])

rule_with_ab_not_advertised = rule(implementation = _rule_with_ab_impl,
                                   attrs = {'deps': attr.label_list()},
                                   provides = [])

EOF

  cat > "${package}/BUILD" <<EOF
load('//${package}:lib.bzl', 'rule_without_providers', 'rule_with_a',
                             'rule_with_c', 'rule_with_ab',
                             'rule_with_ab_not_advertised')

rule_without_providers(
  name = 'main',
  deps = [':target_without_providers', ':target_with_a', ':target_with_c',
          ':target_with_ab', 'target_with_ab_not_advertised'],
)

rule_without_providers(name = 'target_without_providers')

rule_with_a(
  name = 'target_with_a',
  deps = [':target_with_c_indeps_not_reached'],
)

rule_with_c(name = 'target_with_c')

rule_with_c(name = 'target_with_c_indeps_reached')

rule_with_c(name = 'target_with_c_indeps_not_reached')

rule_with_ab(
  name = 'target_with_ab',
  deps = [':target_with_c_indeps_reached']
)

rule_with_ab_not_advertised(name = 'target_with_ab_not_advertised')

EOF

  bazel build "${package}:main" &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # my_aspect will only be applied on target_with_ab, target_wtih_c and
  # target_with_c_indeps_reached because their rules (rule_with_ab and
  # rule_with_c) are the only rules advertising the aspect required providers
  expect_log "my_aspect runs on target @//${package}:target_with_ab"
  expect_log "my_aspect runs on target @//${package}:target_with_c"
  expect_log "my_aspect runs on target @//${package}:target_with_c_indeps_reached"
  expect_not_log "my_aspect runs on target @//${package}:target_without_providers"
  expect_not_log "/^my_aspect runs on target @//${package}:target_with_a$/"
  expect_not_log "my_aspect runs on target @//${package}:target_with_ab_not_advertised"

  # my_aspect cannot be propagated to target_with_c_indeps_not_reached because it was
  # not applied to its parent (target_with_a)
  expect_not_log "my_aspect runs on target @//${package}:target_with_c_indeps_not_reached"
}

function test_aspects_propagating_other_aspects() {
  local package="a"
  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF
prov_a = provider()

def _required_aspect_impl(target, ctx):
  print("required_aspect runs on target {}".format(target.label))
  return []

required_aspect = aspect(implementation = _required_aspect_impl,
                         required_providers = [prov_a])

def _base_aspect_impl(target, ctx):
  print("base_aspect runs on target {}".format(target.label))
  return []

base_aspect = aspect(implementation = _base_aspect_impl,
                     attr_aspects = ['dep'],
                     requires = [required_aspect])

def _rule_impl(ctx):
  pass

base_rule = rule(implementation = _rule_impl,
                 attrs = {'dep': attr.label(aspects=[base_aspect])})

dep_rule_without_providers = rule(implementation = _rule_impl,
                                  attrs = {'dep': attr.label()})

def _dep_rule_with_prov_a_impl(ctx):
  return [prov_a()]

dep_rule_with_prov_a = rule(implementation = _dep_rule_with_prov_a_impl,
                            provides = [prov_a])

EOF

  cat > "${package}/BUILD" <<EOF
load('//${package}:lib.bzl', 'base_rule', 'dep_rule_without_providers',
                             'dep_rule_with_prov_a')

base_rule(
  name = 'main',
  dep = ':dep_target_without_providers',
)

dep_rule_without_providers(
  name = 'dep_target_without_providers',
  dep = ':dep_target_without_providers_1'
)

dep_rule_without_providers(
  name = 'dep_target_without_providers_1',
  dep = ':dep_target_with_prov_a'
)

dep_rule_with_prov_a(
  name = 'dep_target_with_prov_a',
)

EOF

  bazel build "${package}:main" &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # base_aspect will run on dep_target_without_providers,
  # dep_target_without_providers_1 and dep_target_with_prov_a but
  # required_aspect will only run on dep_target_with_prov_a because
  # it satisfies its required providers
  expect_log "base_aspect runs on target @//${package}:dep_target_with_prov_a"
  expect_log "base_aspect runs on target @//${package}:dep_target_without_providers_1"
  expect_log "base_aspect runs on target @//${package}:dep_target_without_providers"
  expect_log "required_aspect runs on target @//${package}:dep_target_with_prov_a"
  expect_not_log "required_aspect runs on target @//${package}:dep_target_without_providers_1"
  expect_not_log "required_aspect runs on target @//${package}:dep_target_without_providers/"
}

function test_aspects_propagating_other_aspects_stack_of_required_aspects() {
  local package="pkg"
  mkdir -p "${package}"

  cat > "${package}/lib.bzl" <<EOF

def _aspect_a_impl(target, ctx):
    print("Aspect 'a' applied on target: {}".format(target.label))
    return []

def _aspect_b_impl(target, ctx):
    print("Aspect 'b' applied on target: {}".format(target.label))
    return []

def _aspect_c_impl(target, ctx):
    print("Aspect 'c' applied on target: {}".format(target.label))
    return []

def r_impl(ctx):
  return []

aspect_c = aspect(implementation = _aspect_c_impl,
                  attr_aspects = ["deps"])

aspect_b = aspect(implementation = _aspect_b_impl,
                  attr_aspects = ["deps"], requires = [aspect_c])

aspect_a = aspect(implementation = _aspect_a_impl,
                  attr_aspects = ["deps"], requires = [aspect_b])

rule_r = rule(
    implementation = r_impl,
    attrs = {
        "deps": attr.label_list(aspects = [aspect_a]),
    }
)

EOF

echo "inline int x() { return 42; }" > "${package}/x.h"
  cat > "${package}/t.cc" <<EOF
#include "${package}/x.h"

int a() { return x(); }
EOF
  cat > "${package}/BUILD" <<EOF
load("//${package}:lib.bzl", "rule_r")

cc_library(
  name = "x",
  hdrs  = ["x.h"],
)

cc_library(
  name = "t",
  srcs = ["t.cc"],
  deps = [":x"],
)

rule_r(
  name = "test",
  deps = [":t"],
)
EOF

  bazel build "${package}:test" &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # Check that aspects: aspect_a, aspect_b, aspect_c were propagated to the
  # dependencies of target test: t and x when only aspect_a is specified
  # in the rule definition
  expect_log_once "Aspect 'c' applied on target: @//${package}:x"
  expect_log_once "Aspect 'c' applied on target: @//${package}:t"
  expect_log_once "Aspect 'b' applied on target: @//${package}:x"
  expect_log_once "Aspect 'b' applied on target: @//${package}:t"
  expect_log_once "Aspect 'a' applied on target: @//${package}:x"
  expect_log_once "Aspect 'a' applied on target: @//${package}:t"
}

function test_aspect_has_access_to_aspect_hints_attribute_in_native_rules() {
  local package="aspect_hints"
  mkdir -p "${package}"
  create_aspect_hints_rule_and_aspect "${package}"
  create_aspect_hints_cc_files "${package}"

  cat > "${package}/BUILD" <<EOF
load("//${package}:hints_counter.bzl", "count_hints")
load("//${package}:hints.bzl", "hint")

hint(name = "first_hint", hints_cnt = 2)
hint(name = "second_hint", hints_cnt = 3)

cc_library(
    name = "cc_foo",
    srcs = ["foo.cc"],
    hdrs = ["foo.h"],
    deps = [":cc_bar"],
    aspect_hints = [":first_hint"],
)
cc_library(
    name = "cc_bar",
    hdrs = ["bar.h"],
    aspect_hints = [":second_hint"]
)

count_hints(name = "cnt", deps = [":cc_foo"])
EOF

  bazel build "//${package}:cnt" \
    --output_groups=out \
    || fail "Build failed"
  assert_contains "Used hints: 5" "./${PRODUCT_NAME}-bin/${package}/cnt_res"
}

function test_aspect_has_access_to_aspect_hints_attribute_in_starlark_rules() {
  local package="aspect_hints"
  mkdir -p "${package}"
  setup_aspect_hints "${package}"

  bazel build "//${package}:cnt" \
    --output_groups=out \
    || fail "Build failed"
  assert_contains "Used hints: 22" "./${PRODUCT_NAME}-bin/${package}/cnt_res"
}

function setup_aspect_hints() {
  local package="$1"
  mkdir -p "${package}"

  create_aspect_hints_rule_and_aspect "${package}"
  create_aspect_hints_custom_rule "${package}"
  create_aspect_hints_BUILD_file "${package}"
}

function create_aspect_hints_BUILD_file() {
  local package="$1"
  mkdir -p "${package}"

cat > "${package}/BUILD" <<EOF
load("//${package}:hints_counter.bzl", "count_hints")
load("//${package}:custom_rule.bzl", "custom_rule")
load("//${package}:hints.bzl", "hint")

hint(name = "first_hint", hints_cnt = 2)
hint(name = "second_hint", hints_cnt = 20)

custom_rule(
    name = "custom_foo",
    deps = [":custom_bar"],
    aspect_hints = [":first_hint"],
)
custom_rule(
    name = "custom_bar",
    aspect_hints = [":second_hint"],
)

count_hints(name = "cnt", deps = [":custom_foo"])
EOF
}

function create_aspect_hints_rule_and_aspect() {
  local package="$1"
  mkdir -p "${package}"

  cat > "${package}/hints.bzl" <<EOF
HintInfo = provider(fields = ['hints_cnt'])

def _hint_impl(ctx):
    return [HintInfo(hints_cnt = ctx.attr.hints_cnt)]

hint = rule(
    implementation = _hint_impl,
    attrs = {'hints_cnt': attr.int(default = 0)},
)
EOF

  cat > "${package}/hints_counter.bzl" <<EOF
load("//${package}:hints.bzl", "HintInfo")
HintsCntInfo = provider(fields = ["cnt"])

def _my_aspect_impl(target, ctx):
    direct_hints_cnt = 0
    transitive_hints_cnt = 0

    for hint in ctx.rule.attr.aspect_hints:
        if HintInfo in hint:
            direct_hints_cnt = direct_hints_cnt + hint[HintInfo].hints_cnt

    for dep in ctx.rule.attr.deps:
        if HintsCntInfo in dep:
            transitive_hints_cnt = transitive_hints_cnt + dep[HintsCntInfo].cnt

    return [HintsCntInfo(cnt = direct_hints_cnt + transitive_hints_cnt)]

my_aspect = aspect(
    implementation = _my_aspect_impl,
    attr_aspects = ["deps"],
    provides = [HintsCntInfo],
)

def _count_hints_impl(ctx):
    cnt = 0
    for dep in ctx.attr.deps:
        if HintsCntInfo in dep:
            cnt = cnt + dep[HintsCntInfo].cnt

    out = ctx.actions.declare_file(ctx.label.name + "_res")
    ctx.actions.run_shell(
        outputs = [out],
        command = "echo Used hints: {} > {}".format(cnt, out.path),
    )
    return [OutputGroupInfo(out = [out])]

count_hints = rule(
    implementation = _count_hints_impl,
    attrs = {
        "deps": attr.label_list(aspects = [my_aspect]),
    },
)
EOF
}

function create_aspect_hints_custom_rule() {
  local package="$1"
  mkdir -p "${package}"

  cat > "${package}/custom_rule.bzl" <<EOF
CustomInfo = provider()

def _custom_rule_impl(ctx):
    return [CustomInfo()]

custom_rule = rule(
    implementation = _custom_rule_impl,
    attrs = {
        "deps": attr.label_list(),
    },
)
EOF
}

function create_aspect_hints_cc_files() {
  local package="$1"
  mkdir -p "${package}"

  cat > "${package}/foo.h" <<EOF
#include "${package}/bar.h"
EOF

  cat > "${package}/foo.cc" <<EOF
#include "${package}/foo.h"
#include "${package}/bar.h"
int callFourtyTwo() {
  return fourtyTwo();
}
EOF

  cat > "${package}/bar.h" <<EOF
inline int fourtyTwo() {
  return 42;
}
EOF
}

function test_aspect_requires_aspect_no_providers_duplicates() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
prov_a = provider()
prov_b = provider()

def _aspect_b_impl(target, ctx):
  res = "aspect_b run on target {}".format(target.label.name)
  print(res)
  return [prov_b(value = res)]

aspect_b = aspect(
  implementation = _aspect_b_impl,
)

def _aspect_a_impl(target, ctx):
  res = "aspect_a run on target {}".format(target.label.name)
  print(res)
  return [prov_a(value = res)]

aspect_a = aspect(
  implementation = _aspect_a_impl,
  requires = [aspect_b],
  attr_aspects = ['dep'],
)

def _rule_1_impl(ctx):
  pass

rule_1 = rule(
  implementation = _rule_1_impl,
  attrs = {
      'dep': attr.label(aspects = [aspect_a]),
  }
)

def _rule_2_impl(ctx):
  pass

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
  name = 't1',
  dep = ':t2',
)
rule_2(
  name = 't2',
  dep = ':t3',
)
rule_2(
  name = 't3',
)
EOF

  bazel build "//${package}:t1" &> $TEST_log || fail "Build failed"

  expect_log "aspect_a run on target t3"
  expect_log "aspect_b run on target t3"
  expect_log "aspect_a run on target t2"
  expect_log "aspect_b run on target t2"
}

# test that although a required aspect runs once on a target, it can still keep
# propagating while inheriting its main aspect attr_aspects
function test_aspect_requires_aspect_no_providers_duplicates_2() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
prov_a = provider()
prov_b = provider()

def _aspect_b_impl(target, ctx):
  res = "aspect_b run on target {}".format(target.label.name)
  print(res)
  return [prov_b(value = res)]

aspect_b = aspect(
  implementation = _aspect_b_impl,
)

def _aspect_a_impl(target, ctx):
  res = "aspect_a run on target {}".format(target.label.name)
  print(res)
  return [prov_a(value = res)]

aspect_a = aspect(
  implementation = _aspect_a_impl,
  requires = [aspect_b],
  attr_aspects = ['dep_1', 'dep_2'],
)

def _empty_impl(ctx):
  pass

rule_1 = rule(
  implementation = _empty_impl,
  attrs = {
      'dep_1': attr.label(aspects = [aspect_a]),
  }
)

rule_2 = rule(
  implementation = _empty_impl,
  attrs = {
    'dep_1': attr.label(aspects = [aspect_b]),
  }
)

rule_3 = rule(
  implementation = _empty_impl,
  attrs = {
    'dep_2': attr.label(),
  }
)
EOF

  cat > "${package}/BUILD" <<EOF
load('//${package}:defs.bzl', 'rule_1', 'rule_2', 'rule_3')
exports_files(["write.sh"])
rule_1(
  name = 't1',
  dep_1 = ':t2',
)
rule_2(
  name = 't2',
  dep_1 = ':t3',
)
rule_3(
  name = 't3',
  dep_2 = ':t4',
)
rule_3(
  name = 't4',
)
EOF

  bazel build "//${package}:t1" &> $TEST_log || fail "Build failed"

  expect_log "aspect_a run on target t4"
  expect_log "aspect_b run on target t4"
  expect_log "aspect_a run on target t3"
  expect_log "aspect_b run on target t3"
  expect_log "aspect_a run on target t2"
  expect_log "aspect_b run on target t2"
}

function test_top_level_aspects_parameters() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}, p1 = {} and p3 = {}'.
                    format(target.label, ctx.attr.p1, ctx.attr.p3))
  return []

aspect_a = aspect(
              implementation = _aspect_a_impl,
              attr_aspects = ['dep'],
              attrs = { 'p1' : attr.string(values = ['p1_v1', 'p1_v2']),
                        'p3' : attr.string(values = ['p3_v1', 'p3_v2', 'p3_v3'])},
)

def _aspect_b_impl(target, ctx):
  print('aspect_b on target {}, p1 = {} and p2 = {}'.
                      format(target.label, ctx.attr.p1, ctx.attr.p2))
  return []

aspect_b = aspect(
            implementation = _aspect_b_impl,
            attr_aspects = ['dep'],
            attrs = { 'p1' : attr.string(values = ['p1_v1', 'p1_v2']),
                      'p2' : attr.string(values = ['p2_v1', 'p2_v2'])},
)

def _my_rule_impl(ctx):
  pass

my_rule = rule(
           implementation = _my_rule_impl,
           attrs = { 'dep' : attr.label() },
)
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'my_rule')
my_rule(name = 'main_target',
        dep = ':dep_target_1',
)
my_rule(name = 'dep_target_1',
        dep = ':dep_target_2',
)
my_rule(name = 'dep_target_2')
EOF

  bazel build "//${package}:main_target" \
      --aspects="//${package}:defs.bzl%aspect_a","//${package}:defs.bzl%aspect_b" \
      --aspects_parameters="p1=p1_v1" \
      --aspects_parameters="p2=p2_v2" \
      --aspects_parameters="p3=p3_v3" \
      &> $TEST_log || fail "Build failed"

  expect_log "aspect_a on target @//test:main_target, p1 = p1_v1 and p3 = p3_v3"
  expect_log "aspect_a on target @//test:dep_target_1, p1 = p1_v1 and p3 = p3_v3"
  expect_log "aspect_a on target @//test:dep_target_2, p1 = p1_v1 and p3 = p3_v3"
  expect_log "aspect_b on target @//test:main_target, p1 = p1_v1 and p2 = p2_v2"
  expect_log "aspect_b on target @//test:dep_target_1, p1 = p1_v1 and p2 = p2_v2"
  expect_log "aspect_b on target @//test:dep_target_2, p1 = p1_v1 and p2 = p2_v2"
}

# aspect_a is propagated from command line on top level target main_target with
# value p_v1 for its parameter p. It is also propagated from main_target rule
# with value p_v2 to its parameter p. So aspect_a should run twice on dependency
# dep_target, once with each parameter value.
function test_top_level_aspects_parameters_with_same_aspect_propagated_from_rule() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}, p = {}'.format(target.label, ctx.attr.p))
  return []

aspect_a = aspect(
              implementation = _aspect_a_impl,
              attr_aspects = ['dep'],
              attrs = { 'p' : attr.string(values = ['p_v1', 'p_v2']) },
)

def _my_rule_impl(ctx):
  pass

my_rule = rule(
           implementation = _my_rule_impl,
           attrs = { 'dep' : attr.label(aspects = [aspect_a]),
                     'p': attr.string()},
)
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'my_rule')
my_rule(name = 'main_target',
        dep = ':dep_target',
        p = 'p_v2',
)
my_rule(name = 'dep_target',
        p = 'p_v1',
)
EOF

  bazel build "//${package}:main_target" \
      --aspects="//${package}:defs.bzl%aspect_a" \
      --aspects_parameters="p=p_v1" \
      &> $TEST_log || fail "Build failed"

  expect_log "aspect_a on target @//test:main_target, p = p_v1"
  expect_log "aspect_a on target @//test:dep_target, p = p_v1"
  expect_log "aspect_a on target @//test:dep_target, p = p_v2"
}

function test_top_level_aspects_parameters_invalid_multiple_param_values() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}, p = {}'.format(target.label, ctx.attr.p))
  return []

aspect_a = aspect(
              implementation = _aspect_a_impl,
              attr_aspects = ['dep'],
              attrs = { 'p' : attr.string(values = ['p_v1', 'p_v2']) },
)

def _my_rule_impl(ctx):
  pass

my_rule = rule(
           implementation = _my_rule_impl,
           attrs = { 'dep' : attr.label(aspects = [aspect_a]),
                     'p': attr.string()},
)
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'my_rule')
my_rule(name = 'main_target',
        dep = ':dep_target',
        p = 'p_v2',
)
my_rule(name = 'dep_target',
        p = 'p_v1',
)
EOF

  bazel build "//${package}:main_target" \
      --aspects="//${package}:defs.bzl%aspect_a" \
      --aspects_parameters="p=p_v1" \
      --aspects_parameters="p=p_v2" \
      &> $TEST_log && fail "Build succeeded, expected to fail"

  expect_log "Error in top-level aspects parameters: Multiple entries with same key: p=p_v2 and p=p_v1"

  bazel build "//${package}:main_target" \
      --aspects="//${package}:defs.bzl%aspect_a" \
      --aspects_parameters="p=p_v1" \
      --aspects_parameters="p=p_v1" \
      &> $TEST_log && fail "Build succeeded, expected to fail"

  expect_log "Error in top-level aspects parameters: Multiple entries with same key: p=p_v1 and p=p_v1"
}

function test_int_top_level_aspects_parameters_invalid_value() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}, p = {}'.format(target.label, ctx.attr.p))
  return []

aspect_a = aspect(
              implementation = _aspect_a_impl,
              attr_aspects = ['dep'],
              attrs = { 'p' : attr.int() },
)

def _aspect_b_impl(target, ctx):
  print('aspect_b on target {}, p = {}'.format(target.label, ctx.attr.p))
  return []

aspect_b = aspect(
              implementation = _aspect_b_impl,
              attr_aspects = ['dep'],
              attrs = { 'p' : attr.string() },
)

def _my_rule_impl(ctx):
  pass

my_rule = rule(
           implementation = _my_rule_impl,
           attrs = { 'dep' : attr.label() },
)
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'my_rule')
my_rule(name = 'main_target',
        dep = ':dep_target',
)
my_rule(name = 'dep_target',
)
EOF

  bazel build "//${package}:main_target" \
      --aspects="//${package}:defs.bzl%aspect_a" \
      --aspects="//${package}:defs.bzl%aspect_b" \
      --aspects_parameters="p=val" \
      &> $TEST_log && fail "Build succeeded, expected to fail"

  expect_log "//test:defs.bzl%aspect_a: expected value of type 'int' for attribute 'p' but got 'val'"
}

function test_aspect_runs_once_with_diff_order_in_base_aspects() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
def _aspect_b_impl(target, ctx):
  print('aspect_b on target {}, p = {}'.format(target.label, ctx.attr.p))
  return []

aspect_b = aspect(
    implementation = _aspect_b_impl,
    attr_aspects = ['deps'],
    attrs = { 'p' : attr.int(values = [3, 4]) },
)

def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}, aspects_path = {}'.format(target.label, ctx.aspect_ids))
  return []

aspect_a = aspect(
    implementation = _aspect_a_impl,
    attr_aspects = ['deps'],
    requires = [aspect_b],
)

def _r1_impl(ctx):
  pass

r1 = rule(
   implementation = _r1_impl,
   attrs = {
     'deps' : attr.label_list(aspects=[aspect_b]),
     'p' : attr.int(),
   },
)

def _r2_impl(ctx):
  pass

r2 = rule(
   implementation = _r2_impl,
   attrs = {
     'deps' : attr.label_list(aspects=[aspect_b]),
     'p' : attr.int(),
   },
)
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'r1', 'r2')
r1(
  name = 't1',
  p = 3,
  # aspect b(p=3) is propagated to t2 and t3 from t1
  # aspect a from command-line runs on t2 and t3 with basekeys = [b(p=4)]
  # so the aspects path propagating to t3 is [b(p=3), b(p=4), a]
  deps = [':t2', ':t3'],
)
r2(
  name = 't2',
  p = 4,
  # aspect b(p=4) is propagated from t2 to t3
  # the aspects path propagating to t3 will be [b(p=4), b(p=3), b(p=4), a]
  # after aspects deduplicating the aspects path will be [b(p=4), b(p=3), a]
  deps = [':t3'],
)
r1(
  name = 't3',
  p = 4,
)
EOF

  bazel build "//${package}:t1" \
      --aspects="//${package}:defs.bzl%aspect_a" \
      --aspects_parameters="p=4" \
      &> $TEST_log || fail "Build failed"

  # check that aspect_a runs only once on t3 with the aspect path ordered as
  # [aspect_b(p=3), aspect_b(p=4), aspect_a]
  expect_log 'aspect_a on target @//test:t3, aspects_path = \["//test:defs.bzl%aspect_b\[p=\\\"3\\\"\]", "//test:defs.bzl%aspect_b\[p=\\\"4\\\"\]", "//test:defs.bzl%aspect_a"\]'
  expect_not_log 'aspect_a on target @//test:t3, aspects_path = \["//test:defs.bzl%aspect_b\[p=\\\"4\\\"\]", "//test:defs.bzl%aspect_b\[p=\\\"3\\\"\]", "//test:defs.bzl%aspect_a"\]'
}

function test_aspect_on_target_with_exec_gp() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}'.format(target.label))
  return []

aspect_a = aspect(
    implementation = _aspect_a_impl,
    attr_aspects = ['deps'],
)

def _rule_impl(ctx):
  pass

r1 = rule(
   implementation = _rule_impl,
   attrs = {
     'deps' : attr.label_list(aspects=[aspect_a]),
   },
)

r2 = rule(
   implementation = _rule_impl,
   attrs = {
     '_tool' : attr.label(
                       default = "//${package}:tool",
                       cfg = config.exec(exec_group = "exec_gp")),
   },
   exec_groups = {"exec_gp": exec_group()},
)
EOF

  cat > "${package}/tool.sh" <<EOF
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'r1', 'r2')
r1(
  name = 't1',
  deps = [':t2'],
)
r2(
  name = 't2',
)

sh_binary(name = "tool", srcs = ["tool.sh"])
EOF

  bazel build "//${package}:t1" \
      &> $TEST_log || fail "Build failed"

  expect_log 'aspect_a on target @//test:t2'
}

function test_aspect_on_aspect_with_exec_gp() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
prov_b = provider()

def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}'.format(target.label))
  return []

aspect_a = aspect(
    implementation = _aspect_a_impl,
    attr_aspects = ['deps'],
    required_aspect_providers = [prov_b],
)

def _aspect_b_impl(target, ctx):
  print('aspect_b on target {}'.format(target.label))
  return [prov_b()]

aspect_b = aspect(
    implementation = _aspect_b_impl,
    attr_aspects = ['deps'],
    provides = [prov_b],
    attrs = {
     '_tool' : attr.label(
                       default = "//${package}:tool",
                       cfg = config.exec(exec_group = "exec_gp")),
   },
   exec_groups = {"exec_gp": exec_group()},
)

def _rule_impl(ctx):
  pass

r1 = rule(
   implementation = _rule_impl,
   attrs = {
     'deps' : attr.label_list(aspects=[aspect_b, aspect_a]),
   },
)

r2 = rule(
   implementation = _rule_impl,
)
EOF

  cat > "${package}/tool.sh" <<EOF
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'r1', 'r2')
r1(
  name = 't1',
  deps = [':t2'],
)
r2(
  name = 't2',
)

sh_binary(name = "tool", srcs = ["tool.sh"])
EOF

  bazel build "//${package}:t1" \
      &> $TEST_log || fail "Build failed"

  expect_log 'aspect_a on target @//test:t2'
  expect_log 'aspect_b on target @//test:t2'
}

function test_aspect_on_aspect_with_missing_exec_gp() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/defs.bzl" <<EOF
prov_b = provider()

def _aspect_a_impl(target, ctx):
  print('aspect_a on target {}'.format(target.label))
  return []

aspect_a = aspect(
    implementation = _aspect_a_impl,
    attr_aspects = ['deps'],
    required_aspect_providers = [prov_b],
    attrs = {
     '_tool' : attr.label(
                       default = "//${package}:tool",
                       # exec_gp not declared
                       cfg = config.exec(exec_group = "exec_gp")),
   },
)

def _aspect_b_impl(target, ctx):
  print('aspect_b on target {}'.format(target.label))
  return [prov_b()]

aspect_b = aspect(
    implementation = _aspect_b_impl,
    attr_aspects = ['deps'],
    provides = [prov_b],
)

def _rule_impl(ctx):
  pass

r1 = rule(
   implementation = _rule_impl,
   attrs = {
     'deps' : attr.label_list(aspects=[aspect_b, aspect_a]),
   },
)

r2 = rule(
   implementation = _rule_impl,
)
EOF

  cat > "${package}/tool.sh" <<EOF
EOF

  cat > "${package}/BUILD" <<EOF
load('//test:defs.bzl', 'r1', 'r2')
r1(
  name = 't1',
  deps = [':t2'],
)
r2(
  name = 't2',
)

sh_binary(name = "tool", srcs = ["tool.sh"])
EOF

  bazel build "//${package}:t1" \
      &> $TEST_log && fail "Build succeeded, expected to fail"

  expect_log "Attr '\$tool' declares a transition for non-existent exec group 'exec_gp'"
}

function test_aspect_with_missing_attr() {
  local package="test"
  mkdir -p "${package}"

  cat > "${package}/BUILD" <<EOF
sh_library(name = "foo")
EOF

  cat > "${package}/a.bzl" <<EOF
def _a_impl(t, ctx):
    return [DefaultInfo()]

a = aspect(
    implementation = _a_impl,
    attrs = {"_missing": attr.label(default = "//missing")},
    attr_aspects = ["*"],
)
EOF

  bazel build -k "//${package}:foo" --aspects="//${package}:a.bzl%a" \
      &> $TEST_log && fail "Build succeeded, expected to fail"

  expect_not_log "IllegalStateException"
  expect_log "no such package 'missing'"
}

run_suite "Tests for aspects"
