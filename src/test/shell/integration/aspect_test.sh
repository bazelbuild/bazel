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
  expect_log "ERROR: Analysis of target '//test:brokentarget' failed"
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
  # propagate to its dependencies where they will run based on the depdencies
  # advertised providers.
  bazel build "${package}:target_with_a" \
        --aspects="//${package}:lib.bzl%aspect_a" \
        --aspects="//${package}:lib.bzl%aspect_b" &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # Only aspect_a is applied on target_with_a because its "provided" providers
  # do not macth aspect_b required providers.
  expect_log "aspect_a runs on target //${package}:target_with_a"
  expect_not_log "aspect_b runs on target //${package}:target_with_a"

  # Only aspect_a can run on target_with_a_indeps
  expect_log "aspect_a runs on target //${package}:target_with_a_indeps"
  expect_not_log "aspect_b runs on target //${package}:target_with_a_indeps"

  # Only aspect_b can run on target_with_bc
  expect_not_log "aspect_a runs on target //${package}:target_with_bc"
  expect_log "aspect_b runs on target //${package}:target_with_bc"

  # using --incompatible_top_level_aspects_require_providers, the top level
  # target rule's advertised providers will be checked and only aspect_a will be
  # applied on target_with_a and propgated to its dependencies.
  bazel build "${package}:target_with_a" \
        --aspects="//${package}:lib.bzl%aspect_a" \
        --aspects="//${package}:lib.bzl%aspect_b" &>"$TEST_log" \
        --incompatible_top_level_aspects_require_providers \
      || fail "Build failed but should have succeeded"

  # Only aspect_a is applied on target_with_a
  expect_log "aspect_a runs on target //${package}:target_with_a"
  expect_not_log "aspect_b runs on target //${package}:target_with_a"

  # Only aspect_a can run on target_with_a_indeps
  expect_log "aspect_a runs on target //${package}:target_with_a_indeps"
  expect_not_log "aspect_b runs on target //${package}:target_with_a_indeps"

  # rule_with_bc advertised provides only match the required providers for
  # aspect_b, but aspect_b is not propagated from target_with_a
  expect_not_log "aspect_a runs on target //${package}:target_with_bc"
  expect_not_log "aspect_b runs on target //${package}:target_with_bc"
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
  expect_log "my_aspect runs on target //${package}:target_without_providers"
  expect_log "my_aspect runs on target //${package}:target_with_providers"
  expect_log "my_aspect runs on target //${package}:target_with_providers_not_advertised"
  expect_log "my_aspect runs on target //${package}:target_with_providers_indeps"
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
  expect_log "my_aspect runs on target //${package}:target_with_ab"
  expect_log "my_aspect runs on target //${package}:target_with_ab_indeps_reached"
  expect_not_log "/^my_aspect runs on target //${package}:target_with_a$/"
  expect_not_log "my_aspect runs on target //${package}:target_without_providers"
  expect_not_log "my_aspect runs on target //${package}:target_with_ab_not_advertised"

  # my_aspect cannot be propagated to target_with_ab_indeps_not_reached
  # because it was not applied to its parent (target_with_a)
  expect_not_log "my_aspect runs on target //${package}:target_with_ab_indeps_not_reached"
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
  expect_log "my_aspect runs on target //${package}:target_with_ab"
  expect_log "my_aspect runs on target //${package}:target_with_c"
  expect_log "my_aspect runs on target //${package}:target_with_c_indeps_reached"
  expect_not_log "my_aspect runs on target //${package}:target_without_providers"
  expect_not_log "/^my_aspect runs on target //${package}:target_with_a$/"
  expect_not_log "my_aspect runs on target //${package}:target_with_ab_not_advertised"

  # my_aspect cannot be propagated to target_with_c_indeps_not_reached because it was
  # not applied to its parent (target_with_a)
  expect_not_log "my_aspect runs on target //${package}:target_with_c_indeps_not_reached"
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

  bazel build "${package}:main" \
      --experimental_required_aspects &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # base_aspect will run on dep_target_without_providers,
  # dep_target_without_providers_1 and dep_target_with_prov_a but
  # required_aspect will only run on dep_target_with_prov_a because
  # it satisfies its required providers
  expect_log "base_aspect runs on target //${package}:dep_target_with_prov_a"
  expect_log "base_aspect runs on target //${package}:dep_target_without_providers_1"
  expect_log "base_aspect runs on target //${package}:dep_target_without_providers"
  expect_log "required_aspect runs on target //${package}:dep_target_with_prov_a"
  expect_not_log "required_aspect runs on target //${package}:dep_target_without_providers_1"
  expect_not_log "required_aspect runs on target //${package}:dep_target_without_providers/"
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

  bazel build "${package}:test" \
      --experimental_required_aspects &>"$TEST_log" \
      || fail "Build failed but should have succeeded"

  # Check that aspects: aspect_a, aspect_b, aspect_c were propagated to the
  # dependencies of target test: t and x when only aspect_a is specified
  # in the rule definition
  expect_log_once "Aspect 'c' applied on target: //${package}:x"
  expect_log_once "Aspect 'c' applied on target: //${package}:t"
  expect_log_once "Aspect 'b' applied on target: //${package}:x"
  expect_log_once "Aspect 'b' applied on target: //${package}:t"
  expect_log_once "Aspect 'a' applied on target: //${package}:x"
  expect_log_once "Aspect 'a' applied on target: //${package}:t"
}

run_suite "Tests for aspects"
