#!/bin/bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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
# An end-to-end test for Bazel's *experimental* subrule API.

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

function set_up() {
  mkdir -p deleted_attrs_testing
  cat > "deleted_attrs_testing/utils.bzl" <<EOF
def print_attr_names(ctx):
    for name in dir(ctx.attr):
        print("attribute {}".format(name))
EOF
}

function test_deleted_attr() {
  cat > "deleted_attrs_testing/rule.bzl" <<EOF
load(":utils.bzl", "print_attr_names")

def _impl(ctx):
    print_attr_names(ctx)

my_rule = rule(
    implementation = _impl,
    attrs = {
        "foo": attr.string(),
    },
    deleted_attrs = ["package_metadata"],
)
EOF
  cat > "deleted_attrs_testing/BUILD" <<EOF
load(":rule.bzl", "my_rule")

my_rule(name = "foo")
EOF
  bazel build --nobuild //deleted_attrs_testing:foo &> $TEST_log || fail "analysis failed"
  expect_log 'attribute name'
  expect_log 'attribute foo'
  expect_log 'attribute exec_compatible_with'
  expect_not_log 'attribute package_metadata'
}

function test_deleted_attr_with_parent_fails() {
  cat > "deleted_attrs_testing/rule.bzl" <<EOF
load(":utils.bzl", "print_attr_names")

def _parent_impl(ctx):
    print_attr_names(ctx)

_my_parent_rule = rule(
    implementation = _parent_impl,
    attrs = {
        "foo": attr.string(),
    },
    deleted_attrs = ["package_metadata"],
)

def _impl(ctx):
    print_attr_names(ctx)

my_rule = rule(
    implementation = _impl,
    attrs = {
        "bar": attr.string(),
    },
    parent = _my_parent_rule,
    deleted_attrs = ["package_metadata"],
)
EOF
  cat > "deleted_attrs_testing/BUILD" <<EOF
load(":rule.bzl", "my_rule")

my_rule(name = "foo")
EOF
  bazel build --nobuild //deleted_attrs_testing:foo &> $TEST_log || true
  expect_log 'Rules with parents must not have deleted attributes'
}

function test_deleted_attr_on_parent() {
  cat > "deleted_attrs_testing/rule.bzl" <<EOF
load(":utils.bzl", "print_attr_names")

def _parent_impl(ctx):
    print_attr_names(ctx)

_my_parent_rule = rule(
    implementation = _parent_impl,
    attrs = {
        "foo": attr.string(),
    },
    deleted_attrs = ["package_metadata"],
)

def _impl(ctx):
    print_attr_names(ctx)

my_rule = rule(
    implementation = _impl,
    attrs = {
        "bar": attr.string(),
    },
    parent = _my_parent_rule,
)
EOF
  cat > "deleted_attrs_testing/BUILD" <<EOF
load(":rule.bzl", "my_rule")

my_rule(name = "foo")
EOF
  bazel build --nobuild //deleted_attrs_testing:foo &> $TEST_log || true
  expect_log 'attribute name'
  expect_log 'attribute foo'
  expect_log 'attribute bar'
  expect_log 'attribute exec_compatible_with'
  expect_not_log 'attribute package_metadata'
}

function test_deleted_attr_with_subrule_fails() {
  cat > "deleted_attrs_testing/rule.bzl" <<EOF
load(":utils.bzl", "print_attr_names")

def _subrule_impl(ctx, foo):
    pass

_my_subrule = subrule(
    implementation = _subrule_impl,
    attrs = {
        "_foo": attr.label(default = "//some:label"),
    },
)

def _impl(ctx):
    print_attr_names(ctx)

my_rule = rule(
    implementation = _impl,
    attrs = {
        "bar": attr.string(),
    },
    subrules = [_my_subrule],
    deleted_attrs = ["package_metadata"],
)
EOF
  cat > "deleted_attrs_testing/BUILD" <<EOF
load(":rule.bzl", "my_rule")

my_rule(name = "foo")
EOF
  bazel build --nobuild //deleted_attrs_testing:foo &> $TEST_log || true
  expect_log 'Rules with subrules must not have deleted attributes'
}

run_suite "Tests for rule.deleted_attrs"
