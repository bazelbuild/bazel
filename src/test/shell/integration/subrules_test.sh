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
  mkdir -p subrule_testing
  cat > subrule_testing/rule.bzl <<EOF
def _subrule_impl(ctx, _foo, _bar):
    pass

my_subrule = subrule(
    implementation = _subrule_impl,
    attrs = {
        "_foo": attr.label(default = "//some:label"),
        "_bar": attr.label(default = "//some:label_1"),
    },
)

def _rule_impl(ctx):
    my_subrule()
    return []

my_rule = rule(
    implementation = _rule_impl,
    attrs = {
        "_foo": attr.label(),
    },
    subrules = [my_subrule],
)

def _aspect_impl(target, ctx):
  return []

_my_aspect = aspect(implementation = _aspect_impl, subrules = [my_subrule])

def _rule_with_aspect_impl(ctx):
  return []

my_rule_with_aspect = rule(
    implementation = _rule_with_aspect_impl,
    attrs = {"dep": attr.label(aspects = [_my_aspect])},
)
EOF
  cat > subrule_testing/BUILD <<EOF
load(":rule.bzl", "my_rule", "my_rule_with_aspect")
my_rule(name = 'foo')
my_rule_with_aspect(name = 'foo_with_aspect', dep = '//some:label_2')
EOF

  mkdir -p some
  cat > some/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
genrule(name = 'label', cmd = '', outs = ['0.out'])
genrule(name = 'label_1', cmd = '', outs = ['1.out'])
genrule(name = 'label_2', cmd = '', outs = ['2.out'])
EOF
}

function test_query_xml_outputs_subrule_implicit_deps() {
  bazel query --output xml //subrule_testing:foo &> $TEST_log || fail "query failed"
  expect_log '<rule-input name="//some:label"/>'
  expect_log '<rule-input name="//some:label_1"/>'
}

function test_query_xml_outputs_subrule_implicit_deps_via_aspect() {
  bazel query --output xml --xml:default_values //subrule_testing:foo_with_aspect &> $TEST_log || fail "query failed"
  expect_log '<rule-input name="//some:label"/>'
  expect_log '<rule-input name="//some:label_1"/>'
}

function test_query_xml_outputs_subrule_attributes() {
  bazel query --output xml --xml:default_values //subrule_testing:foo &> $TEST_log || fail "query failed"
  expect_log '<label name="$//subrule_testing:rule.bzl%my_subrule%_foo" value="//some:label"/>'
  expect_log '<label name="$//subrule_testing:rule.bzl%my_subrule%_bar" value="//some:label_1"/>'
}

# native.existing_rules skips all implicit attributes so this is trivially true
function test_subrule_attributes_are_not_visible_to_native.existing_rules() {
  cat > subrule_testing/helper.bzl <<EOF
def print_rules():
  rules = native.existing_rules()
  for name, rule in rules.items():
    for attr, value in rule.items():
      print('rule:', name + ', attr:', attr + ', value:', value)
EOF
  cat > subrule_testing/BUILD <<EOF
load(":rule.bzl", "my_rule")
load(":helper.bzl", "print_rules")
my_rule(name = 'foo')
print_rules()
EOF
  bazel shutdown # Google-internal testing hook assumes --experimental_rule_extension_api never changes
  bazel build --experimental_rule_extension_api --nobuild //subrule_testing:foo &> $TEST_log || fail "analysis failed"
  expect_log 'rule: foo, attr: kind, value: my_rule'
  expect_not_log '_foo'
}

run_suite "Tests for subrules"
