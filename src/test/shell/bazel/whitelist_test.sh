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
# test behavior with
# //src/main/java/com/google/devtools/build/lib/analysis/Whitelist.java
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


# Test that the whitelist for starlark transitions in bazel will work
# with external dependencies that use starlark transitions.
#
# repo structure:
# ${WORKSPACE_DIR}/
#     hotsauce/
#       BUILD
#       rules.bzl
#         rule_with_transition
# repo2/
#   vinegar/
#   WORKSPACE
#     local_repository
#   rules.bzl
#     rule_with_external_dep
#   BUILD
#
# The whitelist for starlark transitions is set to a package group of "//..."
function test_whitelist_includes_external_deps() {
  create_new_workspace
  repo2=${new_workspace_dir}
  mkdir -p hotsauce
  cat > hotsauce/rules.bzl <<EOF
def _my_transition_impl(settings, attr):
    _ignore = (settings, attr)
    return {'//command_line_option:test_arg': ['tapatio']}
my_transition = transition(
    implementation = _my_transition_impl,
    inputs = [],
    outputs = ["//command_line_option:test_arg"]
)
def _rule_with_transition_impl(ctx):
    return []
rule_with_transition = rule(
    implementation = _rule_with_transition_impl,
    attrs = {
        "dep": attr.label(cfg = my_transition),
        "_whitelist_function_transition": attr.label(default = "@//tools/whitelists/function_transition_whitelist"),
    }
)
EOF
  cat > hotsauce/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
load("//hotsauce:rules.bzl", "rule_with_transition")
rule_with_transition(name = "hotsauce", dep = ":pepper")
rule_with_transition(name = "pepper")
EOF

  cd ${WORKSPACE_DIR}
  mkdir -p vinegar
  cat > WORKSPACE <<EOF
local_repository(name = 'secret_ingredient', path = "${repo2}")
EOF
  cat > vinegar/rules.bzl <<EOF
def _impl(ctx):
  return []
rule_with_external_dep = rule(
  implementation = _impl,
  attrs = {
    "dep": attr.label()
  }
)
EOF
  cat > vinegar/BUILD<<EOF
load("//vinegar:rules.bzl", "rule_with_external_dep")
rule_with_external_dep(
    name = "vinegar",
    dep = "@secret_ingredient//hotsauce:hotsauce"
)
EOF

  bazel cquery "deps(//vinegar)" --test_arg=hotlanta --transitions=full \
    --noimplicit_deps --experimental_starlark_config_transitions \
    >& $TEST_log || fail "failed to query //vinegar"
  expect_log "@secret_ingredient//hotsauce"
  expect_log "test_arg:\[hotlanta\] -> \[\[tapatio\]\]"

}

run_suite "whitelist tests"
