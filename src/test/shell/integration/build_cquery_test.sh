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
# Integration tests for `bazel build --cquery` / `--universe_scope`.

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

add_to_bazelrc "build --package_path=%workspace%"

#### SETUP #############################################################

# Creates a package $pkg with:
#   //pkg:flavor  - a string build setting (default "default")
#   //pkg:lib     - reads :flavor and writes its value to lib.txt
#   //pkg:wrapper - depends on :lib through a transition that sets :flavor=variant
# So //pkg:lib produces "default" as a top-level target and "variant" when reached
# through //pkg:wrapper.
function setup_multiconfig_pkg() {
  local pkg="$1"
  mkdir -p "$pkg"
  cat > "$pkg/defs.bzl" <<EOF
FlavorInfo = provider(fields = ["value"])

def _flavor_impl(ctx):
    return [FlavorInfo(value = ctx.build_setting_value)]

flavor = rule(
    implementation = _flavor_impl,
    build_setting = config.string(flag = True),
)

def _flavor_transition_impl(settings, attr):
    return {"//$pkg:flavor": "variant"}

_flavor_transition = transition(
    implementation = _flavor_transition_impl,
    inputs = [],
    outputs = ["//$pkg:flavor"],
)

def _lib_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(out, ctx.attr._flavor[FlavorInfo].value)
    return [DefaultInfo(files = depset([out]))]

lib = rule(
    implementation = _lib_impl,
    attrs = {"_flavor": attr.label(default = "//$pkg:flavor")},
)

def _wrapper_impl(ctx):
    return [DefaultInfo(
        files = depset(transitive = [d[DefaultInfo].files for d in ctx.attr.deps]),
    )]

wrapper = rule(
    implementation = _wrapper_impl,
    attrs = {
        "deps": attr.label_list(cfg = _flavor_transition),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
EOF
  cat > "$pkg/BUILD" <<EOF
load(":defs.bzl", "flavor", "lib", "wrapper")

flavor(name = "flavor", build_setting_default = "default")

lib(name = "lib")

wrapper(name = "wrapper", deps = [":lib"])
EOF
}

# Echoes the (sorted, unique) contents of all lib.txt outputs across all
# configurations. Uses `find -L` because bazel-out is a symlink (and outputs of
# transitioned configs live under bazel-out/<config>/bin, not under bazel-bin).
function lib_contents() {
  local pkg="$1"
  find -L bazel-out -name "lib.txt" -path "*$pkg*" -exec cat {} \; 2>/dev/null | sort -u
}

function clean_lib_outputs() {
  local pkg="$1"
  find -L bazel-out -name "lib.txt" -path "*$pkg*" -exec rm -f {} \; 2>/dev/null || true
}

#### TESTS #############################################################

# --cquery alone: builds the matched target in its top-level (default) config.
function test_build_cquery_single_target() {
  local pkg=test_build_cquery_single_target
  setup_multiconfig_pkg "$pkg"

  bazel build --cquery="//$pkg:lib" >& "$TEST_log" || fail "Expected success"
  assert_equals "default" "$(lib_contents "$pkg")"
}

# Residue is the build set; --universe_scope is the analysis universe. Building
# //pkg:lib in the //pkg:wrapper universe produces the transitioned (variant)
# output, unlike a plain top-level build of //pkg:lib (see single_target above,
# which yields "default").
function test_build_cquery_residue_built_in_universe() {
  local pkg=test_build_cquery_residue_built_in_universe
  setup_multiconfig_pkg "$pkg"

  # Universe scope = wrapper => lib is reached through the variant transition.
  bazel build --universe_scope="//$pkg:wrapper" "//$pkg:lib" >& "$TEST_log" \
      || fail "Expected success with universe scope"
  assert_equals "variant" "$(lib_contents "$pkg")"
}

# config() narrows a label that exists in multiple configurations to a single
# configured instance. In the {wrapper, lib} universe :lib appears in two configs
# (default as a top-level target, variant via wrapper's transition);
# config(//pkg:lib, target) selects the top-level (default) instance only.
function test_build_cquery_config_function_selects_instance() {
  local pkg=test_build_cquery_config_function_selects_instance
  setup_multiconfig_pkg "$pkg"

  bazel build --cquery="config(//$pkg:lib, target)" \
      --universe_scope="//$pkg:wrapper,//$pkg:lib" >& "$TEST_log" \
      || fail "Expected success selecting the target-config instance"
  # config() must pick exactly the default (target-config) instance, not both.
  assert_equals "default" "$(lib_contents "$pkg")"
}

# A cquery that matches nothing builds nothing and still succeeds.
function test_build_cquery_empty_result() {
  local pkg=test_build_cquery_empty_result
  setup_multiconfig_pkg "$pkg"
  clean_lib_outputs "$pkg"

  bazel build --cquery="//$pkg:lib intersect //$pkg:wrapper" >& "$TEST_log" \
      || fail "Expected success on empty result"
  assert_equals "" "$(lib_contents "$pkg")"
}

# A malformed cquery expression fails cleanly.
function test_build_cquery_syntax_error() {
  local pkg=test_build_cquery_syntax_error
  setup_multiconfig_pkg "$pkg"

  bazel build --cquery="//$pkg:lib +" >& "$TEST_log" && fail "Expected failure"
  expect_log "Error while parsing --cquery"
}

# Regression: build's --universe_scope lives in a build-only options class, so it
# must not collide with the identically-named query option when `bazel help
# completion` aggregates every command's options.
function test_help_completion_has_no_option_collision() {
  bazel help completion >& "$TEST_log" || fail "bazel help completion crashed"
  # And the standalone cquery command (which also defines --universe_scope) still loads.
  bazel help cquery >& "$TEST_log" || fail "bazel help cquery crashed"
}

# The shared --universe_scope must still work on the standalone cquery command.
function test_cquery_universe_scope_still_works() {
  local pkg=test_cquery_universe_scope_still_works
  setup_multiconfig_pkg "$pkg"
  bazel cquery --universe_scope="//$pkg:wrapper,//$pkg:lib" "//$pkg:lib" >& "$TEST_log" \
      || fail "cquery --universe_scope failed"
  expect_log "^//$pkg:lib ("
}

run_suite "Tests for 'bazel build --cquery'"
