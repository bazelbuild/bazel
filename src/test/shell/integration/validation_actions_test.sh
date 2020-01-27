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
fi

function setup_test_project() {
  mkdir -p validation_actions

  cat > validation_actions/defs.bzl <<'EOF'
def _rule_with_implicit_outs_and_validation_impl(ctx):

  ctx.actions.write(ctx.outputs.main, "main output\n")

  ctx.actions.write(ctx.outputs.implicit, "implicit output\n")

  validation_output = ctx.actions.declare_file(ctx.attr.name + ".validation")
  # The actual tool will be created in individual tests, depending on whether
  # validation should pass or fail.
  ctx.actions.run(
      outputs = [validation_output],
      executable = ctx.executable._validation_tool,
      arguments = [validation_output.path])

  return [
    DefaultInfo(files = depset([ctx.outputs.main])),
    OutputGroupInfo(_validation = depset([validation_output])),
  ]


rule_with_implicit_outs_and_validation = rule(
  implementation = _rule_with_implicit_outs_and_validation_impl,
  outputs = {
    "main": "%{name}.main",
    "implicit": "%{name}.implicit",
  },
  attrs = {
    "_validation_tool": attr.label(
        allow_single_file = True,
        default = Label("//validation_actions:validation_tool"),
        executable = True,
        cfg = "host"),
  }
)

def _rule_with_implicit_and_host_deps_impl(ctx):
  return []

rule_with_implicit_and_host_deps = rule(
  implementation = _rule_with_implicit_and_host_deps_impl,
  attrs = {
    "_implicit_dep": attr.label(
        default = Label("//validation_actions:some_implicit_dep")),
    "host_dep": attr.label(
        allow_single_file = True,
        default = Label("//validation_actions:some_host_dep"),
        cfg = "host"),
  }
)
EOF

  cat > validation_actions/BUILD <<'EOF'

load(
    ":defs.bzl",
    "rule_with_implicit_outs_and_validation",
    "rule_with_implicit_and_host_deps")

rule_with_implicit_outs_and_validation(name = "foo0")
rule_with_implicit_outs_and_validation(name = "foo1")
rule_with_implicit_outs_and_validation(name = "foo2")
rule_with_implicit_outs_and_validation(name = "foo3")

filegroup(
  name = "foo2_filegroup",
  srcs = [":foo2"],
)

filegroup(
  name = "foo3_filegroup_implicit",
  srcs = [":foo3.implicit"],
)

# Even when a target, or one of its implicit outputs, is depended upon, its
# validation actions should still be run.
genrule(
  name = "gen",
  srcs = [
    # Dependency on a target
    ":foo0",

    # Dependency on an implicit output
    ":foo1.implicit",

    # Dependency on a filegroup
    ":foo2_filegroup",

    # Dependency on a filegroup which contains an implicit output
    ":foo3_filegroup_implicit",
  ],
  outs = ["generated"],
  cmd = "touch $@",
)


rule_with_implicit_outs_and_validation(name = "some_implicit_dep")
rule_with_implicit_outs_and_validation(name = "some_host_dep")

# this uses the above two rule_with_implicit_outs_and_validation targets
rule_with_implicit_and_host_deps(name = "target_with_implicit_and_host_deps")

rule_with_implicit_outs_and_validation(name = "some_tool_dep")
genrule(
  name = "genrule_with_exec_tool_deps",
  exec_tools = [":some_tool_dep"],
  outs = ["genrule_with_exec_tool_deps_out"],
  cmd = "touch $@",
)

sh_test(
  name = "test_with_rule_with_validation_in_deps",
  srcs = ["test_with_rule_with_validation_in_deps.sh"],
  data = [":some_implicit_dep"],
)

EOF

cat > validation_actions/test_with_rule_with_validation_in_deps.sh <<'EOF'
exit 0
EOF
chmod +x validation_actions/test_with_rule_with_validation_in_deps.sh

}


function setup_passing_validation_action() {
    cat > validation_actions/validation_tool <<'EOF'
#!/bin/bash
echo "validation output" > $1
EOF
  chmod +x validation_actions/validation_tool
}


function setup_failing_validation_action() {
    cat > validation_actions/validation_tool <<'EOF'
#!/bin/bash
echo "validation failed!"
exit 1
EOF
  chmod +x validation_actions/validation_tool
}

function assert_exists() {
  path="$1"
  [ -f "$path" ] && return 0

  fail "Expected file '$path' to exist, but it did not"
  return 1
}

#### Tests #####################################################################

function test_validation_actions() {
  setup_test_project
  setup_passing_validation_action

  bazel build --experimental_run_validations //validation_actions:foo0 || fail "Expected build to succeed"

  assert_exists bazel-bin/validation_actions/foo0.validation
}

function test_validation_actions_implicit_output() {
  setup_test_project
  setup_passing_validation_action

  # Requesting an implicit output
  bazel build --experimental_run_validations //validation_actions:foo0.implicit || fail "Expected build to succeed"

  assert_exists bazel-bin/validation_actions/foo0.validation
}

function test_validation_actions_through_deps() {
  setup_test_project
  setup_passing_validation_action

  bazel build --experimental_run_validations //validation_actions:gen || fail "Expected build to succeed"

  assert_exists bazel-bin/validation_actions/foo0.validation
  assert_exists bazel-bin/validation_actions/foo1.validation
  assert_exists bazel-bin/validation_actions/foo2.validation
  assert_exists bazel-bin/validation_actions/foo3.validation
}

function test_failing_validation_action_fails_build() {
  setup_test_project
  setup_failing_validation_action

  bazel build --experimental_run_validations //validation_actions:gen >& "$TEST_log" && fail "Expected build to fail"
  expect_log "validation failed!"
}

function test_failing_validation_action_with_validations_off_does_not_fail_build() {
  setup_test_project
  setup_failing_validation_action

  bazel build --experimental_run_validations=false //validation_actions:gen >& "$TEST_log" || fail "Expected build to succeed"
  expect_not_log "validation failed!"
}

function test_failing_validation_action_for_host_dep_does_not_fail_build() {
  setup_test_project
  setup_failing_validation_action

  # Validation actions in the host configuration or from implicit deps should
  # not fail the overall build, since those dependencies should have their own
  # builds and tests that should surface any failing validations.
  bazel build --experimental_run_validations //validation_actions:target_with_implicit_and_host_deps >& "$TEST_log" || fail "Expected build to succeed"
  expect_not_log "validation failed!"
}

function test_failing_validation_action_for_tool_dep_does_not_fail_build() {
  setup_test_project
  setup_failing_validation_action

  # Validation actions in the exec configuration or from implicit deps should
  # not fail the overall build, since those dependencies should have their own
  # builds and tests that should surface any failing validations.
  bazel build --experimental_run_validations //validation_actions:genrule_with_exec_tool_deps >& "$TEST_log" || fail "Expected build to succeed"
  expect_not_log "validation failed!"
}

function test_failing_validation_action_for_dep_from_test_fails_build() {
  setup_test_project
  setup_failing_validation_action

  # Validation actions in the the deps of the test should fail the build when
  # run with bazel test.
  bazel test --experimental_run_validations //validation_actions:test_with_rule_with_validation_in_deps >& "$TEST_log" && fail "Expected build to fail"
  expect_log "validation failed!"
}

run_suite "Validation actions integration tests"
