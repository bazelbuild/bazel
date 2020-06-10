#!/bin/bash

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

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-01-15, Bazel on Windows only supports MSYS Bash.
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

function set_up() {
  mkdir -p target_skipping || fail "couldn't create directory"
  cat > target_skipping/pass.sh <<EOF || fail "couldn't create pass.sh"
#!/bin/bash
exit 0
EOF
  chmod +x target_skipping/pass.sh

  cat > target_skipping/fail.sh <<EOF|| fail "couldn't create fail.sh"
#!/bin/bash
exit 1
EOF
  chmod +x target_skipping/fail.sh

  cat > target_skipping/BUILD <<EOF || fail "couldn't create BUILD file"
# We're not validating visibility here. Let everything access these targets.
package(default_visibility = ["//visibility:public"])

constraint_setting(name = "not_compatible_setting")

constraint_value(
    name = "not_compatible",
    constraint_setting = ":not_compatible_setting",
)

constraint_setting(name = "foo_version")

constraint_value(
    name = "foo1",
    constraint_setting = ":foo_version",
)

constraint_value(
    name = "foo2",
    constraint_setting = ":foo_version",
)

constraint_value(
    name = "foo3",
    constraint_setting = ":foo_version",
)

constraint_setting(name = "bar_version")

constraint_value(
    name = "bar1",
    constraint_setting = "bar_version",
)

constraint_value(
    name = "bar2",
    constraint_setting = "bar_version",
)

platform(
    name = "foo1_bar1_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo1",
        ":bar1",
    ],
)

platform(
    name = "foo2_bar1_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo2",
        ":bar1",
    ],
)

platform(
    name = "foo1_bar2_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo1",
        ":bar2",
    ],
)

platform(
    name = "foo3_platform",
    parents = ["@local_config_platform//:host"],
    constraint_values = [
        ":foo3",
    ],
)

sh_test(
    name = "pass_on_foo1",
    srcs = ["pass.sh"],
    target_compatible_with = [":foo1"],
)

sh_test(
    name = "fail_on_foo2",
    srcs = ["fail.sh"],
    target_compatible_with = [":foo2"],
)

sh_test(
    name = "pass_on_foo1_bar2",
    srcs = ["pass.sh"],
    target_compatible_with = [
        ":foo1",
        ":bar2",
    ],
)

sh_binary(
    name = "some_foo3_target",
    srcs = ["pass.sh"],
    target_compatible_with = [
        ":foo3",
    ],
)
EOF

  cat > target_skipping/WORKSPACE <<EOF || fail "couldn't create WORKSPACE"
EOF
}

function tear_down() {
  bazel shutdown
}

# Validates that the console log provides useful information to the user for
# builds.
function test_console_log_for_builds() {
  cd target_skipping || fail "couldn't cd into workspace"

  bazel build \
    --show_result=10 \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    ... &> "${TEST_log}" || fail "Bazel failed the test."

  cp "${TEST_log}" "${TEST_UNDECLARED_OUTPUTS_DIR}"/

  expect_log 'Target //:some_foo3_target up-to-date'
  expect_log 'Target //:fail_on_foo2 was skipped'
  expect_log 'Target //:pass_on_foo1 was skipped'
  expect_log 'Target //:pass_on_foo1_bar2 was skipped'
  expect_not_log 'Target //:fail_on_foo2 failed to build'
  expect_not_log 'Target //:pass_on_foo1 failed to build'
  expect_not_log 'Target //:pass_on_foo1_bar2 failed to build'
  expect_not_log 'Target //:fail_on_foo2 up-to-date'
  expect_not_log 'Target //:pass_on_foo1 up-to-date'
  expect_not_log 'Target //:pass_on_foo1_bar2 up-to-date'
}

# Validates that the console log provides useful information to the user for
# tests.
function test_console_log_for_tests() {
  cd target_skipping || fail "couldn't cd into workspace"

  # Get repeatable results from this test.
  bazel clean --expunge

  bazel test \
    --host_platform=@//:foo1_bar1_platform \
    --platforms=@//:foo1_bar1_platform \
    ... &> "${TEST_log}" || fail "Bazel failed unexpectedly."
  expect_log 'Executed 1 out of 3 tests: 1 test passes and 2 were skipped'
  expect_log '^//:pass_on_foo1  *  PASSED in'
  expect_log '^//:fail_on_foo2  *  SKIPPED$'
  expect_log '^//:pass_on_foo1_bar2  *  SKIPPED$'

  bazel test \
    --host_platform=@//:foo2_bar1_platform \
    --platforms=@//:foo2_bar1_platform \
    ... &> "${TEST_log}" && fail "Bazel passed unexpectedly."
  expect_log 'Executed 1 out of 3 tests: 1 fails locally and 2 were skipped.'
  expect_log '^//:pass_on_foo1  *  SKIPPED$'
  expect_log '^//:fail_on_foo2  *  FAILED in'
  expect_log '^//:pass_on_foo1_bar2  *  SKIPPED$'

  # Use :all for this one to validate similar behaviour.
  bazel test \
    --host_platform=@//:foo1_bar2_platform \
    --platforms=@//:foo1_bar2_platform \
    :all &> "${TEST_log}" || fail "Bazel failed unexpectedly."
  expect_log 'Executed 2 out of 3 tests: 2 tests pass and 1 was skipped'
  expect_log '^//:pass_on_foo1  *  PASSED in'
  expect_log '^//:fail_on_foo2  *  SKIPPED$'
  expect_log '^//:pass_on_foo1_bar2  *  PASSED in'
}

# Validates that incompatible target skipping errors behave nicely with
# --keep_going. In other words, even if there's an error in the target skipping
# (e.g. because the user explicitly requested an incompatible target) we still
# want Bazel to finish the rest of the build when --keep_going is specified.
function test_failure_on_incompatible_top_level_target() {
  cd target_skipping || fail "couldn't cd into workspace"

  # Validate a variety of ways to refer to the same target.
  local -r -a incompatible_targets=(
      :pass_on_foo1_bar2
      //:pass_on_foo1_bar2
      @//:pass_on_foo1_bar2
  )

  for incompatible_target in "${incompatible_targets[@]}"; do
    echo "Testing ${incompatible_target}"

    bazel test \
      --show_result=10 \
      --host_platform=@//:foo1_bar1_platform \
      --platforms=@//:foo1_bar1_platform \
      --build_event_text_file="${TEST_log}".build.json \
      "${incompatible_target}" &> "${TEST_log}" \
      && fail "Bazel passed unexpectedly."

    expect_log 'ERROR: Target //:pass_on_foo1_bar2 is incompatible and cannot be built'
    expect_log '^FAILED: Build did NOT complete successfully'

    # Now look at the build event log.
    mv "${TEST_log}".build.json "${TEST_log}"

    expect_log '^    name: "PARSING_FAILURE"$'
    expect_log 'Target //:pass_on_foo1_bar2 is incompatible and cannot be built.'
  done

  # Run an additional (passing) test and make sure we still fail the build.
  # This is intended to validate that --keep_going works as expected.
  bazel test \
    --show_result=10 \
    --host_platform=@//:foo1_bar1_platform \
    --platforms=@//:foo1_bar1_platform \
    --keep_going \
    //:pass_on_foo1_bar2 \
    //:pass_on_foo1 &> "${TEST_log}" && fail "Bazel passed unexpectedly."

  expect_log '^//:pass_on_foo1  *  PASSED in'
  expect_log '^ERROR: command succeeded, but not all targets were analyzed'
  expect_log '^FAILED: Build did NOT complete successfully'
}

# This is basically the same as test_failure_on_incompatible_top_level_target
# above, but with targets in an external repo.
function test_failure_on_incompatible_top_level_target_in_external_repo() {
  cat >> target_skipping/WORKSPACE <<EOF
local_repository(
    name = "test_repo",
    path = "third_party/test_repo",
)
EOF

  mkdir -p target_skipping/third_party/test_repo/
  touch target_skipping/third_party/test_repo/WORKSPACE
  cat > target_skipping/third_party/test_repo/BUILD <<EOF
cc_binary(
    name = "bin",
    srcs = ["bin.cc"],
    target_compatible_with = [
        "@//:foo1",
    ],
)
EOF
  cat > target_skipping/third_party/test_repo/bin.cc <<EOF
int main() {
    return 0;
}
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  bazel test \
    --show_result=10 \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    --build_event_text_file="${TEST_log}".build.json \
    @test_repo//:bin &> "${TEST_log}" && fail "Bazel passed unexpectedly."

  expect_log 'ERROR: Target @test_repo//:bin is incompatible and cannot be built'
  expect_log '^FAILED: Build did NOT complete successfully'

  # Now look at the build event log.
  mv "${TEST_log}".build.json "${TEST_log}"

  expect_log '^    name: "PARSING_FAILURE"$'
  expect_log 'Target @test_repo//:bin is incompatible and cannot be built.'
}

# Crudely validates that the build event protocol contains useful information
# when targets are skipped due to incompatibilities.
function test_build_event_protocol() {
  cd target_skipping || fail "couldn't cd into workspace"

  bazel test \
    --show_result=10 \
    --host_platform=@//:foo1_bar1_platform \
    --build_event_text_file=$TEST_log \
    --platforms=@//:foo1_bar1_platform \
    ... || fail "Bazel failed unexpectedly."
  expect_not_log 'Target //:pass_on_foo1 build was skipped\.'
  expect_log_n 'reason: SKIPPED\>' 3
  expect_log 'Target //:fail_on_foo2 build was skipped\.'
  expect_log 'Target //:pass_on_foo1_bar2 build was skipped\.'
}

# Validates that incompatibilities are transitive. I.e. targets that depend on
# incompatible targets are themselves deemed incompatible and should therefore
# not be built.
function test_non_top_level_skipping() {
  cat >> target_skipping/BUILD <<EOF
genrule(
    name = "genrule_foo1",
    target_compatible_with = [":foo1"],
    outs = ["foo1.sh"],
    cmd = "echo 'Should not be executed' &>2; exit 1",
)

sh_binary(
    name = "sh_foo2",
    srcs = ["foo1.sh"],
    target_compatible_with = [":foo2"],
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  bazel build \
    --show_result=10 \
    --host_platform=@//:foo2_bar1_platform \
    --platforms=@//:foo2_bar1_platform \
    //:sh_foo2 &> "${TEST_log}" && fail "Bazel passed unexpectedly."
  expect_log 'ERROR: Target //:sh_foo2 is incompatible and cannot be built, but was explicitly requested'
  expect_log 'FAILED: Build did NOT complete successfully'
}

# Validate that targets are skipped when the implementation is in Starlark
# instead of in Java.
function test_starlark_skipping() {
  cat > target_skipping/rules.bzl <<EOF
def _echo_rule_impl(ctx):
    ctx.actions.write(ctx.outputs.out, ctx.attr.text)
    return []

_echo_rule = rule(
    implementation = _echo_rule_impl,
    attrs = {
        "text": attr.string(),
        "out": attr.output(),
    },
)

def echo_rule(name, **kwargs):
    _echo_rule(
        name = name,
        out = name,
        **kwargs
    )
EOF

  cat >> target_skipping/BUILD <<EOF
load("//:rules.bzl", "echo_rule")

echo_rule(
    name = "hello_world",
    text = "Hello World",
    target_compatible_with = [
        "//:foo1",
    ],
)

sh_binary(
    name = "hello_world_bin",
    srcs = ["pass.sh"],
    data = [":hello_world"],
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  bazel build \
    --show_result=10 \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    //:hello_world_bin &> "${TEST_log}" && fail "Bazel passed unexpectedly."
  expect_log 'ERROR: Target //:hello_world_bin is incompatible and cannot be built, but was explicitly requested'
  expect_log 'FAILED: Build did NOT complete successfully'
}

# Validates the same thing as test_non_top_level_skipping, but with a cc_test
# and adding one more level of dependencies.
function test_cc_test() {
  cat > target_skipping/generator_tool.cc <<EOF
#include <cstdio>
int main() {
  printf("int main() { return 1; }\\n");
  return 0;
}
EOF

  cat >> target_skipping/BUILD <<EOF
cc_binary(
    name = "generator_tool",
    srcs = ["generator_tool.cc"],
    target_compatible_with = [":foo1"],
)

genrule(
    name = "generate_with_tool",
    outs = ["generated_test.cc"],
    cmd = "\$(location :generator_tool) > \$(OUTS)",
    tools = [":generator_tool"],
)

cc_test(
    name = "generated_test",
    srcs = ["generated_test.cc"],
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  bazel test \
    --show_result=10 \
    --host_platform=@//:foo2_bar1_platform \
    --platforms=@//:foo2_bar1_platform \
    //:generated_test &> "${TEST_log}" && fail "Bazel passed unexpectedly."
  expect_log 'ERROR: Target //:generated_test is incompatible and cannot be built, but was explicitly requested'

  # Validate that we get the dependency chain printed out.
  expect_log '^Dependency chain:$'
  expect_log '^    //:generated_test$'
  expect_log '^    //:generate_with_tool$'
  expect_log "^    //:generator_tool   <-- target platform didn't satisfy constraint //:foo1$"
  expect_log 'FAILED: Build did NOT complete successfully'
}

# Validates the same thing as test_cc_test, but with multiple volated
# constraints.
function test_cc_test_multiple_constraints() {
  cat > target_skipping/generator_tool.cc <<EOF
#include <cstdio>
int main() {
  printf("int main() { return 1; }\\n");
  return 0;
}
EOF

  cat >> target_skipping/BUILD <<EOF
cc_binary(
    name = "generator_tool",
    srcs = ["generator_tool.cc"],
    target_compatible_with = [":foo1", ":bar2"],
)

genrule(
    name = "generate_with_tool",
    outs = ["generated_test.cc"],
    cmd = "\$(location :generator_tool) > \$(OUTS)",
    tools = [":generator_tool"],
)

cc_test(
    name = "generated_test",
    srcs = ["generated_test.cc"],
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  bazel test \
    --show_result=10 \
    --host_platform=@//:foo2_bar1_platform \
    --platforms=@//:foo2_bar1_platform \
    //:generated_test &> "${TEST_log}" && fail "Bazel passed unexpectedly."
  expect_log 'ERROR: Target //:generated_test is incompatible and cannot be built, but was explicitly requested'

  # Validate that we get the dependency chain and constraints printed out.
  expect_log '^Dependency chain:$'
  expect_log '^    //:generated_test$'
  expect_log '^    //:generate_with_tool$'
  expect_log "^    //:generator_tool   <-- target platform didn't satisfy constraints \[//:foo1, //:bar2\]$"
  expect_log 'FAILED: Build did NOT complete successfully'
}

# Validates that we can express targets being compatible with A _or_ B.
function test_or_logic() {
  cat >> target_skipping/BUILD <<EOF
sh_test(
    name = "pass_on_foo1_or_foo2_but_not_on_foo3",
    srcs = [":pass.sh"],
    target_compatible_with = select({
        ":foo1": [],
        ":foo2": [],
        "//conditions:default": [":not_compatible"],
    }),
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  bazel test \
    --show_result=10 \
    --host_platform=@//:foo1_bar1_platform \
    --platforms=@//:foo1_bar1_platform \
    --nocache_test_results \
    //:pass_on_foo1_or_foo2_but_not_on_foo3 &> "${TEST_log}" \
    || fail "Bazel failed unexpectedly."
  expect_log '^//:pass_on_foo1_or_foo2_but_not_on_foo3  *  PASSED in'

  bazel test \
    --show_result=10 \
    --host_platform=@//:foo2_bar1_platform \
    --platforms=@//:foo2_bar1_platform \
    --nocache_test_results \
    //:pass_on_foo1_or_foo2_but_not_on_foo3 &> "${TEST_log}" \
    || fail "Bazel failed unexpectedly."
  expect_log '^//:pass_on_foo1_or_foo2_but_not_on_foo3  *  PASSED in'

  bazel test \
    --show_result=10 \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    //:pass_on_foo1_or_foo2_but_not_on_foo3 &> "${TEST_log}" \
    && fail "Bazel passed unexpectedly."

  expect_log 'ERROR: Target //:pass_on_foo1_or_foo2_but_not_on_foo3 is incompatible and cannot be built, but was explicitly requested'
  expect_log 'FAILED: Build did NOT complete successfully'
}

# Validates that we can express targets being compatible with everything _but_
# A and B.
function test_inverse_logic() {
  (cd target_skipping && setup_skylib_support)

  cat >> target_skipping/BUILD <<EOF
load("@bazel_skylib//lib:selects.bzl", "selects")

sh_test(
    name = "pass_on_everything_but_foo1_and_foo2",
    srcs = [":pass.sh"],
    target_compatible_with = selects.with_or({
        (":foo1", ":foo2"): [":not_compatible"],
        "//conditions:default": [],
    }),
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  # Try with :foo1. This should fail.
  bazel test \
    --show_result=10 \
    --host_platform=@//:foo1_bar1_platform \
    --platforms=@//:foo1_bar1_platform \
    //:pass_on_everything_but_foo1_and_foo2  &> "${TEST_log}" \
    && fail "Bazel passed unexpectedly."
  expect_log 'ERROR: Target //:pass_on_everything_but_foo1_and_foo2 is incompatible and cannot be built, but was explicitly requested'
  expect_log 'FAILED: Build did NOT complete successfully'

  # Try with :foo2. This should fail.
  bazel test \
    --show_result=10 \
    --host_platform=@//:foo2_bar1_platform \
    --platforms=@//:foo2_bar1_platform \
    //:pass_on_everything_but_foo1_and_foo2 &> "${TEST_log}" \
    && fail "Bazel passed unexpectedly."
  expect_log 'ERROR: Target //:pass_on_everything_but_foo1_and_foo2 is incompatible and cannot be built, but was explicitly requested'
  expect_log 'FAILED: Build did NOT complete successfully'

  # Now with :foo3. This should pass.
  bazel test \
    --show_result=10 \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    --nocache_test_results \
    //:pass_on_everything_but_foo1_and_foo2 &> "${TEST_log}" \
    || fail "Bazel failed unexpectedly."
  expect_log '^//:pass_on_everything_but_foo1_and_foo2  *  PASSED in'
}

# Validates that a tool compatible with the host platform, but incompatible
# with the target platform can still be used as a host tool.
function test_host_tool() {
  # Create an arbitrary host tool.
  cat > target_skipping/host_tool.cc <<EOF
#include <cstdio>
int main() {
  printf("Hello World\\n");
  return 0;
}
EOF

  cat >> target_skipping/BUILD <<EOF
cc_binary(
    name = "host_tool",
    srcs = ["host_tool.cc"],
    target_compatible_with = [
        ":foo1",
    ],
)

genrule(
    name = "use_host_tool",
    outs = ["host_tool_message.txt"],
    cmd = "\$(location :host_tool) >> \$(OUTS)",
    tools = [":host_tool"],
    target_compatible_with = [
        ":foo2",
    ],
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  # Run with :foo1 in the host platform, but with :foo2 in the target platform.
  # Building the host tool should fail because it's incompatible with the
  # target platform.
  bazel build --show_result=10  \
    --host_platform=@//:foo1_bar1_platform \
    --platforms=@//:foo2_bar1_platform \
    //:host_tool &> "${TEST_log}" && fail "Bazel passed unexpectedly."
  expect_log 'ERROR: Target //:host_tool is incompatible and cannot be built, but was explicitly requested'
  expect_log 'FAILED: Build did NOT complete successfully'

  # Run with :foo1 in the host platform, but with :foo2 in the target platform.
  # This should work fine because we're not asking for any constraints to be
  # violated.
  bazel build --host_platform=@//:foo1_bar2_platform \
    --platforms=@//:foo2_bar1_platform \
    //:host_tool_message.txt &> "${TEST_log}" \
    || fail "Bazel failed unexpectedly."
  expect_log ' bazel-bin/host_tool_message.txt$'
  expect_log ' Build completed successfully, '

  # Make sure that the contents of the file are what we expect.
  cp bazel-bin/host_tool_message.txt "${TEST_log}"
  expect_log '^Hello World$'
}

# Validates that incompatible targets provide appropriate errors when queried
# explicitly or skipped when queried via a glob.
function test_queries() {
  cat >> target_skipping/BUILD <<EOF
genrule(
    name = "genrule_foo1",
    target_compatible_with = [":foo1"],
    outs = ["foo1.sh"],
    cmd = "echo 'exit 0' > \$(OUTS)",
)

sh_binary(
    name = "sh_foo1",
    srcs = ["foo1.sh"],
)
EOF

  cd target_skipping || fail "couldn't cd into workspace"

  bazel query \
    'deps(//:sh_foo1)' &> "${TEST_log}" \
    || fail "Bazel query failed unexpectedly."
  expect_log '^//:sh_foo1$'
  expect_log '^//:genrule_foo1$'

  # Run a cquery on a target that is compatible. This should pass.
  bazel cquery \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    'deps(//:some_foo3_target)' &> "${TEST_log}" \
    || fail "Bazel cquery failed unexpectedly."

  # Run a cquery with a glob. This should only pick up compatible targets and
  # should therefore pass.
  bazel cquery \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    'deps(//...)' &> "${TEST_log}" \
    || fail "Bazel cquery failed unexpectedly."
  expect_log '^//:some_foo3_target '
  expect_log '^//:sh_foo1 '
  expect_log '^//:genrule_foo1 '

  # Run a cquery on an incompatible target. This should fail.
  bazel cquery \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    'deps(//:sh_foo1)' &> "${TEST_log}" \
    && fail "Bazel cquery passed unexpectedly."
  expect_log 'Target //:sh_foo1 is incompatible and cannot be built, but was explicitly requested'
  expect_log "target platform didn't satisfy constraint //:foo1"

  # Run an aquery on a target that is compatible. This should pass.
  bazel aquery \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    '//:some_foo3_target' &> "${TEST_log}" \
    || fail "Bazel aquery failed unexpectedly."

  # Run an aquery with a glob. This should only pick up compatible targets and
  # should therefore pass.
  bazel aquery \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    '//...' &> "${TEST_log}" \
    || fail "Bazel aquery failed unexpectedly."

  # Run an aquery on an incompatible target. This should fail.
  bazel aquery \
    --host_platform=@//:foo3_platform \
    --platforms=@//:foo3_platform \
    '//:sh_foo1' &> "${TEST_log}" \
    && fail "Bazel aquery passed unexpectedly."
  expect_log 'Target //:sh_foo1 is incompatible and cannot be built, but was explicitly requested'
  expect_log "target platform didn't satisfy constraint //:foo1"
}

run_suite "target_compatible_with tests"
