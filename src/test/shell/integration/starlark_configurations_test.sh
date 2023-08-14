#!/bin/bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
# starlark_configuration_test.sh: integration tests for starlark build configurations

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
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

function set_up() {
  write_default_bazelrc
  add_to_bazelrc "build --package_path=%workspace%"
}

#### HELPER FXNS #######################################################

function write_build_setting_bzl() {
 cat > $pkg/build_setting.bzl <<EOF
BuildSettingInfo = provider(fields = ['name', 'value'])

def _build_setting_impl(ctx):
  return [BuildSettingInfo(name = ctx.attr.name, value = ctx.build_setting_value)]

drink_attribute = rule(
  implementation = _build_setting_impl,
  build_setting = config.string(flag = True),
)
EOF

  cat > $pkg/rules.bzl <<EOF
load("//$pkg:build_setting.bzl", "BuildSettingInfo")

def _impl(ctx):
  _type_name = ctx.attr._type[BuildSettingInfo].name
  _type_setting = ctx.attr._type[BuildSettingInfo].value
  print(_type_name + "=" + str(_type_setting))
  _temp_name = ctx.attr._temp[BuildSettingInfo].name
  _temp_setting = ctx.attr._temp[BuildSettingInfo].value
  print(_temp_name + "=" + str(_temp_setting))
  print("strict_java_deps=" + ctx.fragments.java.strict_java_deps)

drink = rule(
  implementation = _impl,
  attrs = {
    "_type":attr.label(default = Label("//$pkg:type")),
    "_temp":attr.label(default = Label("//$pkg:temp")),
  },
  fragments = ["java"],
)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:build_setting.bzl", "drink_attribute")
load("//$pkg:rules.bzl", "drink")

drink(name = 'my_drink')

drink_attribute(name = 'type', build_setting_default = 'unknown')
drink_attribute(name = 'temp', build_setting_default = 'unknown')
EOF
}

#### TESTS #############################################################

function test_default_flag() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  write_build_setting_bzl

  bazel build //$pkg:my_drink > output 2>"$TEST_log" || fail "Expected success"

  expect_log "type=unknown"
}

function test_set_flag() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  write_build_setting_bzl

  bazel build //$pkg:my_drink --//$pkg:type="coffee" \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "type=coffee"
}

function test_starlark_and_native_flag() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  write_build_setting_bzl

  bazel build //$pkg:my_drink --//$pkg:type=coffee --strict_java_deps=off \
    > output 2>"$TEST_log" \
    || fail "Expected success"

  expect_log "type=coffee"
  expect_log "strict_java_deps=off"
}

function test_dont_parse_flags_after_dash_dash() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  write_build_setting_bzl

  bazel build //$pkg:my_drink --//$pkg:type=coffee \
    -- --//$pkg:temp=iced > output 2>"$TEST_log" \
    && fail "Expected failure"

  expect_log "invalid package name '-//test_dont_parse_flags_after_dash_dash'"
}

function test_multiple_starlark_flags() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  write_build_setting_bzl

  bazel build //$pkg:my_drink --//$pkg:type="coffee" --//$pkg:temp="iced" \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "type=coffee"
  expect_log "temp=iced"

  bazel clean 2>"$TEST_log" || fail "Clean failed"

  # Ensure that order doesn't matter.
  bazel build //$pkg:my_drink --//$pkg:temp="iced" --//$pkg:type="coffee" \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "type=coffee"
  expect_log "temp=iced"
}

function test_flag_default_change() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  write_build_setting_bzl

  bazel build //$pkg:my_drink > output 2>"$TEST_log" || fail "Expected success"

  expect_log "type=unknown"

  cat > $pkg/BUILD <<EOF
load("//$pkg:build_setting.bzl", "drink_attribute")
load("//$pkg:rules.bzl", "drink")

drink(name = 'my_drink')

drink_attribute(name = 'type', build_setting_default = 'cowabunga')
drink_attribute(name = 'temp', build_setting_default = 'cowabunga')
EOF

  bazel build //$pkg:my_drink > output 2>"$TEST_log" || fail "Expected success"

  expect_log "type=cowabunga"
}

# Regression tests for b/134580627
# Make sure package incrementality works during options parsing
function test_incremental_delete_build_setting() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  cat > $pkg/rules.bzl <<EOF
def _impl(ctx):
  return []

my_flag = rule(
  implementation = _impl,
  build_setting = config.string(flag = True)
)
simple_rule = rule(implementation = _impl)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "my_flag", "simple_rule")

my_flag(name = "my_flag", build_setting_default = "default")
simple_rule(name = "simple_rule")
EOF

  bazel build //$pkg:simple_rule --//$pkg:my_flag=cowabunga \
    > output 2>"$TEST_log" \
    || fail "Expected success"

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "simple_rule")

simple_rule(name = "simple_rule")
EOF

  bazel build //$pkg:simple_rule --//$pkg:my_flag=cowabunga \
    > output 2>"$TEST_log" \
    && fail "Expected failure" || true

  expect_log "no such target '//$pkg:my_flag'"
}

#############################

function test_incremental_delete_build_setting_in_different_package() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  cat > $pkg/rules.bzl <<EOF
def _impl(ctx):
  return []

my_flag = rule(
  implementation = _impl,
  build_setting = config.string(flag = True)
)
simple_rule = rule(implementation = _impl)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "my_flag")
my_flag(name = "my_flag", build_setting_default = "default")
EOF

  mkdir -p pkg2

  cat > pkg2/BUILD <<EOF
load("//$pkg:rules.bzl", "simple_rule")
simple_rule(name = "simple_rule")
EOF

  bazel build //pkg2:simple_rule --//$pkg:my_flag=cowabunga \
    > output 2>"$TEST_log" || fail "Expected success"

  cat > $pkg/BUILD <<EOF
EOF

  bazel build //pkg2:simple_rule --//$pkg:my_flag=cowabunga \
    > output 2>"$TEST_log" && fail "Expected failure" || true

  expect_log "no such target '//$pkg:my_flag'"
}

#############################


function test_incremental_add_build_setting() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  cat > $pkg/rules.bzl <<EOF
def _impl(ctx):
  return []

my_flag = rule(
  implementation = _impl,
  build_setting = config.string(flag = True)
)
simple_rule = rule(implementation = _impl)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "simple_rule")

simple_rule(name = "simple_rule")
EOF

  bazel build //$pkg:simple_rule \
    > output 2>"$TEST_log" || fail "Expected success"

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "my_flag", "simple_rule")

my_flag(name = "my_flag", build_setting_default = "default")
simple_rule(name = "simple_rule")
EOF

  bazel build //$pkg:simple_rule --//$pkg:my_flag=cowabunga \
    > output 2>"$TEST_log" || fail "Expected success"
}

function test_incremental_change_build_setting() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  cat > $pkg/rules.bzl <<EOF
MyProvider = provider(fields = ["value"])

def _flag_impl(ctx):
  return MyProvider(value = ctx.build_setting_value)

my_flag = rule(
  implementation = _flag_impl,
  build_setting = config.string(flag = True)
)

def _rule_impl(ctx):
  print("flag = " + ctx.attr.flag[MyProvider].value)

simple_rule = rule(
  implementation = _rule_impl,
  attrs = {"flag" : attr.label()}
)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "my_flag", "simple_rule")

my_flag(name = "my_flag", build_setting_default = "default")
simple_rule(name = "simple_rule", flag = ":my_flag")
EOF

  bazel build //$pkg:simple_rule --//$pkg:my_flag=yabadabadoo \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "flag = yabadabadoo"

# update the flag to return a different value
  cat > $pkg/rules.bzl <<EOF
MyProvider = provider(fields = ["value"])

def _flag_impl(ctx):
  return MyProvider(value = "scoobydoobydoo")

my_flag = rule(
  implementation = _flag_impl,
  build_setting = config.string(flag = True)
)

def _rule_impl(ctx):
  print("flag = " + ctx.attr.flag[MyProvider].value)

simple_rule = rule(
  implementation = _rule_impl,
  attrs = {"flag" : attr.label()}
)
EOF

  bazel build //$pkg:simple_rule --//$pkg:my_flag=yabadabadoo \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "flag = scoobydoobydoo"
}

# Test that label-typed build setting has access to providers of the
# target it points to.
function test_label_flag() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "my_rule", "simple_rule")

my_rule(name = "my_rule")

simple_rule(name = "default", value = "default_val")

simple_rule(name = "command_line", value = "command_line_val")

label_flag(
    name = "my_label_build_setting",
    build_setting_default = ":default"
)
EOF

  cat > $pkg/rules.bzl <<EOF
def _impl(ctx):
    _setting = ctx.attr._label_flag[SimpleRuleInfo].value
    print("value=" + _setting)

my_rule = rule(
    implementation = _impl,
    attrs = {
        "_label_flag": attr.label(default = Label("//$pkg:my_label_build_setting")),
    },
)

SimpleRuleInfo = provider(fields = ['value'])

def _simple_rule_impl(ctx):
    return [SimpleRuleInfo(value = ctx.attr.value)]

simple_rule = rule(
    implementation = _simple_rule_impl,
    attrs = {
        "value":attr.string(),
    },
)
EOF

  bazel build //$pkg:my_rule > output 2>"$TEST_log" || fail "Expected success"

  expect_log "value=default_val"

  bazel build //$pkg:my_rule --//$pkg:my_label_build_setting=//$pkg:command_line > output \
    2>"$TEST_log" || fail "Expected success"

  expect_log "value=command_line_val"
}

function test_output_same_config_as_generating_target() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  rm -rf tools/allowlists/function_transition_allowlist
  mkdir -p tools/allowlists/function_transition_allowlist
  cat > tools/allowlists/function_transition_allowlist/BUILD <<EOF
package_group(
    name = "function_transition_allowlist",
    packages = [
        "//...",
    ],
)
EOF

  cat > $pkg/rules.bzl <<EOF
def _rule_class_transition_impl(settings, attr):
    return {
        "//command_line_option:platform_suffix": "blah"
    }

_rule_class_transition = transition(
    implementation = _rule_class_transition_impl,
    inputs = [],
    outputs = [
        "//command_line_option:platform_suffix",
    ],
)

def _rule_class_transition_rule_impl(ctx):
    ctx.actions.write(ctx.outputs.artifact, "hello\n")
    return [DefaultInfo(files = depset([ctx.outputs.artifact]))]

rule_class_transition_rule = rule(
    _rule_class_transition_rule_impl,
    cfg = _rule_class_transition,
    attrs = {
        "_allowlist_function_transition": attr.label(default = "//tools/allowlists/function_transition_allowlist"),
    },
    outputs = {"artifact": "%{name}.output"},
)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:rules.bzl", "rule_class_transition_rule")

rule_class_transition_rule(name = "with_rule_class_transition")
EOF

  bazel build //$pkg:with_rule_class_transition.output > output 2>"$TEST_log" || fail "Expected success"

  bazel cquery "deps(//$pkg:with_rule_class_transition.output, 1)" > output 2>"$TEST_log" || fail "Expected success"

  assert_contains "//$pkg:with_rule_class_transition.output " output
  assert_contains "//$pkg:with_rule_class_transition " output

  # Find the lines of output for the output target and the generating target
  OUTPUT=$(grep "//$pkg:with_rule_class_transition.output " output)
  TARGET=$(grep "//$pkg:with_rule_class_transition " output)

  # Trim to just configuration
  OUTPUT_CONFIG=${OUTPUT#* }
  TARGET_CONFIG=${TARGET#* }

  # Confirm same configuration
  assert_equals $OUTPUT_CONFIG $TARGET_CONFIG
}

function test_starlark_flag_change_causes_action_rerun() {
  local -r pkg="$FUNCNAME"
  mkdir -p "$pkg"

  cat > "$pkg/rules.bzl" <<EOF
BuildSettingInfo = provider(fields = ["value"])

def _impl(ctx):
    return BuildSettingInfo(value = ctx.build_setting_value)

bool_flag = rule(
    implementation = _impl,
    build_setting = config.bool(flag = True),
)
EOF

  cat > "$pkg/BUILD" <<EOF
load(":rules.bzl", "bool_flag")

bool_flag(
    name = "myflag",
    build_setting_default = True,
)

config_setting(
    name = "myflag_selectable",
    flag_values = {":myflag": "True"},
)

genrule(
    name = "test_flag",
    outs = ["out-flag.txt"],
    cmd = "echo Value=" + select({
        ":myflag_selectable": "True",
        "//conditions:default": "False",
    }) + " | tee \$@",
)
EOF

  bazel shutdown

  # Setting --noexperimental_check_output_files is required to reproduce the
  # issue: The bug this test covers was caused by starlark flag changes not
  # invalidating the analysis cache, resulting in Bazel not re-running the
  # genrule action during the third invocation (because it sees the action in
  # skyframe as already executed from the first invocation). Bazel will by
  # default also invalidate actions by checking all output files for
  # modification *unless* an output service is registered (which can -
  # correctly - indicate that the output hasn't changed since the last build,
  # here between the second and third runs). As we don't have an output service
  # available in this test we disable output file checking entirely instead.
  bazel build "//$pkg:test_flag" "--//$pkg:myflag=true" \
      --noexperimental_check_output_files \
      2>>"$TEST_log" || fail "Expected build to succeed"
  assert_equals "Value=True" "$(cat bazel-genfiles/$pkg/out-flag.txt)"

  bazel build "//$pkg:test_flag" "--//$pkg:myflag=false" \
      --noexperimental_check_output_files \
      2>>"$TEST_log" || fail "Expected build to succeed"
  assert_equals "Value=False" "$(cat bazel-genfiles/$pkg/out-flag.txt)"

  bazel build "//$pkg:test_flag" "--//$pkg:myflag=true" \
      --noexperimental_check_output_files \
      2>>"$TEST_log" || fail "Expected build to succeed"
  assert_equals "Value=True" "$(cat bazel-genfiles/$pkg/out-flag.txt)"
}

# Integration test for an invalid output directory from a mnemonic via a
# transition. Integration test required because error is emitted in BuildTool.
# Unit test for dep transition in
# StarlarkRuleTransitionProviderTest#invalidMnemonicFromDepTransition.
function test_invalid_mnemonic_from_transition_top_level() {
  mkdir -p tools/allowlists/function_transition_allowlist test
  cat > tools/allowlists/function_transition_allowlist/BUILD <<'EOF'
package_group(
    name = 'function_transition_allowlist',
    packages = [
        '//test/...',
    ],
)
EOF
  cat > test/rule.bzl <<'EOF'
def _trans_impl(settings, attr):
  return {'//command_line_option:cpu': '//bad:cpu'}

my_transition = transition(implementation = _trans_impl, inputs = [],
  outputs = ['//command_line_option:cpu'])

def _impl(ctx):
  return []

my_rule = rule(
  implementation = _impl,
  cfg = my_transition,
  attrs = {
    '_allowlist_function_transition': attr.label(
        default = '//tools/allowlists/function_transition_allowlist',
    ),
  }
)
EOF
  cat > test/BUILD <<'EOF'
load('//test:rule.bzl', 'my_rule')
my_rule(name = 'test')
EOF
  bazel build //test:test >& "$TEST_log" || exit_code="$?"
  assert_equals 1 "$exit_code" || fail "Expected exit code 1"
  expect_log "CPU name '//bad:cpu'"
  expect_log "is invalid as part of a path: must not contain /"
}

function test_rc_flag_alias_canonicalizes() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  add_to_bazelrc "build --flag_alias=drink=//$pkg:type"

  write_build_setting_bzl

  bazel canonicalize-flags -- --drink=coffee \
    >& "$TEST_log" || fail "Expected success"

  expect_log "--//$pkg:type=coffee"
}

function test_rc_flag_alias_unsupported_under_test_command() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  add_to_bazelrc "test --flag_alias=drink=//$pkg:type"
  write_build_setting_bzl

  bazel canonicalize-flags -- --drink=coffee \
    >& "$TEST_log" && fail "Expected failure"
  expect_log "--flag_alias=drink=//$pkg:type\" disallowed. --flag_alias only "\
"supports the \"build\" command."

  bazel build //$pkg:my_drink >& "$TEST_log" && fail "Expected failure"
  expect_log "--flag_alias=drink=//$pkg:type\" disallowed. --flag_alias only "\
"supports the \"build\" command."

  # Post-test cleanup_workspace() calls "bazel clean", which would also fail
  # unless we reset the bazelrc.
  write_default_bazelrc
}

function test_rc_flag_alias_unsupported_under_conditional_build_command() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  add_to_bazelrc "build:foo --flag_alias=drink=//$pkg:type"
  write_build_setting_bzl

  bazel canonicalize-flags -- --drink=coffee \
>& "$TEST_log" && fail "Expected failure"
  expect_log "--flag_alias=drink=//$pkg:type\" disallowed. --flag_alias only "\
"supports the \"build\" command."

  bazel build //$pkg:my_drink >& "$TEST_log" && fail "Expected failure"
  expect_log "--flag_alias=drink=//$pkg:type\" disallowed. --flag_alias only "\
"supports the \"build\" command."

  # Post-test cleanup_workspace() calls "bazel clean", which would also fail
  # unless we reset the bazelrc.
  write_default_bazelrc
}

function test_rc_flag_alias_unsupported_with_space_assignment_syntax() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  add_to_bazelrc "test --flag_alias drink=//$pkg:type"
  write_build_setting_bzl

  bazel canonicalize-flags -- --drink=coffee \
    >& "$TEST_log" && fail "Expected failure"
  expect_log "--flag_alias\" disallowed. --flag_alias only "\
"supports the \"build\" command."

  bazel build //$pkg:my_drink >& "$TEST_log" && fail "Expected failure"
  expect_log "--flag_alias\" disallowed. --flag_alias only "\
"supports the \"build\" command."

  # Post-test cleanup_workspace() calls "bazel clean", which would also fail
  # unless we reset the bazelrc.
  write_default_bazelrc
}

# Regression test for https://github.com/bazelbuild/bazel/issues/13751.
function test_rule_exec_transition_warning() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  cat > "${pkg}/rules.bzl" <<EOF
def _impl(ctx):
    pass

demo = rule(
    implementation = _impl,
    cfg = config.exec(),
)
EOF

  cat > "${pkg}/BUILD" <<EOF
load(":rules.bzl", "demo")

demo(name = "demo")
EOF


  bazel build //$pkg:demo >& "$TEST_log" && fail "Expected failure"
  expect_not_log "crashed due to an internal error"
  expect_log "`cfg` must be set to a transition appropriate for a rule"
}

run_suite "${PRODUCT_NAME} starlark configurations tests"
