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
# Test related to exec groups.
#

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

# NOTE: All tests need to delcare targets in a custom package, which is why they
# all use the pkg=${FUNCNAME[0]} variable.

function test_target_exec_properties_starlark() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/rules.bzl << EOF
def _impl(ctx):
  out_file = ctx.outputs.output
  ctx.actions.run_shell(inputs = [], outputs = [out_file], arguments=[out_file.path], progress_message = "Saying hello", command = "echo hello > \"\$1\"")

my_rule = rule(
  implementation = _impl,
  attrs = {
    "output": attr.output(),
  }
)
EOF
  cat > ${pkg}/BUILD << EOF
load("//${pkg}:rules.bzl", "my_rule")

constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
my_rule(
    name = "a",
    output = "out.txt",
    exec_properties = {"key3": "value3", "overridden": "child_value"},
    exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "key2": "value2",
        "overridden": "parent_value",
    },
    constraint_values = [":local"],
)
EOF

  bazel build --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt &> $TEST_log || fail "Build failed"
  grep "key2" out.txt || fail "Did not find the platform key"
  grep "key3" out.txt || fail "Did not find the target attribute key"
  grep "child_value" out.txt || fail "Did not find the overriding value"
}


function test_target_exec_properties_starlark_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  if "$is_windows"; then
    script_name="test_script.bat"
    script_content="@echo off\necho hello\n"
  else
    script_name="test_script.sh"
    script_content="#!/bin/bash\necho hello\n"
  fi
  cat > ${pkg}/rules.bzl <<EOF
def _impl(ctx):
  out_file = ctx.actions.declare_file("$script_name")
  ctx.actions.write(out_file, "$script_content", is_executable=True)
  return [DefaultInfo(executable = out_file)]

my_rule_test = rule(
  implementation = _impl,
  test = True,
)
EOF
  cat > ${pkg}/BUILD << EOF
load("//${pkg}:rules.bzl", "my_rule_test")

constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
my_rule_test(
    name = "a",
    exec_properties = {"key3": "value3", "overridden": "child_value"},
    exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "key2": "value2",
        "overridden": "parent_value",
    },
    constraint_values = [":local"],
)
EOF

  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt &> $TEST_log || fail "Build failed"
  grep "key2" out.txt || fail "Did not find the platform key"
  grep "key3" out.txt || fail "Did not find the target attribute key"
  grep "child_value" out.txt || fail "Did not find the overriding value"
}

function test_target_exec_properties_cc() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
#include <stdio.h>
int main() {
  printf("Hello\n");
}
EOF
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
cc_binary(
  name = "a",
  srcs = ["a.cc"],
  exec_properties = {"key3": "value3", "overridden": "child_value"},
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "key2": "value2",
        "overridden": "parent_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel build \
      --extra_execution_platforms="${pkg}:my_platform" \
      --toolchain_resolution_debug=.* \
       --execution_log_json_file out.txt \
      ${pkg}:a &> $TEST_log || fail "Build failed"
  grep "key3" out.txt || fail "Did not find the target attribute key"
  grep "child_value" out.txt || fail "Did not find the overriding value"
  grep "key2" out.txt || fail "Did not find the platform key"
}

function test_target_exec_properties_cc_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
#include <stdio.h>
int main() {
  printf("Hello\n");
}
EOF
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
cc_test(
  name = "a",
  srcs = ["a.cc"],
  exec_properties = {"key3": "value3", "overridden": "child_value"},
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "key2": "value2",
        "overridden": "parent_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt &> $TEST_log || fail "Build failed"
  grep "key2" out.txt || fail "Did not find the platform key"
  grep "key3" out.txt || fail "Did not find the target attribute key"
  grep "child_value" out.txt || fail "Did not find the overriding value"
}

function test_target_test_properties_sh_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.sh <<EOF
#!/bin/bash
echo hello
EOF
  chmod u+x ${pkg}/a.sh
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
sh_test(
  name = "a",
  srcs = ["a.sh"],
  exec_properties = {"key3": "value3", "overridden": "child_value"},
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "key2": "value2",
        "overridden": "parent_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt &> $TEST_log || fail "Build failed"
  grep "key2" out.txt || fail "Did not find the platform key"
  grep "key3" out.txt || fail "Did not find the target attribute key"
  grep "child_value" out.txt || fail "Did not find the overriding value"
}

function test_platform_execgroup_properties_cc_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
int main() {}
EOF
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
cc_test(
  name = "a",
  srcs = ["a.cc"],
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "platform_key": "default_value",
        "test.platform_key": "test_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel build --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Build failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "default_value" out.txt || fail "Did not find the default value"
  grep "test_value" out.txt && fail "Used the test-action value when not testing"

  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Test failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "test_value" out.txt || fail "Did not find the test-action value"
}

function test_platform_execgroup_properties_nongroup_override_cc_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
int main() {}
EOF
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
cc_test(
  name = "a",
  srcs = ["a.cc"],
  exec_properties = {
    "platform_key": "override_value",
  },
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "platform_key": "default_value",
        "test.platform_key": "test_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel build --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Build failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "override_value" out.txt || fail "Did not find the overriding value"
  grep "default_value" out.txt && fail "Used the default value"

  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Test failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "override_value" out.txt || fail "Did not find the overriding value"
}

function test_platform_execgroup_properties_group_override_cc_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
int main() {}
EOF
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
cc_test(
  name = "a",
  srcs = ["a.cc"],
  exec_properties = {
    "test.platform_key": "test_override",
  },
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "platform_key": "default_value",
        "test.platform_key": "test_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel build --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Build failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "default_value" out.txt || fail "Used the default value"

  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Test failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "test_override" out.txt || fail "Did not find the overriding test-action value"
}

function test_platform_execgroup_properties_override_group_and_default_cc_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
int main() {}
EOF
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
cc_test(
  name = "a",
  srcs = ["a.cc"],
  exec_properties = {
    "platform_key": "override_value",
    "test.platform_key": "test_override",
  },
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "platform_key": "default_value",
        "test.platform_key": "test_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel build --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Build failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "override_value" out.txt || fail "Did not find the overriding value"
  grep "default_value" out.txt && fail "Used the default value"

  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Test failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "test_override" out.txt || fail "Did not find the overriding test-action value"
}

function test_platform_properties_only_applied_for_relevant_execgroups_cc_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
int main() {}
EOF
  cat > ${pkg}/BUILD <<EOF
constraint_setting(name = "setting")
constraint_value(name = "local", constraint_setting = ":setting")
cc_test(
  name = "a",
  srcs = ["a.cc"],
  exec_compatible_with = [":local"],
)

platform(
    name = "my_platform",
    parents = ["${default_host_platform}"],
    exec_properties = {
        "platform_key": "default_value",
        "unknown.platform_key": "unknown_value",
    },
    constraint_values = [":local"],
)
EOF
  bazel test --extra_execution_platforms="${pkg}:my_platform" ${pkg}:a --execution_log_json_file out.txt || fail "Build failed"
  grep "platform_key" out.txt || fail "Did not find the platform key"
  grep "default_value" out.txt || fail "Did not find the default value"
}

function test_cannot_set_properties_for_irrelevant_execgroup_on_target_cc_test() {
  local -r pkg=${FUNCNAME[0]}
  mkdir $pkg || fail "mkdir $pkg"
  cat > ${pkg}/a.cc <<EOF
int main() {}
EOF
  cat > ${pkg}/BUILD <<EOF
cc_test(
  name = "a",
  srcs = ["a.cc"],
  exec_properties = {
    "platform_key": "default_value",
    "unknown.platform_key": "unknown_value",
  },
)
EOF
  bazel test ${pkg}:a &> $TEST_log && fail "Build passed when we expected an error"
  grep "Tried to set properties for non-existent exec group" $TEST_log || fail "Did not complain about unknown exec group"
}

run_suite "exec group test"

