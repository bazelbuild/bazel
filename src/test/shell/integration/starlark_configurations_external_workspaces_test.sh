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
# starlark_configuration_test.sh: integration tests for starlark build
# configurations with external workspaces.

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

add_to_bazelrc "build --package_path=%workspace%"

#### HELPER FXNS #######################################################

function write_build_setting_bzl() {
  declare -r at_workspace="${1:-}"

  cat > $pkg/provider.bzl <<EOF
BuildSettingInfo = provider(fields = ['name', 'value'])
EOF

  cat > $pkg/build_setting.bzl <<EOF
load("@${WORKSPACE_NAME}//$pkg:provider.bzl", "BuildSettingInfo")

def _build_setting_impl(ctx):
  return [BuildSettingInfo(name = ctx.attr.name, value = ctx.build_setting_value)]

drink_attribute = rule(
  implementation = _build_setting_impl,
  build_setting = config.string(flag = True),
)
EOF

  cat > $pkg/rules.bzl <<EOF
load("@${WORKSPACE_NAME}//$pkg:provider.bzl", "BuildSettingInfo")

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
    "_type":attr.label(default = Label("$at_workspace//$pkg:type")),
    "_temp":attr.label(default = Label("$at_workspace//$pkg:temp")),
  },
  fragments = ["java"],
)
EOF

  cat > $pkg/BUILD <<EOF
load("//$pkg:build_setting.bzl", "drink_attribute")
load("//$pkg:rules.bzl", "drink")

drink(name = 'my_drink')

drink_attribute(
  name = 'type',
  build_setting_default = 'unknown',
  visibility = ['//visibility:public'],
)
drink_attribute(
  name = 'temp',
  build_setting_default = 'unknown',
  visibility = ['//visibility:public'],
)
EOF
}

#### TESTS #############################################################


function test_set_flag_with_workspace_name() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg

  write_build_setting_bzl "@${WORKSPACE_NAME}"

  bazel build //$pkg:my_drink --@//$pkg:type="coffee" \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "type=coffee"
}


run_suite "${PRODUCT_NAME} starlark configurations tests"
