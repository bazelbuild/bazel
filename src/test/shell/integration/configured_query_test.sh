#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# configured_query_test.sh: integration tests for bazel configured query.
# This tests the command line ui of configured query while
# ConfiguredTargetQueryTest tests its internal functionality.

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

add_to_bazelrc "build --package_path=%workspace%"

#### TESTS #############################################################

function test_basic_query() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(name='maple', deps=[':japanese'])
sh_library(name='japanese')
EOF

 bazel cquery "deps(//$pkg:maple)" > output 2>"$TEST_log" || fail "Expected success"

 assert_contains "//$pkg:maple" output
 assert_contains "//$pkg:japanese" output
}

function test_basic_query_output_textproto() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(name='maple', deps=[':japanese'])
sh_library(name='japanese')
EOF

 bazel cquery --output=textproto "deps(//$pkg:maple)" > output 2>"$TEST_log" || fail "Expected success"

 assert_contains "name: \"//$pkg:maple\"" output
 assert_contains "name: \"//$pkg:japanese\"" output
}

function test_respects_selects() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(
    name = "ash",
    deps = select({
        ":excelsior": [":foo"],
        ":americana": [":bar"],
    }),
)
sh_library(name = "foo")
sh_library(name = "bar")
config_setting(
    name = "excelsior",
    values = {"define": "species=excelsior"},
)
config_setting(
    name = "americana",
    values = {"define": "species=americana"},
)
EOF

  bazel cquery "deps(//$pkg:ash)" --define species=excelsior  > output \
    2>"$TEST_log" || fail "Excepted success"
  assert_contains "//$pkg:foo" output
  assert_not_contains "//$pkg:bar" output
}

function test_empty_results_printed() {
  local -r pkg=$FUNCNAME
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
sh_library(name='redwood', deps=[':sequoia',':sequoiadendron'])
sh_library(name='sequoia')
sh_library(name='sequoiadendron')
EOF

  bazel cquery "somepath(//$pkg:sequoia,//$pkg:sequoiadendron)" \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "INFO: Empty query results"
  assert_not_contains "//$pkg:sequoiadendron" output
}

function test_universe_scope_specified() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  # The java_library rule has a host transition on its plugins attribute.
  bazel cquery //$pkg:dep+//$pkg:plugin --universe_scope=//$pkg:my_java \
    > output 2>"$TEST_log" || fail "Excepted success"

  # Find the lines of output for //$pkg:plugin and //$pkg:dep.
  PKG_HOST=$(grep "//$pkg:plugin" output)
  PKG_TARGET=$(grep "//$pkg:dep" output)
  # Trim to just configurations.
  HOST_CONFIG=${PKG_HOST/"//$pkg:plugin"}
  TARGET_CONFIG=${PKG_TARGET/"//$pkg:dep"}
  # Ensure they are are not equal.
  assert_not_equals $HOST_CONFIG $TARGET_CONFIG
}

function test_host_config_output() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  bazel cquery //$pkg:plugin --universe_scope=//$pkg:my_java \
    > output 2>"$TEST_log" || fail "Excepted success"

  assert_contains "//$pkg:plugin (HOST)" output
}

function test_transitions_lite() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  bazel cquery "deps(//$pkg:my_java)" --transitions=lite \
    > output 2>"$TEST_log" || fail "Excepted success"

  assert_contains "//$pkg:my_java" output
  assert_contains "plugins#//$pkg:plugin#HostTransition" output
}


function test_transitions_full() {
  local -r pkg=$FUNCNAME
  write_java_library_build $pkg

  bazel cquery "deps(//$pkg:my_java)" --transitions=full \
    > output 2>"$TEST_log" || fail "Excepted success"

  assert_contains "//$pkg:my_java" output
  assert_contains "plugins#//$pkg:plugin#HostTransition" output
}

function write_java_library_build() {
  local -r pkg=$1
  mkdir -p $pkg
  cat > $pkg/BUILD <<EOF
java_library(
    name = "my_java",
    srcs = ['foo.java'],
    deps = [":dep"],
    plugins = [":plugin"]
)
java_library(name = "dep")
java_plugin(name = "plugin")
EOF
}

run_suite "${PRODUCT_NAME} configured query tests"
