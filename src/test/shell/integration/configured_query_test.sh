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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

add_to_bazelrc "build --package_path=%workspace%"

#### TESTS #############################################################

function test_basic_query() {
  rm -rf maple
  mkdir -p maple
  cat > maple/BUILD <<EOF
sh_library(name='maple', deps=[':japanese'])
sh_library(name='japanese')
EOF

 bazel cquery 'deps(//maple)' > output 2>"$TEST_log" || fail "Expected success"

 assert_contains "//maple:maple" output
 assert_contains "//maple:japanese" output
}

function test_respects_selects() {
 rm -rf ash
 mkdir -p ash
 cat > ash/BUILD <<EOF
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

  bazel cquery 'deps(//ash)' --define species=excelsior  > output \
    2>"$TEST_log" || fail "Excepted success"
  assert_contains "//ash:foo" output
  assert_not_contains "//ash:bar" output
}

function test_empty_results_printed() {
  rm -rf redwood
  mkdir -p redwood
  cat > redwood/BUILD <<EOF
sh_library(name='redwood', deps=[':sequoia',':sequoiadendron'])
sh_library(name='sequoia')
sh_library(name='sequoiadendron')
EOF

  bazel cquery 'somepath(//redwood:sequoia,//redwood:sequoiadendron)' \
    > output 2>"$TEST_log" || fail "Expected success"

  expect_log "INFO: Empty query results"
  assert_not_contains "//redwood:sequoiadendreon" output
}

function test_universe_scope_specified() {
  write_java_library_build

  # The java_library rule has a host transition on its plugins attribute.
  bazel cquery //pine:dep+//pine:plugin --universe_scope=//pine:my_java \
    > output 2>"$TEST_log" || fail "Excepted success"

  # Find the lines of output for //pine:plugin and //pine:dep.
  PINE_HOST=$(grep "//pine:plugin" output)
  PINE_TARGET=$(grep "//pine:dep" output)
  # Trim to just configurations.
  HOST_CONFIG=${PINE_HOST/"//pine:plugin"}
  TARGET_CONFIG=${PINE_TARGET/"//pine:dep"}
  # Ensure they are are not equal.
  assert_not_equals $HOST_CONFIG $TARGET_CONFIG
}

function test_host_config_output() {
 write_java_library_build

 bazel cquery //pine:plugin --universe_scope=//pine:my_java \
   > output 2>"$TEST_log" || fail "Excepted success"

 assert_contains "//pine:plugin (HOST)" output
}

function test_transitions_lite() {
 write_java_library_build

 bazel cquery "deps(//pine:my_java)" --transitions=lite \
   > output 2>"$TEST_log" || fail "Excepted success"

 assert_contains "//pine:my_java" output
 assert_contains "plugins#//pine:plugin#HostTransition" output
}


function test_transitions_full() {
 write_java_library_build

 bazel cquery "deps(//pine:my_java)" --transitions=full \
   > output 2>"$TEST_log" || fail "Excepted success"

 assert_contains "//pine:my_java" output
 assert_contains "plugins#//pine:plugin#HostTransition" output
}

function write_java_library_build() {
  rm -rf pine
  mkdir -p pine
  cat > pine/BUILD <<EOF
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

function tear_down() {
  bazel shutdown
}

run_suite "${PRODUCT_NAME} configured query tests"
