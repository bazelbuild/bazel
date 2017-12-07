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
# configured_query_test.sh: integration tests for bazel configured query

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

add_to_bazelrc "build --package_path=%workspace%"

#### TESTS #############################################################

function do_the_thing() {
 rm -rf maple
 mkdir -p maple
 cat > maple/BUILD <<EOF
sh_library(name='maple', deps=[':japanese'])
sh_library(name='japanese')
EOF

 bazel build --nobuild //maple:japanese \
   --experimental_post_build_query='deps(//maple)' 1>output 2>"$TEST_log" \
   || fail "Expected success"

 expect_log "//maple:maple"
 expect_log "//maple:japanese"
}

function test_empty_results_printed() {
  rm -rf redwood
  mkdir -p redwood
  cat > redwood/BUILD <<EOF
sh_library(name='redwood', deps=[':sequoia',':sequoiadendron'])
sh_library(name='sequoia')
sh_library(name='sequoiadendron')
EOF

  bazel build --nobuild //redwood \
    --experimental_post_build_query='somepath(//redwood:sequoia,//redwood:sequoiadendron)' \
    >& "$TEST_log" || fail "Expected success"
  expect_log "INFO: Empty query results"
  expect_log_not "//redwood:sequoiadendreon"

}


function tear_down() {
  bazel shutdown
}

run_suite "${PRODUCT_NAME} configured query tests"
