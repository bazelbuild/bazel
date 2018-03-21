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

# Set up
source "src/test/shell/integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### TESTS #############################################################

function test_basic() {
  rm -rf peach
  mkdir -p peach
  cat > peach/BUILD <<EOF
sh_library(name='brighton', deps=[':harken'])
sh_library(name='harken')
EOF
  touch WORKSPACE

  bazel info > /dev/null # unpack and create install and output roots.
  ${BAZEL_RUNFILES}/src/tools/package_printer/java/com/google/devtools/build/packageprinter/BazelPackagePrinter --workspace_root=. --install_base=$(bazel info install_base) --output_base=$(bazel info output_base) peach > $TEST_log

  expect_log "//peach:brighton"
  expect_log "//peach:harken"
}

function tear_down() {
  bazel shutdown
}

run_suite "BazelPackagePrinter tests"
