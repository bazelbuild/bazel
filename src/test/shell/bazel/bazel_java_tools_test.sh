#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }


function expect_path_in_java_tools() {
  path="$1"; shift

  count=$(zipinfo -1 $(rlocation io_bazel/src/java_tools.zip) | grep -c "path")
  [[ "$count" -gt 0 ]] || fail "Path $path not found in java_tools"
}

function test_java_tools_has_ijar() {
  expect_path_in_java_tools "java_tools/ijar"
}

function test_java_tools_has_ijar_binary() {
  expect_path_in_java_tools "java_tools/ijar/ijar"
}

function test_java_tools_has_zlib() {
  expect_path_in_java_tools "java_tools/zlib"
}

function test_java_tools_has_native_windows() {
  expect_path_in_java_tools "java_tools/src/main/native/windows"
}

function test_java_tools_has_cpp_util() {
  expect_path_in_java_tools "java_tools/src/main/cpp/util"
}

function test_java_tools_has_desugar_deps() {
  expect_path_in_java_tools "java_tools/src/main/protobuf/desugar_deps.proto"
}

function test_java_tools_has_singlejar() {
  expect_path_in_java_tools "java_tools/src/tools/singlejar"
}

function test_java_tools_has_singlejar_local() {
  expect_path_in_java_tools "java_tools/src/tools/singlejar_local"
}

function test_java_tools_has_singlejar_local() {
  expect_path_in_java_tools "java_tools/src/tools/singlejar_local"
}

function test_java_tools_has_VanillaJavaBuilder() {
  expect_path_in_java_tools "java_tools/VanillaJavaBuilder_deploy.jar"
}

function test_java_tools_has_JavaBuilder() {
  expect_path_in_java_tools "java_tools/JavaBuilder_deploy.jar"
}

function test_java_tools_has_turbine_direct() {
  expect_path_in_java_tools "java_tools/turbine_direct_binary_deploy.jar"
}

function test_java_tools_has_turbine_deploy() {
  expect_path_in_java_tools "java_tools/turbine_deploy.jar"
}

function test_java_tools_has_Runner() {
  expect_path_in_java_tools "java_tools/Runner_deploy.jar"
}

function test_java_tools_has_jdk_compiler() {
  expect_path_in_java_tools "java_tools/jdk_compiler"
}

function test_java_tools_has_java_compiler() {
  expect_path_in_java_tools "java_tools/java_compiler"
}

function test_java_tools_has_javac() {
  expect_path_in_java_tools "java_tools/javac-9+181-r4173-1.jar"
}

function test_java_tools_has_jarjar() {
  expect_path_in_java_tools "java_tools/jarjar_command_deploy.jar"
}

function test_java_tools_has_Jacoco() {
  expect_path_in_java_tools "java_tools/JacocoCoverage_jarjar_deploy.jar"
}

function test_java_tools_has_GenClass() {
  expect_path_in_java_tools "java_tools/GenClass_deploy.jar"
}

function test_java_tools_has_ExperimentalRunner() {
  expect_path_in_java_tools "java_tools/ExperimentalRunner_deploy.jar"
}

run_suite "Java tools archive tests"
