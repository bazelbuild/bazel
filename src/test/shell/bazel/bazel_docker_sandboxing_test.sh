#!/usr/bin/env bash
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

# Load the test setup defined in the parent directory
source $(rlocation io_bazel/src/test/shell/integration_test_setup.sh) \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Platform container is picked up if it specified.
# --remote_default_exec_properties is ignored in this case
function test_platform_container() {
  mkdir -p t
  cat >t/BUILD <<'EOF'
platform(
  name = "bad_docker",
  remote_execution_properties = """
    properties:{
      name: "container-image"
      value: "docker://bad_platform_container"
    }"""
)

genrule(
  name = "echo",
  outs = ["out.txt"],
  cmd = "echo hello > $@",
)
EOF

  bazel build  \
    --extra_execution_platforms=//t:bad_docker \
    --experimental_enable_docker_sandbox --experimental_docker_verbose \
    --spawn_strategy=docker \
    --remote_default_exec_properties="container-image=docker://bad_flag_container" \
    //t:echo \
    &> $TEST_log && fail "Expected build to fail, it succeeded"
  grep "bad_platform_container" $TEST_log || fail "Wrong container was chosen"
}

# If platform container is not specified, --remote_default_exec_properties
# are picked up
function test_flag_container() {
  mkdir -p t
  cat >t/BUILD <<'EOF'
genrule(
  name = "echo",
  outs = ["out.txt"],
  cmd = "echo hello > $@",
)
EOF

  bazel build  \
    --experimental_enable_docker_sandbox --experimental_docker_verbose \
    --spawn_strategy=docker \
    --remote_default_exec_properties="container-image=docker://bad_flag_container" \
    //t:echo \
    &> $TEST_log && fail "Expected build to fail, it succeeded"
  grep "bad_flag_container" $TEST_log || fail "Wrong container was chosen"
}

run_suite "bazel docker sandboxing test"
