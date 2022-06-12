#!/bin/bash -eu
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

# This integration test exists so that we can run our Starlark tests
# for cc_import with Bazel built from head. Once the Stararlark
# implementation can rely on release Bazel, we can add the tests directly.

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${TEST_SRCDIR}/io_bazel/src/test/shell/integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/builtin_test_setup.sh" \
  || { echo "builtin_test_setup.sh not found!" >&2; exit 1; }

function test_starlark_cc() {
  setup_tests src/main/starlark/tests/builtins_bzl/cc

  cat >> WORKSPACE<<EOF
local_repository(
    name = "test_repo",
    path = "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library2",
)
EOF

  bazel test --define=is_bazel=true --test_output=streamed \
    --experimental_link_static_libraries_once \
    --experimental_enable_target_export_check --experimental_cc_shared_library \
    //src/main/starlark/tests/builtins_bzl/cc/... || fail "expected success"
}

run_suite "cc_* built starlark test"
