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

# Define platforms needed for java integration tests.

set -euo pipefail

# Assumes integration_test_setup.sh was loaded elsewhere (can't load it twice)

function create_java_test_platforms() {
  cat >> "$pkg/jvm/BUILD" <<EOF
constraint_setting(
    name = 'constraint_setting',
)
constraint_value(
    name = 'constraint',
    constraint_setting = ':constraint_setting',
)
toolchain(
    name = 'java_runtime_toolchain',
    toolchain = ':runtime',
    toolchain_type = '@bazel_tools//tools/jdk:runtime_toolchain_type',
    target_compatible_with = [':constraint'],
)
platform(
    name = 'platform',
    parents = ['@platforms//host'],
    constraint_values = [':constraint'],
)
EOF
}

