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
# Third-party license checking has been removed from Bazel. See
# https://github.com/bazelbuild/bazel/issues/7444 for details and
# alternatives.
#
# This test just checks that --incompatible_disable_third_party_license_checking
# is a no-op. We can remove it when that flag is removed.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function write_rules {
  mkdir -p third_party/restrictive
  cat <<EOF > third_party/restrictive/BUILD
licenses(['restricted'])
genrule(
    name = 'lib',
    srcs = [],
    outs = ['lib.out'],
    cmd = 'echo hi > $@',
    licenses = ['restricted'],
    visibility = ['//bad:__pkg__']
)
exports_files(['file'])
EOF

  mkdir -p bad
  cat <<EOF > bad/BUILD
distribs(['client'])
genrule(
    name = 'main',
    srcs = [],
    outs = ['bad.out'],
    tools = ['//third_party/restrictive:lib'],
    cmd = 'echo hi > $@'
)
EOF

  mkdir -p third_party/missing_license
  cat <<EOF > third_party/missing_license/BUILD
genrule(
    name = 'lib',
    srcs = [],
    outs = ['lib.out'],
    cmd = 'echo hi > $@'
)
exports_files(['file'])
EOF
}

FLAG_MODES="--incompatible_disable_third_party_license_checking=true \
  --incompatible_disable_third_party_license_checking=false"

function test_license_enforcement_violation {  # Bazel should ignore this.
  create_new_workspace
  write_rules
  for flag_mode in $FLAG_MODES; do
    bazel build --nobuild //bad:main $flag_mode \
      >& $TEST_log || fail "build should succeed"
  done
}

function test_check_licenses_flag { # Bazel should ignore this.
  create_new_workspace
  write_rules
  for flag_mode in $FLAG_MODES; do
    bazel build --nobuild //bad:main --check_licenses $flag_mode \
      >& $TEST_log || fail "build should succeed"
  done
}

function test_third_party_no_license { # Bazel should ignore this.
  create_new_workspace
  write_rules
  for flag_mode in $FLAG_MODES; do
    bazel build --nobuild  //third_party/missing_license:lib $flag_mode \
      >& $TEST_log || fail "build should succeed"
  done
}

run_suite "license checking tests"

