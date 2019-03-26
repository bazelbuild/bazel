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
#
# An end-to-end test that Bazel's experimental UI produces reasonable output.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

test_broken_BUILD_files_ignored() {
    rm -rf work && mkdir work && cd work
    touch WORKSPACE
    mkdir -p ignoreme/deep/reallydep/stillignoreme
    echo This is a broken BUILD file > ignoreme/BUILD
    echo This is a broken BUILD file > ignoreme/deep/BUILD
    echo This is a broken BUILD file > ignoreme/deep/reallydep/BUILD
    echo This is a broken BUILD file > ignoreme/deep/reallydep/stillignoreme/BUILD
    touch BUILD
    bazel build ... && fail "Expected failure" || :

    echo; echo
    echo ignoreme > .bazelignore
    bazel build ... \
        || fail "directory mentioned in .bazelignore not ignored as it should"
}

test_symlink_loop_ignored() {
    rm -rf work && mkdir work && cd work
    touch WORKSPACE
    mkdir -p ignoreme/deep
    (cd ignoreme/deep && ln -s . loop)
    touch BUILD
    bazel build ... && fail "Expected failure" || :

    echo; echo
    echo ignoreme > .bazelignore
    bazel build ... \
        || fail "directory mentioned in .bazelignore not ignored as it should"
}

test_build_specific_target() {
    rm -rf work && mkdir work && cd work
    touch WORKSPACE
    mkdir -p ignoreme
    echo Not a valid BUILD file > ignoreme/BUILD
    mkdir -p foo/bar
    cat > foo/bar/BUILD <<'EOI'
genrule(
  name = "out",
  outs = ["out.txt"],
  cmd = "echo Hello World > $@",
)
EOI
    echo ignoreme > .bazelignore
    bazel build //foo/bar/... || fail "Could not build valid target"
}

test_aquery_specific_target() {
    rm -rf work && mkdir work && cd work
    touch WORKSPACE
    mkdir -p foo/ignoreme
    cat > foo/ignoreme/BUILD <<'EOI'
genrule(
  name = "ignoreme",
  outs = ["ignore.txt"],
  cmd = "echo Hello World > $@",
)
EOI
    mkdir -p foo
    cat > foo/BUILD <<'EOI'
genrule(
  name = "out",
  outs = ["out.txt"],
  cmd = "echo Hello World > $@",
)
EOI
    bazel aquery ... > output 2> "$TEST_log" \
        || fail "Aquery should complete without error."
    cat output >> "$TEST_log"
    assert_contains "ignoreme" output

    echo foo/ignoreme > .bazelignore
    bazel aquery ... > output 2> "$TEST_log" \
        || fail "Aquery should complete without error."
    cat output >> "$TEST_log"
    assert_not_contains "ignoreme" output
}

run_suite "Integration tests for .bazelignore"
