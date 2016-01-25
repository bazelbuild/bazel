#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
# Test runfiles creation
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

# Make sure runfiles are created under a custom-named subdirectory when
# workspace() is specified in the WORKSPACE file.
function test_runfiles() {

  name=blorp_malorp
  cat > WORKSPACE <<EOF
workspace(name = "$name")

EOF

  mkdir foo
  cat > foo/BUILD <<EOF
java_test(
    name = "foo",
    srcs = ["Noise.java"],
    main_class = "Noise",
)
EOF
  cat > foo/Noise.java <<EOF
public class Noise {
  public static void main(String[] args) {
    System.err.println(System.getenv("I'm a test."));
  }
}
EOF

  bazel build //foo:foo >& $TEST_log || fail "Build failed"
  [[ -d bazel-bin/foo/foo.runfiles/$name ]] || fail "$name runfiles directory not created"
  [[ -d bazel-bin/foo/foo.runfiles/$name/foo ]] || fail "No foo subdirectory under $name"
  [[ -x bazel-bin/foo/foo.runfiles/$name/foo/foo ]] || fail "No foo executable under $name"
}

run_suite "runfiles tests"
