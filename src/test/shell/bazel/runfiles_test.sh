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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Make sure runfiles are created under a custom-named subdirectory when
# workspace() is specified in the WORKSPACE file.
function test_runfiles() {
  name="blorp_malorp"
  create_workspace_with_default_repos WORKSPACE "$name"

  mkdir foo
  cat > foo/BUILD <<EOF
java_test(
    name = "foo",
    srcs = ["Noise.java"],
    test_class = "Noise",
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

function test_legacy_runfiles_change() {
  create_workspace_with_default_repos WORKSPACE foo
  cat >> WORKSPACE <<EOF
new_local_repository(
    name = "bar",
    path = ".",
    build_file = "BUILD",
)
EOF
  cat > BUILD <<EOF
exports_files(glob(["*"]))

cc_binary(
    name = "thing",
    srcs = ["thing.cc"],
    data = ["@bar//:thing.cc"],
)
EOF
  cat > thing.cc <<EOF
int main() { return 0; }
EOF
  bazel build --legacy_external_runfiles //:thing &> $TEST_log \
    || fail "Build failed"
  [[ -d bazel-bin/thing.runfiles/foo/external/bar ]] \
    || fail "bar not found"

  bazel build --nolegacy_external_runfiles //:thing &> $TEST_log \
    || fail "Build failed"
  [[ ! -d bazel-bin/thing.runfiles/foo/external/bar ]] \
    || fail "Old bar still found"

  bazel build --legacy_external_runfiles //:thing &> $TEST_log \
    || fail "Build failed"
  [[ -d bazel-bin/thing.runfiles/foo/external/bar ]] \
    || fail "bar not recreated"
}

# Test that the local strategy creates a runfiles tree during test if no --nobuild_runfile_links
# is specified.
function test_nobuild_runfile_links() {
  mkdir data && echo "hello" > data/hello && echo "world" > data/world
  create_workspace_with_default_repos WORKSPACE foo

cat > test.sh <<'EOF'
#!/bin/bash
set -e
[[ -f ${RUNFILES_DIR}/foo/data/hello ]]
[[ -f ${RUNFILES_DIR}/foo/data/world ]]
exit 0
EOF
  chmod 755 test.sh
  cat > BUILD <<'EOF'
filegroup(
  name = "runfiles",
  srcs = ["data/hello", "data/world"],
)

sh_test(
  name = "test",
  srcs = ["test.sh"],
  data = [":runfiles"],
)
EOF

  bazel build --spawn_strategy=local --nobuild_runfile_links //:test \
    || fail "Building //:test failed"

  [[ ! -f bazel-bin/test.runfiles/foo/data/hello ]] || fail "expected no runfile data/hello"
  [[ ! -f bazel-bin/test.runfiles/foo/data/world ]] || fail "expected no runfile data/world"
  [[ ! -f bazel-bin/test.runfiles/MANIFEST ]] || fail "expected output manifest to not exist"

  bazel test --spawn_strategy=local --nobuild_runfile_links //:test \
    || fail "Testing //:foo failed"

  [[ -f bazel-bin/test.runfiles/foo/data/hello ]] || fail "expected runfile data/hello to exist"
  [[ -f bazel-bin/test.runfiles/foo/data/world ]] || fail "expected runfile data/world to exist"
  [[ -f bazel-bin/test.runfiles/MANIFEST ]] || fail "expected output manifest to exist"
}

run_suite "runfiles tests"
