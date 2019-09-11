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

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
  export EXT=".exe"
else
  export EXT=""
fi

# Make sure runfiles are created under a custom-named subdirectory when
# workspace() is specified in the WORKSPACE file.
function test_runfiles() {

  name=blorp_malorp
  cat > WORKSPACE <<EOF
workspace(name = "$name")
EOF
  create_workspace_with_default_repos WORKSPACE

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

  bazel build --enable_runfiles=yes //foo:foo >& $TEST_log || fail "Build failed"
  [[ -d "bazel-bin/foo/foo${EXT}.runfiles/$name" ]] || fail "$name runfiles directory not created"
  [[ -d "bazel-bin/foo/foo${EXT}.runfiles/$name/foo" ]] || fail "No foo subdirectory under $name"
  [[ -x "bazel-bin/foo/foo${EXT}.runfiles/$name/foo/foo" ]] || fail "No foo executable under $name"
}

function test_legacy_runfiles_change() {
  cat > WORKSPACE <<EOF
workspace(name = "foo")

new_local_repository(
    name = "bar",
    path = ".",
    build_file = "BUILD",
)
EOF
  create_workspace_with_default_repos WORKSPACE
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
  bazel build --enable_runfiles=yes --legacy_external_runfiles //:thing &> $TEST_log \
    || fail "Build failed"
  [[ -d bazel-bin/thing${EXT}.runfiles/foo/external/bar ]] \
    || fail "bar not found"

  bazel build --enable_runfiles=yes --nolegacy_external_runfiles //:thing &> $TEST_log \
    || fail "Build failed"
  [[ ! -d bazel-bin/thing${EXT}.runfiles/foo/external/bar ]] \
    || fail "Old bar still found"

  bazel build --enable_runfiles=yes --legacy_external_runfiles //:thing &> $TEST_log \
    || fail "Build failed"
  [[ -d bazel-bin/thing${EXT}.runfiles/foo/external/bar ]] \
    || fail "bar not recreated"
}

# Test that the local strategy creates a runfiles tree during test if no --nobuild_runfile_links
# is specified.
function test_nobuild_runfile_links() {
  mkdir data && echo "hello" > data/hello && echo "world" > data/world
    cat > WORKSPACE <<EOF
workspace(name = "foo")
EOF

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

  [[ ! -f "bazel-bin/test${EXT}.runfiles/foo/data/hello" ]] || fail "expected no runfile data/hello"
  [[ ! -f "bazel-bin/test${EXT}.runfiles/foo/data/world" ]] || fail "expected no runfile data/world"
  [[ ! -f "bazel-bin/test${EXT}.runfiles/MANIFEST" ]] || fail "expected output manifest to not exist"

  bazel test --spawn_strategy=local --nobuild_runfile_links //:test \
    || fail "Testing //:foo failed"

  [[ -f "bazel-bin/test${EXT}.runfiles/foo/data/hello" ]] || fail "expected runfile data/hello to exist"
  [[ -f "bazel-bin/test${EXT}.runfiles/foo/data/world" ]] || fail "expected runfile data/world to exist"
  [[ -f "bazel-bin/test${EXT}.runfiles/MANIFEST" ]] || fail "expected output manifest to exist"
}

function test_space_in_runfile_path() {
  mkdir -p 'foo/a a'         # one space
  mkdir -p 'repo/b b/c   c'  # more than one space
  touch 'foo/x.txt'          # no space
  touch 'foo/a a/y.txt'      # space in runfile link and target path
  touch 'repo/b b/z.txt'     # space in target path only
  touch 'repo/b b/c   c/w.txt' # space in runfile link and target path in ext.repo
  touch 'repo/b b/WORKSPACE'
  cat >'repo/b b/BUILD' <<'eof'
filegroup(
    name = "files",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
eof
  cat >WORKSPACE <<'eof'
workspace(name = "foo_ws")
local_repository(
    name = "space_ws",
    path = "repo/b b",
)
eof
  cat >BUILD <<'eof'
sh_binary(
    name = "x",
    srcs = ["x.sh"],
    data = ["@space_ws//:files"] + glob(["foo/**"]),
    deps = ["@bazel_tools//tools/bash/runfiles"],
)
eof
  cat >x.sh <<'eof'
#!/bin/bash
# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---
echo "Hello from x.sh"
echo "x=($(rlocation "foo_ws/foo/x.txt"))"
echo "y=($(rlocation "foo_ws/foo/a a/y.txt"))"
echo "z=($(rlocation "space_ws/z.txt"))"
echo "w=($(rlocation "space_ws/c   c/w.txt"))"
eof
  chmod +x x.sh

  # Look up runfiles using the runfiles manifest
  bazel run --enable_runfiles=yes --nobuild_runfile_links //:x >&"$TEST_log"
  expect_log "Hello from x.sh"
  expect_log "^x=(.*foo/x.txt)"
  expect_log "^y=(.*foo/a a/y.txt)"
  expect_log "^z=(.*space_ws/z.txt)"
  expect_log "^w=(.*space_ws/c   c/w.txt)"

  # See if runfiles links are generated
  bazel build --enable_runfiles=yes --build_runfile_links //:x >&"$TEST_log"
  [[ -e "bazel-bin/x${EXT}.runfiles/foo_ws/foo/x.txt" ]] || fail "Cannot find x.txt"
  [[ -e "bazel-bin/x${EXT}.runfiles/foo_ws/foo/a a/y.txt" ]] || fail "Cannot find y.txt"
  [[ -e "bazel-bin/x${EXT}.runfiles/space_ws/z.txt" ]] || fail "Cannot find z.txt"
  [[ -e "bazel-bin/x${EXT}.runfiles/space_ws/c   c/w.txt" ]] || fail "Cannot find w.txt"
}

run_suite "runfiles tests"
