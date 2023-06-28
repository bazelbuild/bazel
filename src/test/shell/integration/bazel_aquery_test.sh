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
  declare -r is_macos=false
  declare -r is_windows=true
  ;;
darwin)
  declare -r is_macos=true
  declare -r is_windows=false
  ;;
*)
  declare -r is_macos=false
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

add_to_bazelrc "build --package_path=%workspace%"


function test_repo_mapping_manifest() {
  local pkg="${FUNCNAME[0]}"
  local pkg2="${FUNCNAME[0]}_pkg2"
  mkdir -p "$pkg" || fail "mkdir -p $pkg"
  cat > "$pkg/WORKSPACE" <<EOF
local_repository(
    name = "pkg2",
    path = "../$pkg2",
)
EOF
  touch "$pkg/foo.cpp"
  cat > "$pkg/BUILD" <<EOF
cc_binary(name = "foo",
          srcs = ["foo.cpp"],
          deps = ["@pkg2//:bar"]
)
EOF
  mkdir -p "$pkg2" || fail "mkdir -p $pkg2"
  touch "$pkg2/WORKSPACE"
  touch "$pkg2/bar.cpp"
  cat > "$pkg2/BUILD" <<EOF
cc_binary(name = "bar",
          srcs = ["bar.cpp"],
          visibility=["//visibility:public"],
)
EOF
  cd $pkg
  bazel aquery --output=textproto --include_file_write_contents \
     "//:foo" >output 2> "$TEST_log" || fail "Expected success"
  cat output >> "$TEST_log"
  assert_contains "^file_contents:.*pkg2,__main__,__main__" output

  bazel aquery --output=text --include_file_write_contents "//:foo" | \
    sed -nr '/Mnemonic: RepoMappingManifest/,/^ *$/p' >output \
      2> "$TEST_log" || fail "Expected success"
  cat output >> "$TEST_log"
  assert_contains "^ *FileWriteContents: \[.*\]" output
  # Verify file contents if we can decode base64-encoded data.
  if which base64 >/dev/null; then
    sed -nr 's/^ *FileWriteContents: \[(.*)\]/echo \1 | base64 -d/p' output | \
       sh | tee -a "$TEST_log"  | assert_contains "pkg2,__main__,__main__" -
  fi
}

run_suite "${PRODUCT_NAME} action graph query tests"
