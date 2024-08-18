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

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

function _create_pkg() {
  # Define dummy CPU and OS constraints (cpu1 and cpu2, and os1 and os2).
  # Also define platforms for each combination.
  mkdir "platforms"
  cat > "platforms/BUILD" <<'eof'
package(default_visibility = ["//visibility:public"])

[constraint_value(
    name = "cpu%d" % i,
    constraint_setting = "@platforms//cpu",
) for i in [1, 2]]

[constraint_value(
    name = "os%d" % i,
    constraint_setting = "@platforms//os",
) for i in [1, 2]]

[platform(
    name = "cpu%d_os%d" % (i, j),
    constraint_values = [":cpu%d" % i, ":os%d" % j],
) for i in [1, 2] for j in [1, 2]]
eof

  # Define a windows_resource_compiler_toolchain with our dummy CPUs and OSs.
  mkdir "toolchains"
  cat > "toolchains/BUILD" <<eof
package(default_visibility = ["//visibility:public"])

load(
    "//src/main/res:winsdk_toolchain.bzl",
    "windows_resource_compiler_toolchain",
    "WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE",
)

windows_resource_compiler_toolchain(
    name = "tc_def",
    rc_exe = ":rc-gen",
)

# Need a genrule because of https://github.com/bazelbuild/bazel/issues/9023
genrule(
    name = "rc-gen",
    outs = ["rc-gen.bat"],
    srcs = ["rc-src.bat"],
    executable = True,
    cmd = "cp $< \$@",
)

toolchain(
    name = "tc",
    exec_compatible_with = ["//platforms:cpu1", "//platforms:os1"],
    target_compatible_with = ["//platforms:cpu2"],
    toolchain = ":tc_def",
    toolchain_type = WINDOWS_RESOURCE_COMPILER_TOOLCHAIN_TYPE,
    visibility = ["//visibility:public"],
)
eof

  # On Windows, we write a Batch script. On other platforms, a shell script.
  # File extension must be .bat on Windows, and it doesn't matter on other
  # platforms.
  if "$is_windows"; then
    cat > "toolchains/rc-src.bat" <<'eof'
@echo off
setlocal enabledelayedexpansion

for %%i in (%*) do (
  set j=%%i
  if "!j:~0,3!" == "/fo" (set out=!j:~3!)
  if "!j:~-2!" == "rc" (set src=!j!)
)
set out=%out:/=\%
echo out=%out%

echo src=%src:/=\%>%out%
echo out=%out%>>%out%
dir /s /b /o:n *.dat>>%out%
eof
  else
    cat > "toolchains/rc-src.bat" <<'eof'
#!/bin/bash
for a in $*; do
  if [[ "$a" =~ /fo.* ]]; then
    out="${a#/fo}"
  elif [[ "$a" =~ .*rc$ ]]; then
    src="$a"
  fi
done

cat > "$out" <<EOF
src=$src
out=$out
$(find . -name '*.dat' | sort)
EOF
eof
  fi
  chmod +x "toolchains/rc-src.bat"

  # Define a windows_resources rule we'll try to build with various exec and
  # target platform combinations.
  cat > "BUILD" <<'eof'
load("//src/main/res:win_res.bzl", "windows_resources")

windows_resources(
    name = "res",
    rc_files = ["foo.rc", "bar.rc"],
    resources = ["res1.dat", "res2.dat"],
)
eof

  touch foo.rc bar.rc res1.dat res2.dat
}

function _symlink_res_toolchain_files() {
  mkdir -p "src/main/res"
  for f in BUILD win_res.bzl winsdk_configure.bzl winsdk_toolchain.bzl; do
    real="$(rlocation io_bazel/src/main/res/$f)"
    ln -sf "$real" "src/main/res/$f"
  done
}

function _assert_outputs() {
  [[ -e bazel-bin/foo.res ]] || fail "missing output"
  grep -q "src=foo.rc" "bazel-bin/foo.res" || fail "bad output"
  grep -q "out=.*\bfoo.res" "bazel-bin/foo.res" || fail "bad output"
  grep -q ".*\bres1.dat" "bazel-bin/foo.res" || fail "bad output"
  grep -q ".*\bres2.dat" "bazel-bin/foo.res" || fail "bad output"

  [[ -e bazel-bin/bar.res ]] || fail "missing output"
  grep -q "src=bar.rc" "bazel-bin/bar.res" || fail "bad output"
  grep -q "out=.*\bbar.res" "bazel-bin/bar.res" || fail "bad output"
  grep -q ".*\bres1.dat" "bazel-bin/bar.res" || fail "bad output"
  grep -q ".*\bres2.dat" "bazel-bin/bar.res" || fail "bad output"
}

function _assert_no_outputs() {
  [[ -e bazel-bin/foo.res ]] && fail "unexpected output" || true
  [[ -e bazel-bin/bar.res ]] && fail "unexpected output" || true
}

function test_toolchain_selection() {
  echo "module(name = 'io_bazel')" >> MODULE.bazel
  add_platforms "MODULE.bazel"
  _symlink_res_toolchain_files
  _create_pkg

  # (1) Expect success: host platform matches toolchain.
  bazel build //:res --host_platform=//platforms:cpu1_os1 \
    --platforms=//platforms:cpu2_os1 --extra_toolchains=//toolchains:tc \
    || fail "expected success"
  _assert_outputs

  # (2) Expect failure: same as (1) but without a registered toolchain.
  bazel clean
  bazel build //:res --host_platform=//platforms:cpu1_os1 \
    --platforms=//platforms:cpu2_os1 \
    && fail "expected failure" || true
  _assert_no_outputs

  # (3) Expect failure: same as (1) but host platform does not match toolchain
  bazel build //:res --host_platform=//platforms:cpu2_os1 \
    --platforms=//platforms:cpu2_os1 --extra_toolchains=//toolchains:tc \
    && fail "expected failure" || true
  _assert_no_outputs

  # (4) Expect failure: same as (3) with an extra execution platform, but the
  # toolchain doesn't match.
  bazel build //:res --host_platform=//platforms:cpu2_os1 \
    --platforms=//platforms:cpu2_os1 --extra_toolchains=//toolchains:tc \
    --extra_execution_platforms=//platforms:cpu2_os2 \
    && fail "expected failure" || true
  _assert_no_outputs

  # (5) Expect success: same as (4) with a matching extra execution platform.
  bazel build //:res --host_platform=//platforms:cpu2_os1 \
    --platforms=//platforms:cpu2_os1 --extra_toolchains=//toolchains:tc \
    --extra_execution_platforms=//platforms:cpu2_os2,//platforms:cpu1_os1 \
    || fail "expected success"
  _assert_outputs
}

run_suite "Tests for windows_resources() rule and toolchain selection"
