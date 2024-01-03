#!/usr/bin/env bash

# Copyright 2023 The Bazel Authors. All rights reserved.
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

# Benchmarks prebuilt Turbine shipped in java_tools and Turbine built from HEAD on the following
# scenario:
# 1. Build the Bazel server jar in a full non-incremental build (disk cache +
#    --experimental_java_classpath=bazel).
# 2. Add a public method to Label.java.
# 3. Benchmark the incremental build, which consists of both Java compile and header compile
#    actions.
#
# The profiles for the last run of each benchmark are kept in the root directory.
#
# Requires the following tools to be installed on the host:
# - hyperfine (https://github.com/sharkdp/hyperfine)
# - buildozer (https://github.com/bazelbuild/buildtools/tree/master/buildozer)

# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

# Disable MSYS path conversion, which can otherwise mess up Bazel labels.
export MSYS2_ARG_CONV_EXCL='*'
export MSYS_NO_PATHCONV=1

RUNS=5
TMPDIR="${TMPDIR:-/tmp}"

export DISK_CACHE="$TMPDIR/turbine_benchmark_disk_cache"
export BAZEL_TARGET='//src/main/java/com/google/devtools/build/lib/bazel:BazelServer'
export FILE_TO_MODIFY='src/main/java/com/google/devtools/build/lib/cmdline/Label.java'

function reset_file() {
  git checkout -- "$FILE_TO_MODIFY"
}
export -f reset_file

function reset_and_modify_file() {
  reset_file
  # Add a public method with a unique name to the file to ensure that the incremental build is
  # not using any cached results.
  sed -i"" -e "s/^}/  public void foo$(uuid | tr '-' '_')() {}\n}/g" "$FILE_TO_MODIFY"
}
export -f reset_and_modify_file

function build_target() {
  # Keep the last profile around after a benchmark
  bazel build --experimental_java_classpath=bazel \
    --disk_cache="$DISK_CACHE" \
    --extra_toolchains=//_turbine_benchmark:turbine_benchmark_toolchain_definition \
    --profile="${PROFILE_PATH:-}" \
    $BAZEL_TARGET
}
export -f build_target

mkdir -p "$DISK_CACHE"
mkdir -p "$BUILD_WORKSPACE_DIRECTORY/_turbine_benchmark"
TURBINE_PATH="$(rlocation "$1")"
if [[ ! "$TURBINE_PATH" =~ "-opt/" ]]; then
  echo "ERROR: benchmark must be executed with -c opt"
  exit 1
fi
cp "$TURBINE_PATH" "$BUILD_WORKSPACE_DIRECTORY/_turbine_benchmark/turbine"
# Change directory only after accessing runfiles as rlocation may returns paths relative to the
# initial working directory.
cd "$BUILD_WORKSPACE_DIRECTORY"

function cleanup() {
  rm -rf _turbine_benchmark
  reset_file
  # Do not delete the disk cache to avoid rebuilding Bazel every time the script is executed.
}
trap cleanup EXIT INT TERM

function run_benchmark() {
  # Cache a full non-incremental build.
  build_target
  # Change a file and benchmark the incremental build.
  hyperfine --shell=bash --warmup 1 --runs $RUNS --prepare reset_and_modify_file build_target
}

echo "===== Benchmarking prebuilt Turbine ====="
cat << 'EOF' > _turbine_benchmark/BUILD
load("@rules_java//toolchains:default_java_toolchain.bzl", "default_java_toolchain")
default_java_toolchain(
    name = "turbine_benchmark_toolchain",
    source_version = "11",
    target_version = "11",
    # Work around for https://github.com/bazelbuild/bazel/issues/19837.
    bootclasspath = ["@rules_java//toolchains:platformclasspath"],
)
EOF
export PROFILE_PATH=with_prebuilt_turbine.profile.gz
run_benchmark

echo "===== Benchmarking Turbine built from HEAD ====="
buildozer 'add header_compiler_direct turbine' //_turbine_benchmark:turbine_benchmark_toolchain
export PROFILE_PATH=with_head_turbine.profile.gz
run_benchmark
