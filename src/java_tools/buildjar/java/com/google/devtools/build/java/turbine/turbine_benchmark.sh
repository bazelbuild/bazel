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

# Benchmarks prebuilt Turbine shipped in java_tools, Turbine built from HEAD and optionally further
# Turbine binaries on the following scenario:
# 1. Build the Bazel server jar in a full non-incremental build (disk cache +
#    --experimental_java_classpath=bazel).
# 2. Add a public constant to Label.java.
# 3. Benchmark the incremental build, which consists of both Java compile and header compile
#    actions.
#
# Usage:
#   bazel run -c opt //src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_benchmark -- \
#     [--header_jars_only] [--runs=N] [--warmup=N] [[name=]path/to/turbine ...]
#
# Options:
#   --header_jars_only: Only build the header jars of the transitive dependencies of the Bazel
#     server jar. The incremental build then consists only of header compile actions, which makes
#     it more sensitive to Turbine's performance. Note that header compile actions for targets
#     with annotation processors (mnemonic JavacTurbine) don't use the Turbine binary.
#   [name=]path/to/turbine: Additional Turbine binaries to benchmark, e.g. ones built with a
#     different PGO profile or without PGO (see :use_pgo_profile). The name used in the output
#     defaults to the basename of the binary.
#
# The profiles for the last run of each variant are kept in the root directory.
#
# Requires the following tools to be installed on the host:
# - hyperfine (https://github.com/sharkdp/hyperfine)

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
WARMUP=3
HEADER_JARS_ONLY=
TMPDIR="${TMPDIR:-/tmp}"

export DISK_CACHE="$TMPDIR/turbine_benchmark_disk_cache"
export BAZEL_TARGET='//src/main/java/com/google/devtools/build/lib/bazel:BazelServer'
export FILE_TO_MODIFY='src/main/java/com/google/devtools/build/lib/cmdline/Label.java'

HEAD_TURBINE_PATH="$(rlocation "$1")"
shift
if [[ ! "$HEAD_TURBINE_PATH" =~ "-opt/" ]]; then
  echo "ERROR: benchmark must be executed with -c opt"
  exit 1
fi

# The "prebuilt" variant uses the Turbine binary of the default Java toolchain, all other variants
# use the binary at the same index in VARIANT_PATHS.
VARIANTS=(prebuilt head)
VARIANT_PATHS=("" "$HEAD_TURBINE_PATH")
for arg in "$@"; do
  case "$arg" in
    --header_jars_only) HEADER_JARS_ONLY=1 ;;
    --runs=*) RUNS="${arg#--runs=}" ;;
    --warmup=*) WARMUP="${arg#--warmup=}" ;;
    -*) echo "ERROR: unknown option: $arg"; exit 1 ;;
    *)
      if [[ "$arg" == *=* ]]; then
        name="${arg%%=*}"
        path="${arg#*=}"
      else
        name="$(basename "$arg")"
        path="$arg"
      fi
      if [[ ! "$name" =~ ^[A-Za-z0-9_-]+$ ]]; then
        echo "ERROR: invalid variant name '$name', use name=path/to/turbine"
        exit 1
      fi
      for variant in "${VARIANTS[@]}"; do
        if [[ "$variant" == "$name" ]]; then
          echo "ERROR: duplicate variant name '$name', use name=path/to/turbine"
          exit 1
        fi
      done
      if [[ "$path" != /* ]]; then
        path="$BUILD_WORKING_DIRECTORY/$path"
      fi
      if [[ ! -f "$path" ]]; then
        echo "ERROR: $path does not exist"
        exit 1
      fi
      VARIANTS+=("$name")
      VARIANT_PATHS+=("$path")
      ;;
  esac
done

mkdir -p "$DISK_CACHE"
rm -rf "$BUILD_WORKSPACE_DIRECTORY/_turbine_benchmark"
mkdir -p "$BUILD_WORKSPACE_DIRECTORY/_turbine_benchmark"
for i in "${!VARIANTS[@]}"; do
  if [[ -n "${VARIANT_PATHS[$i]}" ]]; then
    cp "${VARIANT_PATHS[$i]}" "$BUILD_WORKSPACE_DIRECTORY/_turbine_benchmark/turbine_${VARIANTS[$i]}"
  fi
done
# Change directory only after accessing runfiles as rlocation may returns paths relative to the
# initial working directory.
cd "$BUILD_WORKSPACE_DIRECTORY"

function reset_file() {
  git checkout -- "$FILE_TO_MODIFY"
}
export -f reset_file

function cleanup() {
  rm -rf _turbine_benchmark
  reset_file
  # Do not delete the disk cache to avoid rebuilding Bazel every time the script is executed.
}
trap cleanup EXIT INT TERM

function reset_and_modify_file() {
  reset_file
  # Add a public constant with a unique name to the file to ensure that the incremental build is
  # not using any cached results. Unlike a method, a constant is also part of the header jar used by
  # downstream header compile actions, so they have to be rerun.
  sed -i"" -e "s/^}/  public static final int FOO_$(uuidgen | tr '-' '_') = 0; }/" "$FILE_TO_MODIFY"
}
export -f reset_and_modify_file

# Ignore rc files outside the workspace, which could e.g. enable a remote cache. Use the same
# options for all commands to avoid server restarts and discarding the analysis cache.
export BAZEL_FLAGS="--experimental_java_classpath=bazel --disk_cache=$DISK_CACHE"
function bazel_cmd() {
  local command="$1"
  shift
  bazel --nosystem_rc --nohome_rc "$command" $BAZEL_FLAGS "$@"
}
export -f bazel_cmd

if [[ -n "$HEADER_JARS_ONLY" ]]; then
  # Only request the header jars and skip the one-version check, which requires the full jars.
  export BUILD_FLAGS="--aspects=//_turbine_benchmark:header_jars.bzl%header_jars \
    --output_groups=header_jars --norun_validations"
else
  export BUILD_FLAGS=
fi

function build_target() {
  local variant="$1"
  # Keep the last profile around after a benchmark.
  # Also work around https://github.com/bazelbuild/bazel/issues/20161.
  bazel_cmd build $BUILD_FLAGS \
    --experimental_remote_cache_eviction_retries=2 \
    --extra_toolchains="//_turbine_benchmark:${variant}_definition" \
    --profile="with_${variant}_turbine.profile.gz" \
    "$BAZEL_TARGET"
}
export -f build_target

# The Java toolchains below are only selected if their source version matches the language version
# used by Bazel's own build.
JAVA_LANGUAGE_VERSION="$(bazel_cmd cquery --output=starlark \
  --starlark:expr='build_options(target)["//command_line_option:java_language_version"]' \
  "$BAZEL_TARGET")"
if [[ ! "$JAVA_LANGUAGE_VERSION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: failed to determine --java_language_version, got: '$JAVA_LANGUAGE_VERSION'"
  exit 1
fi

cp oneversion_allowlist.csv _turbine_benchmark/
echo 'load("@rules_java//toolchains:default_java_toolchain.bzl", "default_java_toolchain")' \
  > _turbine_benchmark/BUILD
for i in "${!VARIANTS[@]}"; do
  if [[ -n "${VARIANT_PATHS[$i]}" ]]; then
    header_compiler_direct="header_compiler_direct = \"turbine_${VARIANTS[$i]}\","
  else
    header_compiler_direct=
  fi
  cat << EOF >> _turbine_benchmark/BUILD

default_java_toolchain(
    name = "${VARIANTS[$i]}",
    $header_compiler_direct
    oneversion_allowlist = "oneversion_allowlist.csv",
    source_version = "$JAVA_LANGUAGE_VERSION",
    target_version = "$JAVA_LANGUAGE_VERSION",
)
EOF
done
cat << 'EOF' > _turbine_benchmark/header_jars.bzl
load("@rules_java//java/common:java_info.bzl", "JavaInfo")

_HeaderJarsInfo = provider(fields = ["header_jars"])

_ATTRS = ["deps", "exports", "runtime_deps"]

def _header_jars_impl(target, ctx):
    transitive = [target[JavaInfo].compile_jars] if JavaInfo in target else []
    for attr in _ATTRS:
        deps = getattr(ctx.rule.attr, attr, [])
        if type(deps) == "list":
            transitive.extend([dep[_HeaderJarsInfo].header_jars for dep in deps if _HeaderJarsInfo in dep])
    header_jars = depset(transitive = transitive)
    return [
        _HeaderJarsInfo(header_jars = header_jars),
        OutputGroupInfo(header_jars = header_jars),
    ]

header_jars = aspect(
    implementation = _header_jars_impl,
    attr_aspects = _ATTRS,
)
EOF

# Fail if the header compile actions don't use the Turbine binary of a variant, e.g. because its
# toolchain isn't selected. Without this check, the benchmark would silently compare the same
# binary against itself.
function count_turbine_actions() {
  local variant="$1"
  local query="$2"
  { bazel_cmd aquery --extra_toolchains="//_turbine_benchmark:${variant}_definition" \
    --output=summary "$query" || true; } | sed -n 's/^  Turbine: //p'
}
for i in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$i]}"
  if [[ -z "${VARIANT_PATHS[$i]}" ]]; then
    continue
  fi
  all_actions="mnemonic('Turbine', deps($BAZEL_TARGET))"
  total="$(count_turbine_actions "$variant" "$all_actions")"
  used="$(count_turbine_actions "$variant" "inputs('.*/turbine_$variant', $all_actions)")"
  if [[ -z "$used" || "$used" == 0 ]]; then
    echo "ERROR: no header compile action uses the Turbine binary of '$variant'"
    exit 1
  fi
  echo "$used of $total native header compile actions use the Turbine binary of '$variant'"
done

hyperfine --shell=bash --warmup "$WARMUP" --runs "$RUNS" \
  --parameter-list variant "$(IFS=,; echo "${VARIANTS[*]}")" \
  --command-name '{variant}' \
  --setup 'build_target {variant}' \
  --prepare reset_and_modify_file \
  'build_target {variant}'
