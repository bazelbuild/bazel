#!/usr/bin/env bash

# Copyright 2024 The Bazel Authors. All rights reserved.
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

# Updates the PGO profile for the Turbine Graal Native Image on a representative
# target in the Bazel source tree.

set -euo pipefail

# The target must not use annotation processors as its header compilation would
# otherwise not use the native image.
REPRESENTATIVE_TARGET="${1:-//src/main/java/com/google/devtools/build/lib/actions:actions}"

cd "$BUILD_WORKSPACE_DIRECTORY"

echo "===== Building turbine with PGO instrumentation ====="
bazel build \
  -c opt \
  --//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:pgo_instrument \
  --//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:use_oracle_graalvm \
  //src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_direct_graal
TURBINE_PATH="$(
  bazel cquery \
    --output=files \
    -c opt \
    --//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:pgo_instrument \
    --//src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:use_oracle_graalvm \
    //src/java_tools/buildjar/java/com/google/devtools/build/java/turbine:turbine_direct_graal
)"

echo "===== Collecting profile for $REPRESENTATIVE_TARGET ====="

if [[ "$(bazel aquery --output=summary "mnemonic('Turbine', $REPRESENTATIVE_TARGET)" \
    | sed -n 's/^  Turbine: //p')" != 1 ]]; then
  echo "ERROR: $REPRESENTATIVE_TARGET is not header compiled with the native image, most likely"
  echo "because it uses annotation processors. Pass a different target as an argument."
  exit 1
fi

# The toolchain defined below is only selected if its source version matches
# the language version used by Bazel's own build.
JAVA_LANGUAGE_VERSION="$(bazel cquery --output=starlark \
  --starlark:expr='build_options(target)["//command_line_option:java_language_version"]' \
  "$REPRESENTATIVE_TARGET")"
if [[ ! "$JAVA_LANGUAGE_VERSION" =~ ^[0-9]+$ ]]; then
  echo "ERROR: failed to determine --java_language_version, got: '$JAVA_LANGUAGE_VERSION'"
  exit 1
fi

# The random path serves as a cache breaker for the build below.
PROFILE="$(pwd)/_turbine_pgo/profile.$(uuidgen).iprof"
PROFILE_JAVACOPTS="-XX:ProfilesDumpFile=$PROFILE"

function cleanup() {
  rm -rf _turbine_pgo
  buildozer "remove javacopts $PROFILE_JAVACOPTS" "$REPRESENTATIVE_TARGET"
}
trap cleanup EXIT INT TERM

mkdir -p _turbine_pgo
cp "$TURBINE_PATH" "_turbine_pgo/turbine_instrumented"
# Use a custom rule to build only the .hjar of the representative target.
cat << 'EOF' > _turbine_pgo/rules.bzl
load("@rules_java//java:defs.bzl", "JavaInfo")
def _build_header_jar_impl(ctx):
    header_jars = ctx.attr.target[JavaInfo].compile_jars
    return [
        DefaultInfo(files = header_jars),
    ]

build_header_jar = rule(
    implementation = _build_header_jar_impl,
    attrs = {
        "target": attr.label(),
    },
)
EOF
cat << EOF > _turbine_pgo/BUILD
load(":rules.bzl", "build_header_jar")
load("@rules_java//toolchains:default_java_toolchain.bzl", "default_java_toolchain")
default_java_toolchain(
    name = "turbine_benchmark_toolchain",
    source_version = "$JAVA_LANGUAGE_VERSION",
    target_version = "$JAVA_LANGUAGE_VERSION",
    header_compiler_direct = "turbine_instrumented",
)

build_header_jar(
    name = "header_jar",
    target = "$REPRESENTATIVE_TARGET",
)
EOF

# This is also passed to Turbine, but would fail javac actions, so we ensure
# that only the hjar is built.
buildozer "add javacopts $PROFILE_JAVACOPTS" "$REPRESENTATIVE_TARGET"
# We need to disable param files to ensure that the native image picks up the
# -XX:ProfilesDumpFile flag.
bazel build \
  --experimental_java_classpath=bazel \
  --extra_toolchains=//_turbine_pgo:turbine_benchmark_toolchain_definition \
  --nocheck_visibility \
  --sandbox_writable_path="$(dirname "$PROFILE")" \
  --min_param_file_size=100000 \
  //_turbine_pgo:header_jar

if [[ ! -f "$PROFILE" ]]; then
  echo "ERROR: Turbine did not write a profile, check that the instrumented binary is used"
  exit 1
fi
mv "$PROFILE" src/java_tools/buildjar/java/com/google/devtools/build/java/turbine/profile.iprof
