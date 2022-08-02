#!/bin/bash
# Copyright 2017 The Bazel Authors. All rights reserved.
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

# A wrapper around the desugar binary that sets
# jdk.internal.lambda.dumpProxyClasses and configures Java 8 library rewriting
# through additional flags.

RUNFILES="${RUNFILES:-$0.runfiles}"
CHECK_FOR_EXE=0
if [[ ! -d $RUNFILES ]]; then
  # Try the Windows path
  RUNFILES="${RUNFILES:-$0.exe.runfiles}"
  CHECK_FOR_EXE=1
fi
RUNFILES_MANIFEST_FILE="${RUNFILES_MANIFEST_FILE:-$RUNFILES/MANIFEST}"
export JAVA_RUNFILES=$RUNFILES
export RUNFILES_LIB_DEBUG=1
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

if [[ $CHECK_FOR_EXE -eq 0 ]]; then
  DESUGAR="$(rlocation "bazel_tools/src/tools/android/java/com/google/devtools/build/android/desugar/Desugar")"
else
  DESUGAR="$(rlocation "bazel_tools/src/tools/android/java/com/google/devtools/build/android/desugar/Desugar.exe")"
fi

readonly TMPDIR="$(mktemp -d)"
trap "rm -rf ${TMPDIR}" EXIT

readonly DESUGAR_JAVA8_LIBS_CONFIG=(--rewrite_core_library_prefix java/time/ \
    --rewrite_core_library_prefix java/lang/Double8 \
    --rewrite_core_library_prefix java/lang/Integer8 \
    --rewrite_core_library_prefix java/lang/Long8 \
    --rewrite_core_library_prefix java/lang/Math8 \
    --rewrite_core_library_prefix java/io/Desugar \
    --rewrite_core_library_prefix java/io/UncheckedIOException \
    --rewrite_core_library_prefix java/util/stream/ \
    --rewrite_core_library_prefix java/util/function/ \
    --rewrite_core_library_prefix java/util/Desugar \
    --rewrite_core_library_prefix java/util/DoubleSummaryStatistics \
    --rewrite_core_library_prefix java/util/IntSummaryStatistics \
    --rewrite_core_library_prefix java/util/LongSummaryStatistics \
    --rewrite_core_library_prefix java/util/Objects \
    --rewrite_core_library_prefix java/util/Optional \
    --rewrite_core_library_prefix java/util/PrimitiveIterator \
    --rewrite_core_library_prefix java/util/Spliterator \
    --rewrite_core_library_prefix java/util/StringJoiner \
    --rewrite_core_library_prefix javadesugar/testing/ \
    --rewrite_core_library_prefix java/util/concurrent/ConcurrentHashMap \
    --rewrite_core_library_prefix java/util/concurrent/ThreadLocalRandom \
    --rewrite_core_library_prefix java/util/concurrent/atomic/DesugarAtomic \
    --auto_desugar_shadowed_api_use \
    --emulate_core_library_interface java/util/Collection \
    --emulate_core_library_interface java/util/Map \
    --emulate_core_library_interface java/util/Map\$Entry \
    --emulate_core_library_interface java/util/Iterator \
    --emulate_core_library_interface java/util/Comparator \
    --dont_rewrite_core_library_invocation "java/util/Iterator#remove" )

# Check for params file.  Desugar doesn't accept a mix of params files and flags
# directly on the command line, so we need to build a new params file that adds
# the flags we want.
if [[ "$#" -gt 0 ]]; then
  arg="$1";
  case "${arg}" in
    @*)
      params="${TMPDIR}/desugar.params"
      cat "${arg:1}" > "${params}"  # cp would create file readonly
      for o in "${DESUGAR_JAVA8_LIBS_CONFIG[@]}"; do
        echo "${o}" >> "${params}"
      done

      "${DESUGAR}" \
          "--jvm_flag=-XX:+IgnoreUnrecognizedVMOptions" \
          "--jvm_flags=--add-opens=java.base/java.lang.invoke=ALL-UNNAMED" \
          "--jvm_flags=--add-opens=java.base/java.nio=ALL-UNNAMED" \
          "--jvm_flags=--add-opens=java.base/java.lang=ALL-UNNAMED" \
          "--jvm_flag=-Djdk.internal.lambda.dumpProxyClasses=${TMPDIR}" \
          "@${params}"
      # temp dir deleted by TRAP installed above
      exit 0
    ;;
  esac
fi

"${DESUGAR}" \
    "--jvm_flag=-XX:+IgnoreUnrecognizedVMOptions" \
    "--jvm_flags=--add-opens=java.base/java.lang.invoke=ALL-UNNAMED" \
    "--jvm_flags=--add-opens=java.base/java.nio=ALL-UNNAMED" \
    "--jvm_flags=--add-opens=java.base/java.lang=ALL-UNNAMED" \
    "--jvm_flag=-Djdk.internal.lambda.dumpProxyClasses=${TMPDIR}" \
    "$@" \
    "${DESUGAR_JAVA8_LIBS_CONFIG[@]}"

