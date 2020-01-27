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

# exit on errors and uninitialized variables
set -eu

RUNFILES="${RUNFILES:-$0.runfiles}"
RUNFILES_MANIFEST_FILE="${RUNFILES_MANIFEST_FILE:-$RUNFILES/MANIFEST}"

IS_WINDOWS=false
case "$(uname | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  IS_WINDOWS=true
esac

if "$IS_WINDOWS" && ! type rlocation &> /dev/null; then
  function rlocation() {
    # Use 'sed' instead of 'awk', so if the absolute path ($2) has spaces, it
    # will be printed completely.
    local result="$(grep "$1" "${RUNFILES_MANIFEST_FILE}" | head -1)"
    # If the entry has a space, it is a mapping from a runfiles-path to absolute
    # path, otherwise it resolves to itself.
    echo "$result" | grep -q " " \
        && echo "$result" | sed 's/^[^ ]* //' \
        || echo "$result"
  }
fi

# Find script to call:
#   Windows (in MANIFEST):  <repository_name>/<path/to>/tool
#   Linux/MacOS (symlink):  ${RUNFILES}/<repository_name>/<path/to>/tool
if "$IS_WINDOWS"; then
  DESUGAR="$(rlocation "[^/]*/src/tools/android/java/com/google/devtools/build/android/desugar/Desugar")"
else
  DESUGAR="$(find "${RUNFILES}" -path "*/src/tools/android/java/com/google/devtools/build/android/desugar/Desugar" | head -1)"
fi

readonly TMPDIR="$(mktemp -d)"
trap "rm -rf ${TMPDIR}" EXIT

readonly DESUGAR_JAVA8_LIBS_CONFIG=(--rewrite_core_library_prefix java/time/ \
    --rewrite_core_library_prefix java/lang/Double8 \
    --rewrite_core_library_prefix java/lang/Integer8 \
    --rewrite_core_library_prefix java/lang/Long8 \
    --rewrite_core_library_prefix java/lang/Math8 \
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
    --rewrite_core_library_prefix java/util/concurrent/ConcurrentHashMap \
    --rewrite_core_library_prefix java/util/concurrent/ThreadLocalRandom \
    --rewrite_core_library_prefix java/util/concurrent/atomic/DesugarAtomic \
    --retarget_core_library_member "java/lang/Double#max->java/lang/Double8" \
    --retarget_core_library_member "java/lang/Double#min->java/lang/Double8" \
    --retarget_core_library_member "java/lang/Double#sum->java/lang/Double8" \
    --retarget_core_library_member "java/lang/Integer#max->java/lang/Integer8" \
    --retarget_core_library_member "java/lang/Integer#min->java/lang/Integer8" \
    --retarget_core_library_member "java/lang/Integer#sum->java/lang/Integer8" \
    --retarget_core_library_member "java/lang/Long#max->java/lang/Long8" \
    --retarget_core_library_member "java/lang/Long#min->java/lang/Long8" \
    --retarget_core_library_member "java/lang/Long#sum->java/lang/Long8" \
    --retarget_core_library_member "java/lang/Math#toIntExact->java/lang/Math8" \
    --retarget_core_library_member "java/util/Arrays#stream->java/util/DesugarArrays" \
    --retarget_core_library_member "java/util/Arrays#spliterator->java/util/DesugarArrays" \
    --retarget_core_library_member "java/util/Calendar#toInstant->java/util/DesugarCalendar" \
    --retarget_core_library_member "java/util/Date#from->java/util/DesugarDate" \
    --retarget_core_library_member "java/util/Date#toInstant->java/util/DesugarDate" \
    --retarget_core_library_member "java/util/GregorianCalendar#from->java/util/DesugarGregorianCalendar" \
    --retarget_core_library_member "java/util/GregorianCalendar#toZonedDateTime->java/util/DesugarGregorianCalendar" \
    --retarget_core_library_member "java/util/LinkedHashSet#spliterator->java/util/DesugarLinkedHashSet" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicInteger#getAndUpdate->java/util/concurrent/atomic/DesugarAtomicInteger" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicInteger#updateAndGet->java/util/concurrent/atomic/DesugarAtomicInteger" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicInteger#getAndAccumulate->java/util/concurrent/atomic/DesugarAtomicInteger" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicInteger#accumulateAndGet->java/util/concurrent/atomic/DesugarAtomicInteger" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicLong#getAndUpdate->java/util/concurrent/atomic/DesugarAtomicLong" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicLong#updateAndGet->java/util/concurrent/atomic/DesugarAtomicLong" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicLong#getAndAccumulate->java/util/concurrent/atomic/DesugarAtomicLong" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicLong#accumulateAndGet->java/util/concurrent/atomic/DesugarAtomicLong" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicReference#getAndUpdate->java/util/concurrent/atomic/DesugarAtomicReference" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicReference#updateAndGet->java/util/concurrent/atomic/DesugarAtomicReference" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicReference#getAndAccumulate->java/util/concurrent/atomic/DesugarAtomicReference" \
    --retarget_core_library_member "java/util/concurrent/atomic/AtomicReference#accumulateAndGet->java/util/concurrent/atomic/DesugarAtomicReference" \
    --emulate_core_library_interface java/util/Collection \
    --emulate_core_library_interface java/util/Map \
    --emulate_core_library_interface java/util/Map\$Entry \
    --emulate_core_library_interface java/util/Iterator \
    --emulate_core_library_interface java/util/Comparator \
    --dont_rewrite_core_library_invocation "java/util/Iterator#remove")

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
