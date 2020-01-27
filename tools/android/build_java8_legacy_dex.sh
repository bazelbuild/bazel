#!/bin/bash
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

# Find helper artifacts:
#   Windows (in MANIFEST):  <repository_name>/<path/to>/file
#   Linux/MacOS (symlink):  ${RUNFILES}/<repository_name>/<path/to>/file
if "$IS_WINDOWS"; then
  INPUT="$(rlocation "[^/]*/tools/android/desugared_java8_legacy_libs.jar")"
  CONFIG="$(rlocation "[^/]*/tools/android/minify_java8_legacy_libs.cfg")"
  SCAN="$(rlocation "[^/]*/src/tools/android/java/com/google/devtools/build/android/desugar/scan/KeepScanner")"
  PG="$(rlocation "[^/]*/third_party/java/proguard/proguard")"
  DEXER="$(rlocation "[^/]*/tools/android/dexer")"
else
  INPUT="$(find "${RUNFILES}" -path "*/tools/android/desugared_java8_legacy_libs.jar" | head -1)"
  CONFIG="$(find "${RUNFILES}" -path "*/tools/android/minify_java8_legacy_libs.cfg" | head -1)"
  SCAN="$(find "${RUNFILES}" -path "*/src/tools/android/java/com/google/devtools/build/android/desugar/scan/KeepScanner" | head -1)"
  DEXER="$(find "${RUNFILES}" -path "*/tools/android/dexer" | head -1)"
fi

android_jar=
binary_jar=
dest=
while [[ "$#" -gt 0 ]]; do
  arg="$1"; shift;
  case "${arg}" in
    --binary) binary_jar="$1"; shift ;;
    --binary=*) binary_jar="${arg:9}" ;;
    --output) dest="$1"; shift ;;
    --output=*) dest="${arg:9}" ;;
    --android_jar) android_jar="$1"; shift ;;
    --android_jar=*) android_jar="${arg:14}" ;;
    *) echo "Unknown flag: ${arg}"; exit 1 ;;
  esac
done

todex="${INPUT}"
if [[ -n "${binary_jar}" ]]; then
  tmpdir=$(mktemp -d)
  trap "rm -rf ${tmpdir}" EXIT

  # Minification requested
  # 1. compute -keep rules from binary
  seeds="${tmpdir}/seeds.cfg"
  "${SCAN}" \
      --input "${binary_jar}" \
      --classpath_entry "${todex}" \
      --bootclasspath_entry "${android_jar}" \
      --keep_file "${seeds}"

  # 2. proguard with -keep rules generated above and standard config file.
  # Use app's android.jar as -libraryjar.
  todex="${tmpdir}/proguarded.jar"
  "${PG}" \
      -injars "${INPUT}" \
      -outjars "${todex}" \
      -libraryjars "${android_jar}" \
      "@${CONFIG}" \
      "@${seeds}"
fi
# Convert .jar file to .dex
"${DEXER}" --dex "--output=${dest}" "${todex}"
