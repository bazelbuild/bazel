#!/usr/bin/env bash
# Copyright 2026 The Bazel Authors. All rights reserved.
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

# --- begin runfiles.bash initialization v3 ---
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

readonly SCRIPT="$(rlocation io_bazel/src/combine_derived_java_srcs.sh)"
readonly TEST_DIR="$(mktemp -d "${TEST_TMPDIR}/combine-derived-java-srcs.XXXXXXXX")"
readonly JAVABASE="${TEST_DIR}/fake jdk"
readonly JAR_LOG="${TEST_DIR}/jar.log"
readonly ZIP_INPUT="${TEST_DIR}/zip.input"
export JAR_LOG ZIP_INPUT
trap 'rm -rf "${TEST_DIR}"' EXIT

mkdir -p "${JAVABASE}/bin" "${TEST_DIR}/bin" "${TEST_DIR}/work"

cat >"${JAVABASE}/bin/jar" <<'EOF'
#!/bin/sh
set -eu
[ "$#" -eq 2 ]
[ "$1" = "xf" ]
[ -f "$2" ]
printf '%s\n' "$2" >>"${JAR_LOG}"
EOF
chmod +x "${JAVABASE}/bin/jar"

cat >"${TEST_DIR}/bin/zip" <<'EOF'
#!/bin/sh
set -eu
output=
for arg do
  output="$arg"
done
cat >"${ZIP_INPUT}"
: >"${output}"
EOF
chmod +x "${TEST_DIR}/bin/zip"

cd "${TEST_DIR}/work"
touch "first source.jar" "second.jar"

PATH="${TEST_DIR}/bin:${PATH}" "${SCRIPT}" \
  "${JAVABASE}" "combined sources.zip" "first source.jar" "second.jar"

printf '%s\n' \
  "${TEST_DIR}/work/first source.jar" \
  "${TEST_DIR}/work/second.jar" >expected.log
cmp expected.log "${JAR_LOG}"
[ -f "combined sources.zip" ]
