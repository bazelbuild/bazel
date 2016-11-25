#!/bin/bash

# Copyright 2016 The Bazel Authors. All rights reserved.
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

ROOT="$PWD"
if [[ $COVERAGE_OUTPUT_FILE != /* ]]; then
  COVERAGE_OUTPUT_FILE="${ROOT}/${COVERAGE_OUTPUT_FILE}"
fi
if [[ "$COVERAGE_MANIFEST" != /* ]]; then
  export COVERAGE_MANIFEST="${ROOT}/${COVERAGE_MANIFEST}"
fi

export COVERAGE_DIR="$(mktemp -d ${TMPDIR:-/tmp}/tmp.XXXXXXXXXX)"
trap "{ rm -rf ${COVERAGE_DIR} }" EXIT

# C++ env variables
export GCOV_PREFIX_STRIP=3
export GCOV_PREFIX="${COVERAGE_DIR}"

touch "${COVERAGE_OUTPUT_FILE}"

DIR="$TEST_SRCDIR"
if [ ! -z "$TEST_WORKSPACE" ]; then
  DIR="$DIR"/"$TEST_WORKSPACE"
fi
cd "$DIR" || { echo "Could not chdir $DIR"; exit 1; }
"$@"
TEST_STATUS=$?

if [[ ${TEST_STATUS} -ne 0 ]]; then
  echo "--"
  echo "Coverage runner: Not collecting coverage for failed test."
  echo "The following commands failed with status ${TEST_STATUS}:"
  echo "$@"
  exit ${TEST_STATUS}
fi

echo "--"
echo "Post-processing coverage results:"

cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read path; do
  mkdir -p "${COVERAGE_DIR}/$(dirname ${path})"
  cp "${ROOT}/${path}" "${COVERAGE_DIR}/${path}"
done

# Unfortunately, lcov messes up the source file names if it can't find the files
# at their relative paths. Workaround by creating empty source files according
# to the manifest (i.e., only for files that are supposed to be instrumented).
cat "${COVERAGE_MANIFEST}" | egrep ".(cc|h)$" | while read path; do
  mkdir -p "${COVERAGE_DIR}/$(dirname ${path})"
  touch "${COVERAGE_DIR}/${path}"
done

# Run lcov over the .gcno and .gcda files to generate the lcov tracefile.
/usr/bin/lcov -c --no-external -d "${COVERAGE_DIR}" -o "${COVERAGE_OUTPUT_FILE}"

# The paths are all wrong, because they point to /tmp. Fix up the paths to
# point to the exec root instead (${ROOT}).
sed -i -e "s*${COVERAGE_DIR}*${ROOT}*g" "${COVERAGE_OUTPUT_FILE}"

