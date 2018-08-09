#!/bin/bash -x
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

# This script collects code coverage data for C++ sources, after the tests
# were executed.
#
# Bazel C++ code coverage collection support is poor and limited.
#
# Bazel uses the lcov tool for gathering coverage data. There is also
# an experimental support for clang llvm coverage, which uses the .profraw
# data files to compute the coverage report.
#
# This script assumes the following environment variables are set:
# - COVERAGE_DIR            Directory containing metadata files needed for
#                           coverage collection (e.g. gcda files, profraw).
# - COVERAGE_MANIFEST       Location of the instrumented file manifest.
# - CC_COVERAGE_OUTPUT_FILE Location of the final coverage report.
# - ROOT                    Location from where the code coverage collection
#                           was invoked.
#
# The script looks in $COVERAGE_DIR for the C++ metadata coverage files (either
# gcda or profraw) and uses either lcov or gcov to get the coverage data.
# The coverage data is placed in $COVERAGE_OUTPUT_FILE.

 function uses_llvm() {
  if stat "${COVERAGE_DIR}"/*.profraw >/dev/null 2>&1; then
    return
  fi
  false
}

function uses_gcov() {
  if [[ "$GCOV_COVERAGE" ]]; then
    return
  fi
  false
}

function init_gcov() {
  # Symlink the gcov tool such with a link called gcov. Clang comes with a tool
  # called llvm-cov, which behaves like gcov if symlinked in this way (otherwise
  # we would need to invoke it with "llvm-cov gcov").
  GCOV="${COVERAGE_DIR}/gcov"
  ln -s "${COVERAGE_GCOV_PATH}" "${GCOV}"
}

function llvm_coverage() {
  export LLVM_PROFILE_FILE="${COVERAGE_DIR}/%h-%p-%m.profraw"
  "${COVERAGE_GCOV_PATH}" merge -output "${CC_COVERAGE_OUTPUT_FILE}" "${COVERAGE_DIR}"/*.profraw
}

function lcov_coverage() {
  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read gcno; do
    mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno})"
    cp "${ROOT}/${gcno}" "${COVERAGE_DIR}/${gcno}"
  done

  # Run lcov over the .gcno and .gcda files to generate the lcov tracefile.
  # -c                    - Collect coverage data
  # --no-external         - Do not collect coverage data for system files
  # --ignore-errors graph - Ignore missing .gcno files; Bazel only instruments some files
  # -q                    - Quiet mode
  # --gcov-tool "${GCOV}" - Pass the local symlink to be uses as gcov by lcov
  # -b /proc/self/cwd     - Use this as a prefix for all source files instead of
  #                         the current directory
  # -d "${COVERAGE_DIR}"  - Directory to search for .gcda files
  # -o "${CC_COVERAGE_OUTPUT_FILE}" - Output file
  /usr/bin/lcov -c --no-external --ignore-errors graph -q \
      --gcov-tool "${GCOV}" -b /proc/self/cwd \
      -d "${COVERAGE_DIR}" -o "${CC_COVERAGE_OUTPUT_FILE}"
   # Fix up the paths to be relative by removing the prefix we specified above.
  sed -i -e "s*/proc/self/cwd/**g" "${CC_COVERAGE_OUTPUT_FILE}"
}

function gcc_gcov_coverage() {
  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read gcno; do
    mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno})"
    cp "$ROOT/${gcno}" "${COVERAGE_DIR}/${gcno}"
    gcda="${COVERAGE_DIR}/$(dirname ${gcno})/$(basename ${gcno} .gcno).gcda"
    "${GCOV}" --branch-probabilities --branch-counts --function-summaries --intermediate-format "${gcda}"
  done

  export CC_COVERAGE_OUTPUT_FILE="$COVERAGE_DIR/_coverage.gcov"
  find . -name "*.gcov" | while read path; do
    echo "Processing $path"
    cat $path >> "${CC_COVERAGE_OUTPUT_FILE}"
  done
}

init_gcov
if uses_llvm; then
  llvm_coverage
elif uses_gcov; then
  gcc_gcov_coverage
else
  lcov_coverage
fi