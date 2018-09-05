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
# Bazel C++ code coverage collection support is poor and limited. There is
# an ongoing effort to improve this (tracking issue #1118).
#
# Bazel uses the lcov tool for gathering coverage data. There is also
# an experimental support for clang llvm coverage, which uses the .profraw
# data files to compute the coverage report.
#
# This script assumes the following environment variables are set:
# - COVERAGE_DIR            Directory containing metadata files needed for
#                           coverage collection (e.g. gcda files, profraw).
# - COVERAGE_MANIFEST       Location of the instrumented file manifest.
# - COVERAGE_GCOV_PATH      Location of gcov. This is set by the TestRunner.
# - CC_COVERAGE_OUTPUT_FILE Location of the final coverage report.
# - ROOT                    Location from where the code coverage collection
#                           was invoked.
#
# The script looks in $COVERAGE_DIR for the C++ metadata coverage files (either
# gcda or profraw) and uses either lcov or gcov to get the coverage data.
# The coverage data is placed in $COVERAGE_OUTPUT_FILE.

# Checks if clang llvm coverage should be used instead of lcov.
function uses_llvm() {
  if stat "${COVERAGE_DIR}"/*.profraw >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

# Returns 0 if gcov must be used, 1 otherwise.
function uses_gcov() {
  [[ "$GCOV_COVERAGE" -eq "1"  ]] && return 0
  return 1
}

function init_gcov() {
  # Symlink the gcov tool such with a link called gcov. Clang comes with a tool
  # called llvm-cov, which behaves like gcov if symlinked in this way (otherwise
  # we would need to invoke it with "llvm-cov gcov").
  # For more details see https://llvm.org/docs/CommandGuide/llvm-cov.html.
  GCOV="${COVERAGE_DIR}/gcov"
  ln -s "${COVERAGE_GCOV_PATH}" "${GCOV}"
}

# Computes code coverage data using the clang generated metadata found under $COVERAGE_DIR.
# Writes the collected coverage into the given output file.
function llvm_coverage() {
  local output_file="${1}"
  export LLVM_PROFILE_FILE="${COVERAGE_DIR}/%h-%p-%m.profraw"
  "${COVERAGE_GCOV_PATH}" merge -output "${output_file}" "${COVERAGE_DIR}"/*.profraw
}

# Computes code coverage data using gcda files found under $COVERAGE_DIR.
# Writes the collected coverage into the given output file in lcov format.
function lcov_coverage() {
  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read gcno; do
    mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno})"
    cp "${ROOT}/${gcno}" "${COVERAGE_DIR}/${gcno}"
  done

  local lcov_tool=$(which lcov)
  if [[ ! -x $lcov_tool ]]; then
    lcov_tool=/usr/bin/lcov
  fi

  # Run lcov over the .gcno and .gcda files to generate the lcov tracefile.
  # -c                    - Collect coverage data
  # --no-external         - Do not collect coverage data for system files
  # --ignore-errors graph - Ignore missing .gcno files; Bazel only instruments some files
  # -q                    - Quiet mode
  # --gcov-tool "${GCOV}" - Pass the local symlink to be uses as gcov by lcov
  # -b /proc/self/cwd     - Use this as a prefix for all source files instead of
  #                         the current directory
  # -d "${COVERAGE_DIR}"  - Directory to search for .gcda files
  # -o "${COVERAGE_OUTPUT_FILE}" - Output file
  $lcov_tool -c --no-external --ignore-errors graph \
      --gcov-tool "${GCOV}" -b /proc/self/cwd \
      -d "${COVERAGE_DIR}" -o "${CC_COVERAGE_OUTPUT_FILE}"

  # Fix up the paths to be relative by removing the prefix we specified above.
  sed -i -e "s*/proc/self/cwd/**g" "${CC_COVERAGE_OUTPUT_FILE}"
}

# Generates a code coverage report in gcov intermediate text format by invoking
# gcov and using the profile (.gcda) and notes (.gcno) files.
#
# The profile files are expected to be found under $COVERAGE_DIR.
# The notes file are expected to be found under $ROOT.
#
# - output_file     The location of the file where the generated code coverage
#                   report is written.
function gcov_coverage() {
  local output_file="${1}"

  # Move .gcno files in $COVERAGE_DIR as the gcda files, because gcov
  # expects them to be under the same directory.
  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read gcno; do
    mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno})"
    cp "$ROOT/${gcno}" "${COVERAGE_DIR}/${gcno}"
    local gcda="${COVERAGE_DIR}/$(dirname ${gcno})/$(basename ${gcno} .gcno).gcda"

    # Invoking gcov for each gcda file. This generates a code coverage report
    # under the current directory that has the extension ".gcov".
    "${GCOV}" --branch-probabilities --branch-counts --function-summaries \
            --intermediate-format "${gcda}"
  done

  # Find all the .gcov code coverage reports generated previously and
  # concatenate them together into the output file.
  find . -name "*.gcov" | while read path; do
    echo "Processing $path"
    cat $path >> "${CC_COVERAGE_OUTPUT_FILE}"
  done
}

function main() {
  init_gcov

  # If llvm code coverage is used, we output the raw code coverage report in
  # the $COVERAGE_OUTPUT_FILE. This report will not be converted to any other
  # format by LcovMerger.
  if uses_llvm; then
    llvm_coverage "$COVERAGE_OUTPUT_FILE" && exit 0
  fi

  # When using either gcov or lcov, use an output file specific to the test
  # and format used. For lcov we generate a ".dat" output file and for gcov
  # a ".gcov" output file.
  # When this script is invoked by tools/test/collect_coverage.sh either of
  # these two coverage reports will be picked up by LcovMerger and their
  # content will be converted and/or merged with other reports to an lcov
  # format, generating the final code coverage report.
  case "$BAZEL_CC_COVERAGE_TOOL" in
        ("gcov") gcov_coverage "$COVERAGE_DIR/_cc_coverage.gcov" ;;
        ("lcov") lcov_coverage "$COVERAGE_DIR/_cc_coverage.dat" ;;
        (*) echo "Coverage tool $BAZEL_CC_COVERAGE_TOOL not supported" && exit 1
  esac
}

main
