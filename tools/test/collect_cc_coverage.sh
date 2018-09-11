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
  local output_file="${1}"
  
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
      -d "${COVERAGE_DIR}" -o "${output_file}"

  # Fix up the paths to be relative by removing the prefix we specified above.
  sed -i -e "s*/proc/self/cwd/**g" "${output_file}"
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

  touch $output_file

  # Move .gcno files in $COVERAGE_DIR as the gcda files, because gcov
  # expects them to be under the same directory.
  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read gcno; do

    local gcda="${COVERAGE_DIR}/$(dirname ${gcno})/$(basename ${gcno} .gcno).gcda"
    # Check if .gcda file exists.
    # If no gcda file was found we generate empty coverage from the gcno file.
    if [ -f "$gcda" ]; then
        # If it exists, proceed in collecting code coverage.
        local tmp_gcno="${COVERAGE_DIR}/${gcno}"
        if [ ! -f "$tmp_gcno" ]; then
            # gcov expects both gcno and gcda files to be in the same directory.
            # We overcome this by copying the gcnos next to the gcdas
            mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno})"
            cp "$ROOT/${gcno}" "${tmp_gcno}"
        fi
        local gcno_basename="$(basename ${gcno})"
        # gcov produces files called <source file name>.gcov in the current
        # directory. These contain the coverage information of the source file
        # they correspond to. One .gcov file is produced for each source (or
        # header) file containing code, which was compiled to produce the data
        # (.gcda) files.
        "${GCOV}" -i -b -o "$(dirname ${gcda})" "${gcda}"
        echo "Let's see what is here in the cwd:"
        ls .
        local gcov_file=$(get_gcov_file $gcno)
        if [ -f $gcov_file ]; then
            cat "$gcov_file" >> "${output_file}"
            # We don't need this file anymore.
            rm -f "$gcov_file"
        fi

        gcov_file=$(get_gcov_header_file $gcno)
        if [ -f $gcov_file ]; then
            cat "$gcov_file" >> "${output_file}"
            # We don't need this file anymore.
            rm -f "$gcov_file"
        fi
    fi
  done
}

# Returns the .gcov file generated by gcov for the given gcno file.
#
# - gcno_file  The .gcno filename.
function get_gcov_file() {
    local gcno_file="${1}"
    # gcov places results in the current working dir. The gcov documentation
    # doesn't provide much details about how the name of the output file is
    # generated, other than hinting at it being named  <source file name>.gcov.
    # Since we only know the gcno filename, we try and see which of the following
    # extensions the source file had.
    local gcov_file="$(basename ${gcno} .gcno).gcov"
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .gcno).cc.gcov"
    fi
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .gcno).cpp.gcov"
    fi
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .gcno).c.gcov"
    fi
    # If we still haven't found it, try to find the files with
    # the .pic extensions.
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .pic.gcno).cc.gcov"
    fi
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .pic.gcno).cpp.gcov"
    fi
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .pic.gcno).c.gcov"
    fi
    echo "$gcov_file"
}

# Returns the .gcov file generated by gcov for the given gcno file.
#
# - gcno_file  The .gcno filename.
function get_gcov_header_file() {
    local gcno_file="${1}"
    # gcov places results in the current working dir. The gcov documentation
    # doesn't provide much details about how the name of the output file is
    # generated, other than hinting at it being named  <source file name>.gcov.
    # Since we only know the gcno filename, we try and see which of the following
    # extensions the source file had.
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .gcno).h.gcov"
    fi
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .gcno).hh.gcov"
    fi
    # If we still haven't found it, try to find the files with
    # the .pic extensions.
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .pic.gcno).h.gcov"
    fi
    if [ ! -f "$gcov_file" ]; then
        gcov_file="$(basename ${gcno} .pic.gcno).hh.gcov"
    fi
    echo "$gcov_file"
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