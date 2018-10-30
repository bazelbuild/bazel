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

# Computes code coverage data using the clang generated metadata found under
# $COVERAGE_DIR.
# Writes the collected coverage into the given output file.
function llvm_coverage() {
  local output_file="${1}"; shift
  export LLVM_PROFILE_FILE="${COVERAGE_DIR}/%h-%p-%m.profraw"
  "${COVERAGE_GCOV_PATH}" merge -output "${output_file}" \
      "${COVERAGE_DIR}"/*.profraw
}

# Computes code coverage data using gcda files found under $COVERAGE_DIR.
# Writes the collected coverage into the given output file in lcov format.
function lcov_coverage() {
  local output_file="${1}"; shift

  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read gcno; do
    mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno})"
    cp "${ROOT}/${gcno}" "${COVERAGE_DIR}/${gcno}"
  done

  local lcov_tool="$(which lcov)"
  if [[ ! -x "$lcov_tool" ]]; then
    lcov_tool=/usr/bin/lcov
  fi

  # Run lcov over the .gcno and .gcda files to generate the lcov tracefile.
  # -c                    - Collect coverage data
  # --no-external         - Do not collect coverage data for system files
  # --ignore-errors graph - Ignore missing .gcno files; Bazel only instruments
  #                         some files
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
# gcov and using the profile data (.gcda) and notes (.gcno) files.
#
# The profile data files are expected to be found under $COVERAGE_DIR.
# The notes file are expected to be found under $ROOT.
#
# - output_file     The location of the file where the generated code coverage
#                   report is written.
function gcov_coverage() {
  local output_file="${1}"; shift

  touch "$output_file"

  # Move .gcno files in $COVERAGE_DIR as the gcda files, because gcov
  # expects them to be under the same directory.
  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read gcno_path; do

    local gcda="${COVERAGE_DIR}/$(dirname ${gcno_path})/$(basename ${gcno_path} .gcno).gcda"
    # If the gcda file was not found we generate empty coverage from the gcno
    # file.
    if [[ -f "$gcda" ]]; then
        # gcov expects both gcno and gcda files to be in the same directory.
        # We overcome this by copying the gcno to $COVERAGE_DIR where the gcda
        # files are expected to be.
        if [ ! -f "${COVERAGE_DIR}/${gcno_path}" ]; then
            mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno_path})"
            cp "$ROOT/${gcno_path}" "${COVERAGE_DIR}/${gcno_path}"
        fi
        # Invoke gcov to generate a code coverage report with the flags:
        # -i              Output gcov file in an intermediate text format.
        #                 The output is a single .gcov file per .gcda file.
        #                 No source code is required.
        # -b              Write branch frequencies to the output file, and
        #                 write branch summary info to the standard output.
        # -o directory    The directory containing the .gcno and
        #                 .gcda data files.
        # "${gcda"}       The input file name. gcov is looking for data files
        #                 named after the input filename without its extension.
        "${GCOV}" -i -b -o "$(dirname ${gcda})" "${gcda}"

        # gcov produces files called <source file name>.gcov in the current
        # directory. These contain the coverage information of the source file
        # they correspond to. One .gcov file is produced for each source
        # (and/or header) file containing code which was compiled to produce
        # the .gcda files.
        # We try to find the correct source and header files that were generated
        # for the current gcno.
        # Retrieving every .gcov file that was generated in the current
        # directory is not correct because it can contain coverage information
        # for sources that are not included by the command line flag
        # --instrumentation_filter.

        local gcov_file="$(get_source_or_header_file source $gcno_path)"
        if [ -f "$gcov_file" ]; then
            cat "$gcov_file" >> "${output_file}"
            # We don't need this file anymore.
            rm -f "$gcov_file"
        fi

        gcov_file="$(get_source_or_header_file header $gcno_path)"
        if [ -f "$gcov_file" ]; then
            cat "$gcov_file" >> "${output_file}"
            # We don't need this file anymore.
            rm -f "$gcov_file"
        fi
    fi
  done
  echo "Coverage output file contains:"
  cat "${output_file}"
}

# Returns a .gcov corresponding to either a C++ source file or a C++ header
# file depending on the given file type, that could have been generated by gcov
# for the given gcno file.
#
# - filetype     Can be either "source" or "header".
# - gcno_file    The .gcno filename.
function get_source_or_header_file() {
    local filetype="${1}"; shift
    local gcno_file="${1}"; shift

    # gcov places results in the current working dir. The gcov documentation
    # doesn't provide much details about how the name of the output file is
    # generated, other than hinting at it being named  <source file name>.gcov.
    # Since we only know the gcno filename, we try and see which of the
    # following extensions the source file had.
    declare -a source_extensions

    case "$filetype" in
      ("source") source_extensions=("" ".cc" ".cpp" ".c") ;;
      ("header") source_extensions=(".h" ".hh") ;;
    esac

    declare -a is_pic_extensions=("" ".pic")

    declare -a gcov_extensions=(".gcda.gcov" ".gcov")

    local gcov_file=""
    for ext in "${source_extensions[@]}"
    do
      for pic_ext in "${is_pic_extensions[@]}"
      do
        for gcov_ext in "${gcov_extensions[@]}"
        do
          gcov_file="$(basename ${gcno_file} "$pic_ext.gcno")${ext}${gcov_ext}"
          if [ -f "$gcov_file" ]; then
            echo "$gcov_file" && return
          fi
        done
      done
    done
}

function main() {
  init_gcov

  if uses_llvm; then
    BAZEL_CC_COVERAGE_TOOL="PROFDATA"
  fi

  # All the output files must be generated under COVERAGE_DIR.
  #
  # When this script is invoked by tools/test/collect_coverage.sh the
  # .dat and .gcov files will be picked up by CoverageOutputGenerator and their
  # content will be converted and/or merged with other reports to an lcov
  # format, generating the final code coverage report.
  # The .profdata file will also be picked up by CoverageOutputGenerator but it
  # won't be merged or converted to lcov, but its content will be copied to the
  # final code coverage report.
  case "$BAZEL_CC_COVERAGE_TOOL" in
        ("GCOV") gcov_coverage "$COVERAGE_DIR/_cc_coverage.gcov" ;;
        ("LCOV") lcov_coverage "$COVERAGE_DIR/_cc_coverage.dat" ;;
        ("PROFDATA") llvm_coverage "$COVERAGE_DIR/_cc_coverage.profdata" ;;
        (*) echo "Coverage tool $BAZEL_CC_COVERAGE_TOOL not supported" \
            && exit 1
  esac
}

main
