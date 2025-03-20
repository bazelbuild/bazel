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
# - COVERAGE_GCOV_OPTIONS   Additional options to pass to gcov.
# - ROOT                    Location from where the code coverage collection
#                           was invoked.
# - VERBOSE_COVERAGE        Print debug info from the coverage scripts
#
# The script looks in $COVERAGE_DIR for the C++ metadata coverage files (either
# gcda or profraw) and uses either lcov or gcov to get the coverage data.
# The coverage data is placed in $COVERAGE_OUTPUT_FILE.

if [[ -n "$VERBOSE_COVERAGE" ]]; then
  set -x
fi

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
  if [ ! -f "${COVERAGE_GCOV_PATH}" ]; then
    echo "GCov does not exist at the given path: '${COVERAGE_GCOV_PATH}'"
    exit 1
  fi
  # When using a tool from a toolchain COVERAGE_GCOV_PATH will be a relative
  # path. To make it work on different working directories it's required to
  # convert the path to an absolute one.
  COVERAGE_GCOV_PATH_ABS="$(cd "${COVERAGE_GCOV_PATH%/*}" && pwd)/${COVERAGE_GCOV_PATH##*/}"
  ln -s "${COVERAGE_GCOV_PATH_ABS}" "${GCOV}"
}

# Computes code coverage data using the clang generated metadata found under
# $COVERAGE_DIR.
# Writes the collected coverage into the given output file.
function llvm_coverage_lcov() {
  local output_file="${1}"; shift
  export LLVM_PROFILE_FILE="${COVERAGE_DIR}/%h-%p-%m.profraw"
  "${COVERAGE_GCOV_PATH}" merge -output "${output_file}.data" \
      "${COVERAGE_DIR}"/*.profraw

  local object_param=""
  while read -r line; do
    if [[ ${line: -24} == "runtime_objects_list.txt" ]]; then
      while read -r line_runtime_object; do
        object_param+=" -object ${line_runtime_object}"
      done < "${line}"
    fi
  done < "${COVERAGE_MANIFEST}"

  "${LLVM_COV}" export -instr-profile "${output_file}.data" -format=lcov \
      -ignore-filename-regex='^/tmp/.+' \
      ${object_param} | sed 's#/proc/self/cwd/##' > "${output_file}"
}

function llvm_coverage_profdata() {
  local output_file="${1}"; shift
  export LLVM_PROFILE_FILE="${COVERAGE_DIR}/%h-%p-%m.profraw"
  "${COVERAGE_GCOV_PATH}" merge -output "${output_file}" \
      "${COVERAGE_DIR}"/*.profraw
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

  # We'll save the standard output of each the gcov command in this log.
  local gcov_log="$output_file.gcov.log"

  # Copy .gcno files next to their corresponding .gcda files in $COVERAGE_DIR
  # because gcov expects them to be in the same directory.
  while read -r line; do
    if [[ ${line: -4} == "gcno" ]]; then
      gcno_path=${line}
      local gcda="${COVERAGE_DIR}/$(dirname ${gcno_path})/$(basename ${gcno_path} .gcno).gcda"
      # If the gcda file was not found we skip generating coverage from the gcno
      # file.
      if [[ -f "$gcda" ]]; then
          # gcov expects both gcno and gcda files to be in the same directory.
          # We overcome this by copying the gcno to $COVERAGE_DIR where the gcda
          # files are expected to be.
          if [ ! -f "${COVERAGE_DIR}/${gcno_path}" ]; then
              mkdir -p "${COVERAGE_DIR}/$(dirname ${gcno_path})"
              cp "$ROOT/${gcno_path}" "${COVERAGE_DIR}/${gcno_path}"
          fi

          # Extract gcov's version: the output of `gcov --version` contains the
          # version as a set of major-minor-patch numbers, of which we extract
          # the major version.
          # gcov --version outputs a line like:
          #   gcov (Debian 7.3.0-5) 7.3.0
          # llvm-cov gcov --version outputs a line like:
          #   LLVM version 9.0.1
          gcov_major_version=$("${GCOV}" --version | sed -n -E -e 's/^.*\s([0-9]+)\.[0-9]+\.[0-9]+\s?.*$/\1/p')

          # Invoke gcov to generate a code coverage report with the flags:
          # -i              Output gcov file in an intermediate text format.
          #                 The output is a single .gcov file per .gcda file.
          #                 No source code is required.
          # -o directory    The directory containing the .gcno and
          #                 .gcda data files.
          # "${gcda"}       The input file name. gcov is looking for data files
          #                 named after the input filename without its extension.
          # gcov produces files called <source file name>.gcov in the current
          # directory. These contain the coverage information of the source file
          # they correspond to. One .gcov file is produced for each source
          # (or header) file containing code which was compiled to produce the
          # .gcda files.
          # Don't generate branch coverage (-b) because of a gcov issue that
          # segfaults when both -i and -b are used (see
          # https://gcc.gnu.org/bugzilla/show_bug.cgi?id=84879).

          # Don't generate branch coverage (-b) when using gcov 7 or earlier
          # because of a gcov issue that segfaults when both -i and -b are used
          # (see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=84879).
          if [[ $gcov_major_version -le 7 ]]; then
              "${GCOV}" -i $COVERAGE_GCOV_OPTIONS -o "$(dirname ${gcda})" "${gcda}"
          else
              "${GCOV}" -i -b $COVERAGE_GCOV_OPTIONS -o "$(dirname ${gcda})" "${gcda}"
          fi

          # Check the type of output: gcov 9 or later outputs compressed JSON
          # files, but earlier versions of gcov, and all versions of llvm-cov,
          # do not. These output textual information.
          if stat --printf='' *.gcov.json.gz > /dev/null 2>&1; then
              # Concatenating JSON documents does not yield a valid document, so they are moved individually
              mv -- *.gcov.json.gz "$(dirname "$output_file")/$(dirname ${gcno_path})"
          else
              # Append all .gcov files in the current directory to the output file.
              cat -- *.gcov >> "$output_file"
              # Delete the .gcov files.
              rm -- *.gcov
          fi
      fi
    fi
  done < "${COVERAGE_MANIFEST}"
}

function main() {
  init_gcov

  # If llvm code coverage is used, we output the raw code coverage report in
  # the $COVERAGE_OUTPUT_FILE. This report will not be converted to any other
  # format by LcovMerger.
  # TODO(#5881): Convert profdata reports to lcov.
  if uses_llvm; then
    if [[ "${GENERATE_LLVM_LCOV}" == "1" ]]; then
        BAZEL_CC_COVERAGE_TOOL="LLVM_LCOV"
    else
        BAZEL_CC_COVERAGE_TOOL="PROFDATA"
    fi
  fi

  # When using either gcov or lcov, have an output file specific to the test
  # and format used. For lcov we generate a ".dat" output file and for gcov
  # a ".gcov" output file. It is important that these files are generated under
  # COVERAGE_DIR.
  # When this script is invoked by tools/test/collect_coverage.sh either of
  # these two coverage reports will be picked up by LcovMerger and their
  # content will be converted and/or merged with other reports to an lcov
  # format, generating the final code coverage report.
  case "$BAZEL_CC_COVERAGE_TOOL" in
        ("GCOV") gcov_coverage "$COVERAGE_DIR/_cc_coverage.gcov" ;;
        ("PROFDATA") llvm_coverage_profdata "$COVERAGE_DIR/_cc_coverage.profdata" ;;
        ("LLVM_LCOV") llvm_coverage_lcov "$COVERAGE_DIR/_cc_coverage.dat" ;;
        (*) echo "Coverage tool $BAZEL_CC_COVERAGE_TOOL not supported" \
            && exit 1
  esac
}

main
