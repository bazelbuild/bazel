#!/bin/bash -eu
#
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
# limitations under the License.

# Unit tests for tools/test/collect_cc_code_coverage.sh


# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Check if all the tools required by CC coverage are installed.
[[ ! -x /usr/bin/lcov ]] && fail "lcov not installed. Skipping test."
[[ -z $( which gcov ) ]] && fail "gcov not installed."
[[ -z $( which g++ ) ]] && fail "g++ not installed."

# These are the variables needed by tools/test/collect_cc_coverage.sh
# They will be properly sub-shelled when invoking the script.
readonly coverage_dir="${PWD}"
readonly coverage_gcov_path="${PWD}/mygcov"
readonly root="${PWD}"
readonly coverage_manifest="${PWD}/coverage_manifest.txt"
readonly coverage_output_file="${PWD}/coverage_report.dat"

# Setup to be run for every test.
function set_up() {
  # The script expects gcov to be at $COVERAGE_GCOV_PATH.
  local gcov_location=$( which gcov )
  cp "$gcov_location" "${PWD}/mygcov"

  # The script expects the output file to already exist.
  touch "${PWD}/coverage_report.dat"
  echo "coverage_srcs/a.gcno" >> "${PWD}/coverage_manifest.txt"

  # Create the CC sources.
  mkdir -p coverage_srcs/
  cat << EOF > coverage_srcs/a.h
int a(bool what);
EOF

  cat << EOF > coverage_srcs/a.cc
#include "a.h"

int a(bool what) {
  if (what) {
    return 1;
  } else {
    return 2;
  }
}
EOF

  cat << EOF > coverage_srcs/t.cc
#include <stdio.h>
#include "a.h"

int main(void) {
  a(true);
}
EOF

  generate_gcno_files coverage_srcs/a.h coverage_srcs/a.cc coverage_srcs/t.cc
  generate_instrumented_binary ./coverage_srcs/test coverage_srcs/a.h \
      coverage_srcs/a.cc coverage_srcs/t.cc
  generate_gcda_file ./coverage_srcs/test
}

# Reads the list of arguments provided by the caller (using $@) and uses them
# to produco .gcno files using g++.
function generate_gcno_files() {
  # "-fprofile-arcs -ftest-coverage" tells the compiler to generate coverage
  # information needed by gcov and include additional code in the object files
  # for generating the profiling.
  g++ -fprofile-arcs -ftest-coverage "$@" && return 0
  fail "Couldn't produce .gcno files for $@"
  return 1
}

# Reads the list of arguments provided by the caller (using $@) and uses them
# to produce an instrumented binary using g++.
function generate_instrumented_binary() {
  local path_to_binary="${1}"; shift
  # "-fprofile-arcs -ftest-coverage" tells the compiler to generate coverage
  # information needed by gcov and include additional code in the object files
  # for generating the profiling.
  g++ -fprofile-arcs -ftest-coverage "$@" -o "$path_to_binary"  && return 0
  fail "Couldn't produce the instrumented binary for $@"
  return 1
}

# Execute the test coverage_srcs/test and generate the
# coverage_srcs/test.gcda file.
function generate_gcda_file() {
  local path_to_binary="${1}"
  "$path_to_binary" && return 0
  fail "Couldn't execute the instrumented binary $path_to_binary"
  return 1
}

function tear_down() {
  rm -rf coverage_srcs/
}

function run_coverage() {
  (COVERAGE_DIR="$coverage_dir" COVERAGE_GCOV_PATH="$coverage_gcov_path" \
   ROOT="$root" COVERAGE_MANIFEST="$coverage_manifest" \
   COVERAGE_OUTPUT_FILE="$coverage_output_file" tools/test/collect_cc_coverage.sh)
}

function test_cc_test_coverage() {
  run_coverage >> $TEST_log

  # After running the test in coverage_srcs/t.cc, the sources covered are the
  # test itself and the source file a.cc.
  # For more details about the lcov format see
  # http://ltp.sourceforge.net/coverage/lcov/geninfo.1.php
  # The expected result can be constructed manually by following the lcov
  # documentation and manually checking what lines of code are covered when
  # running the test.
  cat <<EOF > expected_result.dat
TN:
SF:coverage_srcs/a.cc
FN:3,_Z1ab
FNDA:1,_Z1ab
FNF:1
FNH:1
DA:3,1
DA:4,1
DA:5,1
DA:7,0
LF:4
LH:3
end_of_record
TN:
SF:coverage_srcs/t.cc
FN:4,main
FNDA:1,main
FNF:1
FNH:1
DA:4,1
DA:5,1
DA:6,1
LF:3
LH:3
end_of_record
EOF

  # tools/test/collect_cc_coverage.sh places the coverage result in $COVERAGE_OUTPUT_FILE
  diff -u expected_result.dat "$coverage_output_file" >> $TEST_log \
    || fail "Coverage output file is different than the expected file"
}

run_suite "testing tools/test/collect_cc_coverage.sh"