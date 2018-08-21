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

function set_up() {
  if [[ ! -x /usr/bin/lcov ]]; then
    fail "lcov not installed. Skipping test."
  fi

  if [[ -z $( which gcov ) ]]; then
    fail "gcov not installed."
  fi

  if [[ -z $( which g++ ) ]]; then
    fail "g++ not installed."
  fi

  # The script expects gcov to be at $COVERAGE_GCOV_PATH.
  gcov_location=$( which gcov )
  cp $gcov_location "${PWD}/mygcov"

  # The script expects the output file to already exist.
  touch "${PWD}/coverage_report.dat"
  echo "coverage_srcs/a.gcno" >> "${PWD}/coverage_manifest.txt"

  setup_cc_sources
  generate_gcno_files coverage_srcs/a.h coverage_srcs/a.cc coverage_srcs/t.cc
  generate_instrumented_binary coverage_srcs/a.h coverage_srcs/a.cc coverage_srcs/t.cc
  generate_gcda_file
}

function setup_cc_sources() {
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
  # "-fprofile-arcs -ftest-coverage" tells the compiler to generate coverage
  # information needed by gcov and include additional code in the object files
  # for generating the profiling.
  g++ -fprofile-arcs -ftest-coverage "$@" -o ./coverage_srcs/test && return 0
  fail "Couldn't produce the instrumented binary for $@"
  return 1
}

# Execute the test coverage_srcs/test and generate the
# coverage_srcs/test.gcda file.
function generate_gcda_file() {
  ./coverage_srcs/test && return 0
  fail "Couldn't execute the instrumented binary for $@"
  return 1
}

function tear_down() {
  rm -rf coverage_srcs/
}

function test_cc_test_coverage() {
  (COVERAGE_DIR=${PWD} COVERAGE_GCOV_PATH=${PWD}/mygcov ROOT=${PWD} \
   COVERAGE_MANIFEST=${PWD}/coverage_manifest.txt \
   COVERAGE_OUTPUT_FILE=${PWD}/coverage_report.dat \
   tools/test/collect_cc_coverage.sh) >> $TEST_log

  # After running the test in coverage_srcs/t.cc, the sources covered are the
  # test itself and the source file a.cc.
  # For more details about the lcov format see
  # http://ltp.sourceforge.net/coverage/lcov/geninfo.1.php
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
  diff -u expected_result.dat ${PWD}/coverage_report.dat >> $TEST_log \
    || fail "Coverage output file is different than the expected file"
}

run_suite "testing tools/test/collect_cc_coverage.sh"