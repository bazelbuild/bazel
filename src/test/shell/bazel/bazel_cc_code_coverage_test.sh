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
[[ -z $( which gcov ) ]] && fail "gcov not installed. Skipping test" && exit 0
[[ -z $( which g++ ) ]] && fail "g++ not installed. Skipping test" && exit 0

# These are the variables needed by tools/test/collect_cc_coverage.sh
# They will be properly sub-shelled when invoking the script.

# Directory containing gcno and gcda files.
readonly COVERAGE_DIR_VAR="${PWD}/coverage_dir"
# Location of gcov.
readonly COVERAGE_GCOV_PATH_VAR="${PWD}/mygcov"
# Location from where the code coverage collection was invoked.
readonly ROOT_VAR="${PWD}"
# Location of the instrumented file manifest.
readonly COVERAGE_MANIFEST_VAR="${PWD}/coverage_manifest.txt"

# Path to the canonical C++ coverage script.
readonly COLLECT_CC_COVERAGE_SCRIPT=tools/test/collect_cc_coverage.sh

# Return a string in the form "device_id%inode" for a given file.
#
# Checking if two files have the same deviceID and inode is enough to
# determine if they are the same. For more details about inodes see
# http://www.grymoire.com/Unix/Inodes.html.
#
# - file   The absolute path of the file.
function get_file_id() {
  local file="${1}"; shift
  stat -c "%d:%i" ${file}
}

# Setup to be run for every test.
function set_up() {
   mkdir -p "${COVERAGE_DIR_VAR}"

  # COVERAGE_DIR has to be different than ROOT and PWD for the test to be
  # accurate.
  local coverage_dir_id=$(get_file_id "$COVERAGE_DIR_VAR")
  [[ $coverage_dir_id == $(get_file_id "$ROOT_VAR") ]] \
      && fail "COVERAGE_DIR_VAR must be different than ROOT_VAR"
  [[ $coverage_dir_id == $(get_file_id "$PWD") ]] \
      && fail "COVERAGE_DIR_VAR must be different than PWD"

  # The script expects gcov to be at $COVERAGE_GCOV_PATH.
  cp $( which gcov ) "$COVERAGE_GCOV_PATH_VAR"
  mkdir -p "$COVERAGE_DIR_VAR/coverage_srcs"

  # All generated .gcno files need to be in the manifest otherwise
  # the coverage report will be incomplete.
  echo "coverage_srcs/t.gcno" >> "$COVERAGE_MANIFEST_VAR"
  echo "coverage_srcs/a.gcno" >> "$COVERAGE_MANIFEST_VAR"

  # Create the CC sources.
  mkdir -p "$ROOT_VAR/coverage_srcs/"
  cat << EOF > "$ROOT_VAR/coverage_srcs/a.h"
int a(bool what);
EOF

  cat << EOF > "$ROOT_VAR/coverage_srcs/a.cc"
#include "a.h"
#include "b.h"

int a(bool what) {
  if (what) {
    return b(1);
  } else {
    return b(-1);
  }
}
EOF

  cat << EOF > "$ROOT_VAR/coverage_srcs/b.h"
int b(int what) {
  if (what > 0) {
    return 1;
  } else {
    return 2;
  }
}
EOF

  cat << EOF > "$ROOT_VAR/coverage_srcs/t.cc"
#include <stdio.h>
#include "a.h"

int main(void) {
  a(true);
}
EOF

  generate_and_execute_instrumented_binary coverage_srcs/test \
      "$COVERAGE_DIR_VAR/coverage_srcs" \
      coverage_srcs/a.h coverage_srcs/a.cc \
      coverage_srcs/b.h \
      coverage_srcs/t.cc

   # g++ generates the notes files in the current directory. The documentation
   # (https://gcc.gnu.org/onlinedocs/gcc/Gcov-Data-Files.html#Gcov-Data-Files)
   # says they are placed in the same directory as the object file, but they
   # are not. Therefore we move them in the same directory.
   mv a.gcno coverage_srcs/a.gcno
   mv t.gcno coverage_srcs/t.gcno
}

# Generates and executes an instrumented binary:
#
# Reads the list of arguments provided by the caller (using $@) and uses them
# to produce an instrumented binary using g++. This step also generates
# the notes (.gcno) files.
#
# Executes the instrumented binary. This step also generates the
# profile data (.gcda) files.
# - path_to_binary destination of the binary produced by g++
function generate_and_execute_instrumented_binary() {
  local path_to_binary="${1}"; shift
  local gcda_directory="${1}"; shift
  # -fprofile-arcs   Instruments $path_to_binary. During execution the binary
  #                  records code coverage information.
  # -ftest-coverage  Produces a notes (.gcno) file that coverage utilities
  #                  (e.g. gcov, lcov) can use to show a coverage report.
  # -fprofile-dir    Sets the directory where the profile data (gcda) appears.
  #
  # The profile data files need to be at a specific location where the C++
  # coverage scripts expects them to be ($COVERAGE_DIR/path/to/sources/).
  g++ -fprofile-arcs -ftest-coverage \
      -fprofile-dir="$gcda_directory" \
      "$@" -o "$path_to_binary"  \
       || fail "Couldn't produce the instrumented binary for $@ \
            with path_to_binary $path_to_binary"

   # Execute the instrumented binary and generates the profile data (.gcda)
   # file.
   # The profile data file is placed in $gcda_directory.
  "$path_to_binary" || fail "Couldn't execute the instrumented binary \
      $path_to_binary"
}

function tear_down() {
  rm -f "$COVERAGE_MANIFEST_VAR"
  rm -f "$COVERAGE_GCOV_PATH_VAR"
  rm -rf "$COVERAGE_DIR_VAR"
  rm -rf coverage_srcs/
}

# Asserts if the given expected coverage result is included in the given output
# file.
#
# - expected_coverage The expected result that must be included in the output.
# - output_file       The location of the coverage output file.
function assert_coverage_entry_in_file() {
    local expected_coverage="${1}"; shift
    local output_file="${1}"; shift

    # Replace newlines with commas to facilitate the assertion.
    local expected_coverage_no_newlines="$( echo "$expected_coverage" | tr '\n' ',' )"
    local output_file_no_newlines="$( cat "$output_file" | tr '\n' ',' )"

    (echo "$output_file_no_newlines" | grep  "$expected_coverage_no_newlines")\
        || fail "Expected coverage result
<$expected_coverage>
was not found in actual coverage report:
<$( cat $output_file )>"
}

# Asserts if coverage result in gcov format for coverage_srcs/a.cc is included
# in the given output file.
#
# - output_file    The location of the coverage output file.
function assert_gcov_coverage_srcs_a_cc() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/a.cc in gcov format.
    local expected_gcov_result_a_cc="file:coverage_srcs/a.cc
function:4,1,_Z1ab
lcount:4,1
lcount:5,1
lcount:6,1
lcount:8,0"
    assert_coverage_entry_in_file "$expected_gcov_result_a_cc" "$output_file"
}


# Asserts if coverage result in gcov format for coverage_srcs/t.cc is included
# in the given output file.
#
# - output_file    The location of the coverage output file.
function assert_gcov_coverage_srcs_t_cc() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/t.cc in gcov format.
    local expected_gcov_result_t_cc="file:coverage_srcs/t.cc
function:4,1,main
lcount:4,1
lcount:5,1
lcount:6,1"
    assert_coverage_entry_in_file "$expected_gcov_result_t_cc" "$output_file"
}

function assert_gcov_coverage_srcs_b_h() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/t.cc in gcov format.
    local expected_gcov_result="file:coverage_srcs/b.h
function:1,1,_Z1bi
lcount:1,1
lcount:2,1
lcount:3,1
lcount:5,0"
    assert_coverage_entry_in_file "$expected_gcov_result" "$output_file"
}


function test_cc_test_coverage_gcov() {
    "$gcov_location" -version | grep "LLVM" && \
      echo "gcov LLVM version not supported. Skipping test." && return
    # gcov -v | grep "gcov" outputs a line that looks like this:
    # gcov (Debian 7.3.0-5) 7.3.0
    local gcov_version="$(gcov -v | grep "gcov" | cut -d " " -f 4 | cut -d "." -f 1)"
    [ "$gcov_version" -lt 7 ] \
        && echo "gcov version before 7.0 is not supported. Skipping test." \
        && return

    (COVERAGE_DIR="$COVERAGE_DIR_VAR" \
    COVERAGE_GCOV_PATH="$COVERAGE_GCOV_PATH_VAR" \
    ROOT="$ROOT_VAR" COVERAGE_MANIFEST="$COVERAGE_MANIFEST_VAR" \
    BAZEL_CC_COVERAGE_TOOL="GCOV" \
    "$COLLECT_CC_COVERAGE_SCRIPT") > "$TEST_log"

    # Location of the output file of the C++ coverage script when gcov is used.
    local output_file="$COVERAGE_DIR_VAR/_cc_coverage.gcov"

    # Assert that the coverage output file contains the coverage data for the
    # two cc files: coverage_srcs/a.cc and coverage_srcs/t.cc.
    # The result for each source file must be asserted separately because the
    # coverage gcov does not guarantee any particular order.
    # The order can differ for example based on OS or version. The source files
    # order in the coverage report is not relevant.
    assert_gcov_coverage_srcs_a_cc "$output_file"
    assert_gcov_coverage_srcs_t_cc "$output_file"
    assert_gcov_coverage_srcs_b_h "$output_file"

    # This assertion is needed to make sure no other source files are included
    # in the output file.
    local nr_lines="$(wc -l < "$output_file")"
    [[ "$nr_lines" == 17 ]] || \
      fail "Number of lines in C++ gcov coverage output file is "\
      "$nr_lines and different than 17"
}

run_suite "Testing tools/test/collect_cc_coverage.sh"
