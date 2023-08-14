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
source "${CURRENT_DIR}/coverage_helpers.sh" \
  || { echo "coverage_helpers.sh not found!" >&2; exit 1; }

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

  # Create the CC sources.
  mkdir -p "$ROOT_VAR/coverage_srcs/"
  mkdir -p "$ROOT_VAR/coverage_srcs/different"
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

  cat << EOF > "$ROOT_VAR/coverage_srcs/different/a.h"
int different_a(bool what);
EOF

  cat << EOF > "$ROOT_VAR/coverage_srcs/different/a.cc"
int different_a(bool what) {
  if (what) {
    return 1;
  } else {
    return 2;
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
#include "different/a.h"

int main(void) {
  a(true);
  different_a(false);
}
EOF

  generate_and_execute_instrumented_binary coverage_srcs/test \
      coverage_srcs/a.cc coverage_srcs/a.o \
      coverage_srcs/different/a.cc coverage_srcs/different/a.o \
      coverage_srcs/t.cc coverage_srcs/t.o

  agcno=$(ls coverage_srcs/*a.gcno)
  dagcno=$(ls coverage_srcs/different/*a.gcno)
  tgcno=$(ls coverage_srcs/*t.gcno)
  agcda=$(ls coverage_srcs/*a.gcda)
  dagcda=$(ls coverage_srcs/different/*a.gcda)
  tgcda=$(ls coverage_srcs/*t.gcda)
  # Even though gcov expects the gcda files to be next to the gcno files,
  # during Bazel execution this will not be the case. collect_cc_coverage.sh
  # expects them to be in the COVERAGE_DIR and will move the gcno files itself.
  # We cannot use -fprofile-dir during compilation because this causes the
  # filenames to undergo mangling; see
  # https://github.com/bazelbuild/bazel/issues/16229
  mv $agcda "$COVERAGE_DIR_VAR/$agcda"
  mv $tgcda "$COVERAGE_DIR_VAR/$tgcda"
  mkdir "$COVERAGE_DIR_VAR/$(dirname ${dagcda})"
  mv $dagcda "$COVERAGE_DIR_VAR/$dagcda"

  # All generated .gcno files need to be in the manifest otherwise
  # the coverage report will be incomplete.
  echo "$tgcno" >> "$COVERAGE_MANIFEST_VAR"
  echo "$agcno" >> "$COVERAGE_MANIFEST_VAR"
  echo "$dagcno" >> "$COVERAGE_MANIFEST_VAR"
}

# Generates and executes an instrumented binary:
#
# Reads the list of source files along with object files paths. Uses them
# to produce object files and link them to an instrumented binary using g++.
# This step also generates the notes (.gcno) files.
#
# Executes the instrumented binary. This step also generates the
# profile data (.gcda) files.
#
# - [(source_file, object_file),...] - source files and object file paths
# - path_to_binary destination of the binary produced by g++
function generate_and_execute_instrumented_binary() {
  local path_to_binary="${1}"; shift
  local src_file=""
  local obj_file=""
  local obj_files=""
  while [[ $# -ge 2 ]]; do
    src_file=$1; shift
    obj_file=$1; shift
    obj_files="${obj_files} $obj_file"
    g++ -coverage -fprofile-arcs -ftest-coverage -lgcov -c \
      "$src_file" -o "$obj_file"
  done

  g++ -coverage -fprofile-arcs -ftest-coverage -lgcov -o "$path_to_binary" $obj_files
  "$path_to_binary" || fail "Couldn't execute the instrumented binary \
      $path_to_binary"
}

function tear_down() {
  rm -f "$COVERAGE_MANIFEST_VAR"
  rm -f "$COVERAGE_GCOV_PATH_VAR"
  rm -rf "$COVERAGE_DIR_VAR"
  rm -rf coverage_srcs/
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
    assert_coverage_result "$expected_gcov_result_a_cc" "$output_file"
}


# Asserts if coverage result in gcov format for coverage_srcs/t.cc is included
# in the given output file.
#
# - output_file    The location of the coverage output file.
function assert_gcov_coverage_srcs_t_cc() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/t.cc in gcov format.
    local expected_gcov_result_t_cc="coverage_srcs/t.cc
function:5,1,main
lcount:5,1
lcount:6,1
lcount:7,1
lcount:8,1"
    assert_coverage_result "$expected_gcov_result_t_cc" "$output_file"
}

function assert_gcov_coverage_srcs_b_h() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/b.h in gcov format.
    local expected_gcov_result="file:coverage_srcs/b.h
function:1,1,_Z1bi
lcount:1,1
lcount:2,1
lcount:3,1
lcount:5,0"
    assert_coverage_result "$expected_gcov_result" "$output_file"
}

function assert_gcov_coverage_srcs_d_a_cc() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/different/a.cc in gcov format.
    local expected_gcov_result_d_a_cc="file:coverage_srcs/different/a.cc
function:1,1,_Z11different_ab
lcount:1,1
lcount:2,1
lcount:3,0
lcount:5,1"
    assert_coverage_result "$expected_gcov_result_d_a_cc" "$output_file"
}

function assert_gcov_coverage_srcs_a_cc_json() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/a.cc in gcov format.
    cat > expected_gcov_result_a_cc <<EOF
{"lines": [{"branches": [], "count": 1, "line_number": 4, "unexecuted_block": false, "function_name": "_Z1ab"}, {"branches": [], "count": 1, "line_number": 5, "unexecuted_block": false, "function_name": "_Z1ab"}, {"branches": [], "count": 1, "line_number": 6, "unexecuted_block": false, "function_name": "_Z1ab"}, {"branches": [], "count": 0, "line_number": 8, "unexecuted_block": true, "function_name": "_Z1ab"}], "functions": [{"blocks": 4, "end_column": 1, "start_line": 4, "name": "_Z1ab", "blocks_executed": 3, "execution_count": 1, "demangled_name": "a(bool)", "start_column": 5, "end_line": 10}], "file": "coverage_srcs/a.cc"}
EOF
local expected_gcov_result_a_cc=$(cat expected_gcov_result_a_cc | tr -d '\n')
    assert_coverage_result "$expected_gcov_result_a_cc" "$output_file"
}


# Asserts if coverage result in gcov format for coverage_srcs/t.cc is included
# in the given output file.
#
# - output_file    The location of the coverage output file.
function assert_gcov_coverage_srcs_t_cc_json() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/t.cc in gcov format.
    cat > expected_gcov_result_t_cc <<EOF
{"lines": [{"branches": [], "count": 1, "line_number": 5, "unexecuted_block": false, "function_name": "main"}, {"branches": [], "count": 1, "line_number": 6, "unexecuted_block": false, "function_name": "main"}, {"branches": [], "count": 1, "line_number": 7, "unexecuted_block": false, "function_name": "main"}, {"branches": [], "count": 1, "line_number": 8, "unexecuted_block": false, "function_name": "main"}], "functions": [{"blocks": 4, "end_column": 1, "start_line": 5, "name": "main", "blocks_executed": 4, "execution_count": 1, "demangled_name": "main", "start_column": 5, "end_line": 8}], "file": "coverage_srcs/t.cc"}
EOF
    local expected_gcov_result_t_cc=$(cat expected_gcov_result_t_cc | tr -d '\n')
    assert_coverage_result "$expected_gcov_result_t_cc" "$output_file"
}

function assert_gcov_coverage_srcs_b_h_json() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/b.h in gcov format.
    cat > expected_gcov_result_b_h <<EOF
{"lines": [{"branches": [], "count": 1, "line_number": 1, "unexecuted_block": false, "function_name": "_Z1bi"}, {"branches": [], "count": 1, "line_number": 2, "unexecuted_block": false, "function_name": "_Z1bi"}, {"branches": [], "count": 1, "line_number": 3, "unexecuted_block": false, "function_name": "_Z1bi"}, {"branches": [], "count": 0, "line_number": 5, "unexecuted_block": true, "function_name": "_Z1bi"}], "functions": [{"blocks": 4, "end_column": 1, "start_line": 1, "name": "_Z1bi", "blocks_executed": 3, "execution_count": 1, "demangled_name": "b(int)", "start_column": 5, "end_line": 7}], "file": "coverage_srcs/b.h"}
EOF
    local expected_gcov_result_b_h=$(cat expected_gcov_result_b_h | tr -d '\n')
    assert_coverage_result "$expected_gcov_result_b_h" "$output_file"
}

function assert_gcov_coverage_srcs_d_a_cc_json() {
    local output_file="${1}"; shift

    # The expected coverage result for coverage_srcs/different/a.cc in gcov format.
    cat > expected_gcov_result_d_a_cc <<EOF
{"lines": [{"branches": [], "count": 1, "line_number": 1, "unexecuted_block": false, "function_name": "_Z11different_ab"}, {"branches": [], "count": 1, "line_number": 2, "unexecuted_block": false, "function_name": "_Z11different_ab"}, {"branches": [], "count": 0, "line_number": 3, "unexecuted_block": true, "function_name": "_Z11different_ab"}, {"branches": [], "count": 1, "line_number": 5, "unexecuted_block": false, "function_name": "_Z11different_ab"}], "functions": [{"blocks": 4, "end_column": 1, "start_line": 1, "name": "_Z11different_ab", "blocks_executed": 3, "execution_count": 1, "demangled_name": "different_a(bool)", "start_column": 5, "end_line": 7}], "file": "coverage_srcs/different/a.cc"}
EOF
local expected_gcov_result_d_a_cc=$(cat expected_gcov_result_d_a_cc | tr -d '\n')
    assert_coverage_result "$expected_gcov_result_d_a_cc" "$output_file"
}

function test_cc_test_coverage_gcov() {
    local -r gcov_location=$(which gcov)
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

    # If the file _cc_coverge.gcov does not exist, it means we are using gcc 9
    # or higher which uses json.gz files as intermediate format. We will keep
    # testing gcc 7 and 8 until most users have migrated.
    if [ -f $output_file ]; then
      # Assert that the coverage output file contains the coverage data for the
      # two cc files: coverage_srcs/a.cc and coverage_srcs/t.cc.
      # The result for each source file must be asserted separately because the
      # coverage gcov does not guarantee any particular order.
      # The order can differ for example based on OS or version. The source files
      # order in the coverage report is not relevant.
      assert_gcov_coverage_srcs_a_cc "$output_file"
      assert_gcov_coverage_srcs_t_cc "$output_file"
      assert_gcov_coverage_srcs_b_h "$output_file"
      assert_gcov_coverage_srcs_d_a_cc "$output_file"

      # This assertion is needed to make sure no other source files are included
      # in the output file.
      local nr_lines="$(wc -l < "$output_file")"
      [[ "$nr_lines" == 24 ]] || \
        fail "Number of lines in C++ gcov coverage output file is "\
        "$nr_lines and different than 24"
    else

      # There may or may not be "gcda" in the extension.
      local not_found=0
      ls $COVERAGE_DIR_VAR/coverage_srcs/*.gcda.gcov.json.gz > /dev/null 2>&1 || not_found=$?
      if [[ $not_found -ne 0 ]]; then
        agcda=$(ls $COVERAGE_DIR_VAR/coverage_srcs/*a.gcov.json.gz)
        tgcda=$(ls $COVERAGE_DIR_VAR/coverage_srcs/*t.gcov.json.gz)
        dagcda=$(ls $COVERAGE_DIR_VAR/coverage_srcs/different/*a.gcov.json.gz)
      else
        agcda=$(ls $COVERAGE_DIR_VAR/coverage_srcs/*a.gcda.gcov.json.gz)
        tgcda=$(ls $COVERAGE_DIR_VAR/coverage_srcs/*t.gcda.gcov.json.gz)
        dagcda=$(ls $COVERAGE_DIR_VAR/coverage_srcs/different/*a.gcda.gcov.json.gz)
      fi
      output_file_json="output_file.json"
      zcat $agcda $tgcda $dagcda > $output_file_json

      # Remove all branch information from the json.
      # We disable branch coverage for gcov 7 and it is easier to remove it
      # than it is to put it back.
      # Replace "branches": [..] with "branches": [] - (non-empty [..])
      sed -E -i 's/"branches": \[[^]]+]/"branches": \[\]/g' "$output_file_json"
      assert_gcov_coverage_srcs_a_cc_json "$output_file_json"
      assert_gcov_coverage_srcs_t_cc_json "$output_file_json"
      assert_gcov_coverage_srcs_b_h_json "$output_file_json"
      assert_gcov_coverage_srcs_d_a_cc_json "$output_file_json"

      local nr_files="$(grep -o -i "\"file\":" "$output_file_json" | wc -l)"
      [[ "$nr_files" == 4 ]] || \
        fail "Number of files in C++ gcov coverage output file is "\
        "$nr_files and different than 4"
    fi
}

run_suite "Testing tools/test/collect_cc_coverage.sh"
