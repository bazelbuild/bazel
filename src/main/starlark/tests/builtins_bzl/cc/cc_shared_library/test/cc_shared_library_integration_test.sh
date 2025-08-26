#!/usr/bin/env bash

# Copyright 2019 The Bazel Authors. All rights reserved.
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

set -euo pipefail

if [[ "$1" = "non_linux" ]]; then
  exit 0
fi

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source ${CURRENT_DIR}/testenv.sh

function check_symbol_present() {
  message="Should have seen '$2' but didn't."
  echo "$1" | (grep "$2" || (echo "$message" && exit 1))
}

function check_symbol_absent() {
  message="Shouldn't have seen '$2' but did."
  if [ "$(echo $1 | grep -c $2)" -gt 0 ]; then
    echo "$message"
    exit 1
  fi
}

function test_shared_library_symbols() {
  foo_so=$(find . -name libfoo_so.so)
  symbols=$(nm $foo_so)
  check_symbol_present "$symbols" "U _Z3barv"
  check_symbol_present "$symbols" "T _Z3bazv"
  check_symbol_present "$symbols" "T _Z3foov"
  check_symbol_present "$symbols" "T _Z3foov"
  check_symbol_present "$symbols" "t _Z3quxv"
  check_symbol_present "$symbols" "t _Z12indirect_depv"
  check_symbol_present "$symbols" "t _Z13indirect_dep2v"
  check_symbol_absent "$symbols" "_Z13indirect_dep3v"
  check_symbol_absent "$symbols" "_Z4bar3v"
}

function test_shared_library_user_link_flags() {
  foo_so=$(find . -name libfoo_so.so)
  # $RPATH defined in testenv.sh
  objdump -x $foo_so | grep $RPATH | grep "kittens" > /dev/null \
      || (echo "Expected to have RUNPATH contain 'kittens' (set by user_link_flags)" \
          && exit 1)
}

function do_test_binary() {
  symbols=$(nm -D $1)
  check_symbol_present "$symbols" "U _Z3foov"
  $1 | (grep -q "hello 42" || (echo "Expected 'hello 42'" && exit 1))
}

function test_binary() {
  binary=$(find . -name binary)
  do_test_binary $binary
}

function test_cc_test() {
  cc_test=$(find . -name cc_test)
  do_test_binary $cc_test
}

function test_number_of_linked_libs() {
  binary=$(find . -name binary)
  num_libs=$(readelf -d  $binary | grep NEEDED | wc -l)
  echo "$num_libs" | (grep -q  "$EXPECTED_NUM_LIBS" \
    || (echo "Expected $EXPECTED_NUM_LIBS linked libraries but was $num_libs" && exit 1))
}

test_shared_library_user_link_flags
test_shared_library_symbols
test_binary
test_cc_test
test_number_of_linked_libs
