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

function test_static_library_symbols() {
  libstatic_a=$(find . -name libstatic.a)
  symbols=$(nm -C $libstatic_a)
  check_symbol_present "$symbols" "T foo"
  check_symbol_present "$symbols" "T bar"
  check_symbol_present "$symbols" "T unused"
  check_symbol_absent "$symbols" "lib_only"
}

test_static_library_symbols
