#!/bin/bash
#
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
#

function print_message_and_exit() {
  echo $1 >&2; exit 1;
}

if ! type rlocation >&/dev/null; then
  print_message_and_exit "ERROR[integration_test_setup.sh]: rlocation is not defined, you must source runfiles.bash"
fi

# Load the unit test framework
source "$(rlocation "io_bazel/src/test/shell/unittest.bash")" || print_message_and_exit "unittest.bash not found!"
# Load the test environment
source "$(rlocation "io_bazel/src/test/shell/testenv.sh")" || print_message_and_exit "testenv.sh not found!"
