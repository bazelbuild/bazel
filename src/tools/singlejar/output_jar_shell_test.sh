#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
#
# Arguments:
#   unittest.bash script
#   singlejar path
#   jar tool path

(($# >= 2)) || \
  { echo "Usage: $0 <unittest.bash dir> <singlejar>" >&2; exit 1; }

# Load test environment
source $1/unittest.bash \
  || { echo "unittest.bash not found!" >&2; exit 1; }

set -e
declare -r singlejar="$2"


# Test that the entries single jar creates can be extracted (that is, they do
# not have some funny Unix access more settings making them unreadable).
function test_new_entries() {
  local -r out_jar="${TEST_TMPDIR}/out.jar"
  "$singlejar" --output "$out_jar"
  cd "${TEST_TMPDIR}"
  unzip "$out_jar" build-data.properties
  [[ -r build-data.properties ]] || \
    { echo "build-data.properties is not readable" >&2; exit 1; }
}

run_suite "Misc shell tests"
#!/bin/bash

