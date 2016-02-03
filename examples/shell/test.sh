#!/bin/bash

# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -eu

# This allows RUNFILES to be declared outside the script it you want.
# RUNFILES for test is the directory of the script.
RUNFILES=${RUNFILES:-$($(cd $(dirname ${BASH_SOURCE[0]})); pwd)}

source "${RUNFILES}/examples/shell/lib.sh"

function test_output {
  OUTPUT=$(showfile)
  EXPECTED_OUTPUT=$(cat "${RUNFILES}/examples/shell/data/test_file.txt")

  if [ "${OUTPUT}" != "${EXPECTED_OUTPUT}" ]; then
    # This would be a failure case.
    exit 1
  fi
}

test_output
