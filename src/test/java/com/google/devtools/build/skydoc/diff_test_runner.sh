#!/bin/bash
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
#
# A shell test that the contents of an input file match a golden file.
#
# Usage: diff_test_runner.sh ACTUAL_FILE GOLDEN_FILE

set -u

actual_file=$1
shift 1
golden_file=$1
shift 1

DIFF="$(diff ${actual_file} ${golden_file})"

if [ "$DIFF" != "" ]
then
    echo "FAIL: Actual did not match golden."
    echo "${DIFF}"
    exit 1
else
    echo "SUCCESS: Result matches golden file"
fi
