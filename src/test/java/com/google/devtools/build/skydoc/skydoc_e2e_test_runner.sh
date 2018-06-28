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
# Test skydoc output matches the expected golden file output.

set -u

skydoc_bin=$1
input_file=$2
golden_file=$3

actual_file="${TEST_TMPDIR}/actual"

set -e
${skydoc_bin} ${input_file} ${actual_file}
set +e

DIFF="$(diff ${actual_file} ${golden_file})"

if [ "$DIFF" != "" ]
then
    echo "Actual did not match golden."
    echo "${DIFF}"
    exit 1
else
    echo "Result matches golden file"
fi
