#!/bin/bash -e
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

# Test that lists the content of the desugared Jar and compares it to a golden
# file.  This makes sure that output is deterministic and the resulting Jar
# doesn't contain any unwanted files, such as lambdas generated as part of
# running the desugaring tool.

progdir="$(dirname "$0")"

if [ -d "$TEST_TMPDIR" ]; then
  # Running as part of blaze test
  tmpdir="$TEST_TMPDIR"
else
  # Manual run from command line
  tmpdir="/tmp/test-$$"
  mkdir "${tmpdir}"
fi

if [ -d "$TEST_UNDECLARED_OUTPUTS_DIR" ]; then
  # Running as part of blaze test: capture test output
  output="$TEST_UNDECLARED_OUTPUTS_DIR"
else
  # Manual run from command line: just write into temp dir
  output="${tmpdir}"
fi

JAVABASE=$3
# Dump Jar file contents but drop coverage artifacts in case coverage is enabled
$JAVABASE/bin/jar tf "$1" \
    | grep -v '\.uninstrumented$' \
    | grep -v '\-paths\-for\-coverage\.txt$' >"${output}/actual_toc.txt"
# sorting can be removed when cl/145334839 is released
diff <(sort "$2") <(sort "${output}/actual_toc.txt")
