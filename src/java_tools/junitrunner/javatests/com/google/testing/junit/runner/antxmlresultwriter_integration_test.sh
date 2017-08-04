#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All Rights Reserved.
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
# Integration testing of the AntXmlResultWriter.
#

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

TESTBED="${PWD}/$1"
XML_GOLDEN_OUTPUT_FILE="${PWD}/$2"
XML_OUTPUT_FILE="${TEST_TMPDIR}/test.xml"
SUITE_PARAMETER="$3"
SUITE="com.google.testing.junit.runner.testbed.XmlOutputExercises"
SUITE_FLAG="-D${SUITE_PARAMETER}=${SUITE}"

shift 3
source ${DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

function test_XmlOutputExercises() {
  cd $TEST_TMPDIR
  EXT_REGEX_FLAG="-E"
  # This test sometimes runs with an old version of sed that has -r but not -E.
  sed "$EXT_REGEX_FLAG" "" /dev/null 2> /dev/null || EXT_REGEX_FLAG="-r"

  $TESTBED --jvm_flag=${SUITE_FLAG} || true  # Test failures

  # Remove timestamps and test runtime from the XML files as they will always differ and cause a
  # mismatch.
  sed -i.bak "$EXT_REGEX_FLAG" "s/(time[^=]*)='[^']*/\1='/g" $XML_OUTPUT_FILE \
    || fail "sed to remove timestamps failed"

  # Removes the stacktrace from the XML files, it can vary between JDK versions.
  sed -i.bak "$EXT_REGEX_FLAG" '/\w*at [a-zA-Z0-9\$\.]+\([a-zA-Z0-9 \.]*(:[0-9]+)?\)$/d' \
      "${XML_OUTPUT_FILE}" || fail "sed to remove stacktraces failed"

  diff -wu $XML_GOLDEN_OUTPUT_FILE $XML_OUTPUT_FILE \
    || fail "Did not expect a diff between the golden file and the generated XML output."
}

run_suite "antxmlresultwriter"
