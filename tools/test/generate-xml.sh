#!/bin/bash

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

set -euo pipefail

TEST_LOG="$1"
XML_OUTPUT_FILE="$2"
DURATION_IN_SECONDS="$3"
EXIT_CODE="$4"

# Keep this in sync with test-setup.sh!
function encode_as_xml {
  if [[ -f "$1" ]]; then
    # Replace invalid XML characters and invalid sequence in CDATA
    # cf. https://stackoverflow.com/a/7774512/4717701
    perl -CSDA -pe's/[^\x9\xA\xD\x20-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}]+/?/g;' "$1" \
      | sed 's|]]>|]]>]]<![CDATA[>|g'
  fi
}

test_name="${TEST_BINARY#./}"
errors=0
error_msg=""
if (( $EXIT_CODE != 0 )); then
  errors=1
  error_msg="<error message=\"exited with error code $EXIT_CODE\"></error>"
fi

# Ensure that test shards have unique names in the xml output.
if [[ -n "${TEST_TOTAL_SHARDS+x}" ]] && ((TEST_TOTAL_SHARDS != 0)); then
  ((shard_num=TEST_SHARD_INDEX+1))
  test_name="${test_name}"_shard_"${shard_num}"/"${TEST_TOTAL_SHARDS}"
fi

FAILED=0
ENCODED_LOG="$(encode_as_xml "${TEST_LOG}")" || FAILED=1
cat <<EOF >${XML_OUTPUT_FILE}
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
<testsuite name="${test_name}" tests="1" failures="0" errors="${errors}">
  <testcase name="${test_name}" status="run" duration="${DURATION_IN_SECONDS}" time="${DURATION_IN_SECONDS}">${error_msg}</testcase>
  <system-out><![CDATA[${ENCODED_LOG}]]></system-out>
</testsuite>
</testsuites>
EOF
exit "$FAILED"
