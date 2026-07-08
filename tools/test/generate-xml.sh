#!/usr/bin/env bash

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

function encode_stream {
  if command -v sed >/dev/null; then
    LC_ALL=C sed -E \
        -e 's/.*/& /g' \
        -e 's/(('\
"$(echo -e '[\x9\x20-\x7f]')|"\
"$(echo -e '[\xc0-\xdf][\x80-\xbf]')|"\
"$(echo -e '[\xe0-\xec][\x80-\xbf][\x80-\xbf]')|"\
"$(echo -e '[\xed][\x80-\x9f][\x80-\xbf]')|"\
"$(echo -e '[\xee-\xef][\x80-\xbf][\x80-\xbf]')|"\
"$(echo -e '[\xf0][\x80-\x8f][\x80-\xbf][\x80-\xbf]')"\
')*)./\1?/g' \
        -e 's/(.*)\?/\1/g' \
        -e 's|]]>|]]>]]<![CDATA[>|g'
    return
  fi

  # Use only Bash builtins so this action does not depend on utilities installed
  # on the execution platform. Match the byte sequences accepted by the former
  # sed implementation and replace every other byte with a question mark.
  local LC_ALL=C
  local line
  while IFS= read -r line || [[ -n "$line" ]]; do
    local encoded=""
    local i=0
    local length=${#line}
    while ((i < length)); do
      local b0 b1=-1 b2=-1 b3=-1 sequence_length=0
      printf -v b0 '%d' "'${line:i:1}"
      if ((i + 1 < length)); then
        printf -v b1 '%d' "'${line:i+1:1}"
      fi
      if ((i + 2 < length)); then
        printf -v b2 '%d' "'${line:i+2:1}"
      fi
      if ((i + 3 < length)); then
        printf -v b3 '%d' "'${line:i+3:1}"
      fi

      if ((b0 == 9 || b0 == 13 || (b0 >= 32 && b0 <= 127))); then
        sequence_length=1
      elif ((b0 >= 192 && b0 <= 223 && b1 >= 128 && b1 <= 191)); then
        sequence_length=2
      elif ((
        ((b0 >= 224 && b0 <= 236) || (b0 >= 238 && b0 <= 239)) &&
          b1 >= 128 && b1 <= 191 && b2 >= 128 && b2 <= 191
      )); then
        sequence_length=3
      elif ((b0 == 237 && b1 >= 128 && b1 <= 159 && b2 >= 128 && b2 <= 191)); then
        sequence_length=3
      elif ((
        b0 == 240 && b1 >= 128 && b1 <= 143 &&
          b2 >= 128 && b2 <= 191 && b3 >= 128 && b3 <= 191
      )); then
        sequence_length=4
      fi

      if ((sequence_length > 0)); then
        encoded+="${line:i:sequence_length}"
        ((i += sequence_length))
      else
        encoded+='?'
        ((i += 1))
      fi
    done
    encoded="${encoded//]]>/]]>]]<![CDATA[>}"
    printf '%s\n' "$encoded"
  done
}

function encode_as_xml {
  if [ -f "$1" ]; then
    encode_stream <"$1"
  fi
}

# For testing, we allow calling this script with "-", in which case we only
# perform the encoding step. We intentionally ignore the rest of the parameters.
if [ "$TEST_LOG" == "-" ]; then
  encode_stream
  exit 0
fi

test_name="${TEST_BINARY#./}"
test_name="${test_name#../}"
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
ENCODED_LOG="$(encode_as_xml "${TEST_LOG}")" || FAILED=$?
printf '%s' "<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<testsuites>
  <testsuite name=\"${test_name}\" tests=\"1\" failures=\"0\" errors=\"${errors}\">
    <testcase name=\"${test_name}\" status=\"run\" duration=\"${DURATION_IN_SECONDS}\" time=\"${DURATION_IN_SECONDS}\">${error_msg}</testcase>
      <system-out>
Generated test.log (if the file is not UTF-8, then this may be unreadable):
<![CDATA[${ENCODED_LOG}]]>
      </system-out>
    </testsuite>
</testsuites>
" >"${XML_OUTPUT_FILE}"
exit "$FAILED"
