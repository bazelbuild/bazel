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
function encode_stream {
  # Replace invalid XML characters and invalid sequence in CDATA
  # We do this in four steps:
  #
  # 1. Add a single whitespace character to the end of every line
  #
  # 2. Replace every sequence of legal characters followed by an illegal
  #    character *or* followed by a legal character at the end of the line with
  #    the same sequence of legal characters followed by a question mark
  #    character (replacing the illegal or last character). Since this will
  #    always replace the last character in a line with a question mark, we
  #    make sure to append a whitespace in step #1.
  #
  #    A character is legal if it is a valid UTF-8 character that is allowed in
  #    an XML file (this excludes a few control codes, but otherwise allows
  #    most UTF-8 characters).
  #
  #    We can't use sed in UTF-8 mode, because it would fail on the first
  #    illegal character. Instead, we have to match legal characters by their
  #    8-bit binary sequences, and also switch sed to an 8-bit mode.
  #
  #    The legal UTF codepoint ranges are 9,a,d,20-d7ff,e000-fffd,10000-10ffff,
  #    which results in the following 8-bit binary UTF-8 matchers:
  #       [\x9\xa\xd\x20-\x7f]                         <--- (9,A,D,20-7F)
  #       [\xc0-\xdf][\x80-\xbf]                       <--- (0080-07FF)
  #       [\xe0-\xec][\x80-\xbf][\x80-\xbf]            <--- (0800-CFFF)
  #       [\xed][\x80-\x9f][\x80-\xbf]                 <--- (D000-D7FF)
  #       [\xee][\x80-\xbf][\x80-\xbf]                 <--- (E000-EFFF)
  #       [\xef][\x80-\xbe][\x80-\xbf]                 <--- (F000-FFEF)
  #       [\xef][\xbf][\x80-\xbd]                      <--- (FFF0-FFFD)
  #       [\xf0-\xf7][\x80-\xbf][\x80-\xbf][\x80-\xbf] <--- (010000-10FFFF)
  #
  #    We omit \xa and \xd below since sed already splits the input into lines.
  #
  # 3. Remove the last character in the line, which we expect to be a
  #    question mark (that was originally added as a whitespace in step #1).
  #
  # 4. Replace the string ']]>' with ']]>]]<![CDATA[>' to prevent escaping the
  #    surrounding CDATA block.
  #
  # Sed supports the necessary operations as of version 4.4, but not in all
  # earlier versions. Specifically, we have found that sed 4.1.5 is not 8-bit
  # safe even when set to an 8-bit locale.
  #
  # OSX sed does not support escape sequences (\xhh), use echo as workaround.
  #
  # Alternatives considered:
  # Perl - We originally used Perl, but wanted to avoid the dependency.
  #        Recent versions of Perl now error on invalid utf-8 characters.
  # tr   - tr only replaces single-byte sequences, so cannot handle utf-8.
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
}

function encode_as_xml {
  if [ -f "$1" ]; then
    cat "$1" | encode_stream
  fi
}

# For testing, we allow calling this script with "-", in which case we only
# perform the encoding step. We intentionally ignore the rest of the parameters.
if [ "$TEST_LOG" == "-" ]; then
  encode_stream
  exit 0
fi

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
ENCODED_LOG="$(encode_as_xml "${TEST_LOG}")" || FAILED=$?
cat >"${XML_OUTPUT_FILE}" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="${test_name}" tests="1" failures="0" errors="${errors}">
    <testcase name="${test_name}" status="run" duration="${DURATION_IN_SECONDS}" time="${DURATION_IN_SECONDS}">${error_msg}</testcase>
      <system-out>
Generated test.log (if the file is not UTF-8, then this may be unreadable):
<![CDATA[${ENCODED_LOG}]]>
      </system-out>
    </testsuite>
</testsuites>
EOF
exit "$FAILED"
