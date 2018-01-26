#!/bin/bash
function encode_output_file {
  if [ -f "$1" ]; then
    # Replace invalid XML characters and invalid sequence in CDATA
    # cf. https://stackoverflow.com/a/7774512/4717701
    perl -CSDA -pe's/[^\x9\xA\xD\x20-\x{D7FF}\x{E000}-\x{FFFD}\x{10000}-\x{10FFFF}]+/?/g;' "$1" \
      | sed 's|]]>|]]>]]<![CDATA[>|g'
  fi
}

function write_xml_output_file {
  local duration=$(expr $(date +%s) - $start)
  local errors=0
  local error_msg=
  local signal="${1-}"
  local test_name=
  if [ -n "${XML_OUTPUT_FILE-}" -a ! -f "${XML_OUTPUT_FILE-}" ]; then
    # Create a default XML output file if the test runner hasn't generated it
    if [ -n "${signal}" ]; then
      errors=1
      if [ "${signal}" = "SIGTERM" ]; then
        error_msg="<error message=\"Timed out\"></error>"
      else
        error_msg="<error message=\"Terminated by signal ${signal}\"></error>"
      fi
    elif (( $exitCode != 0 )); then
      errors=1
      error_msg="<error message=\"exited with error code $exitCode\"></error>"
    fi
    test_name="${TEST_BINARY#./}"
    # Ensure that test shards have unique names in the xml output.
    if [[ -n "${TEST_TOTAL_SHARDS+x}" ]] && ((TEST_TOTAL_SHARDS != 0)); then
      ((shard_num=TEST_SHARD_INDEX+1))
      test_name="${test_name}"_shard_"$shard_num"/"$TEST_TOTAL_SHARDS"
    fi
    cat <<EOF >${XML_OUTPUT_FILE}
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="$test_name" tests="1" failures="0" errors="${errors}">
    <testcase name="$test_name" status="run" duration="${duration}" time="${duration}">${error_msg}</testcase>
    <system-out><![CDATA[$(encode_output_file "${XML_OUTPUT_FILE}.log")]]></system-out>
  </testsuite>
</testsuites>
EOF
  fi
  rm -f "${XML_OUTPUT_FILE}.log"
}

TEST_LOG="$1"
XML_OUTPUT_FILE="$2"
DURATION_IN_SECONDS="$3"
EXIT_CODE="$4"

write_xml_output_file
