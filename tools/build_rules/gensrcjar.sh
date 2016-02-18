#! /bin/sh
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

# List of environment variables that must be defined as input.
#
# We use environment variables instead of positional arguments or flags for
# clarity in the caller .bzl file and for simplicity of processing.  There is no
# need to implement a full-blown argument parser for this simple script.
INPUT_VARS="JAR OUTPUT PROTO_COMPILER SOURCE"

# Basename of the script for error reporting purposes.
PROGRAM_NAME="${0##*/}"

# A timestamp to mark all generated files with to get deterministic JAR outputs.
TIMESTAMP=198001010000

# Prints an error and exits.
#
# Args:
#   ...: list(str).  Parts of the message to print; all of them are joined
#       with a single space in between.
err() {
  echo "${PROGRAM_NAME}: ${*}" 1>&2
  exit 1
}

# Entry point.
main() {
  [ ${#} -eq 0 ] || err "No arguments allowed; set the following environment" \
      "variables for configuration instead: ${INPUT_VARS}"
  for var in ${INPUT_VARS}; do
    local value
    eval "value=\"\$${var}\""
    [ -n "${value}" ] || err "Input environment variable ${var} is not set"
  done

  rm -f "${OUTPUT}"

  local proto_output="${OUTPUT}.proto_output"
  rm -rf "${proto_output}"
  mkdir -p "${proto_output}"

  "${PROTO_COMPILER}" --java_out="${proto_output}" "${SOURCE}" \
      || err "proto_compiler failed"
  find "${proto_output}" -exec touch -t "${TIMESTAMP}" '{}' \; \
      || err "Failed to reset timestamps"
  "${JAR}" cMf "${OUTPUT}.tmp" -C "${proto_output}" . \
      || err "jar failed"
  mv "${OUTPUT}.tmp" "${OUTPUT}"
}

main "${@}"
