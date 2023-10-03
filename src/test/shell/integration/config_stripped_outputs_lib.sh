#!/bin/bash
#
# Copyright 2023 The Bazel Authors. All rights reserved.
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
# Helper functions for config_stripped_outputs_test.sh.

# Asserts an action strips all output paths.
#
# Arguments:
#   Path to the subcommand output of a "$ bazel build //foo -s" invocation.
#   Identifying path to isolate subcommand output to a single action
#     (from a grep). If multiple lines match, only checks the first. This can
#     happen, for example, when one actions write output O, then another action
#     reads O as an input.
function assert_paths_stripped() {
  local subcommand_output=$1
  local identifying_action_output=$2

  # For "bazel-out/x86-fastbuild/bin/...", return "bazel-out".
  output_path=$(bazel info | grep '^output_path:')
  bazel_out="${output_path##*/}"

  cmd=$(grep $identifying_action_output $subcommand_output | head -n 1)
  # Check this function's input was precise enough to isolate a single action.
  assert_equals 1 $(echo "$cmd" | wc -l)

  # Sanity check to ensure we actually find output paths to verify.
  found_identifying_output=0

  # Check every output path in the action command line is stripped.
  for o in $(echo $cmd | xargs -d' ' -n 1 | egrep -o "${bazel_out}[^)]*"); do
    echo "$o" | grep -v "${bazel_out}/bin" \
      && fail "expected all \"${bazel_out}\" paths to start with " \
      "\"${bazel_out}/bin.*\": $o"
    if [[ "$o" == *"$identifying_action_output"* ]]; then
      found_identifying_output=1
    fi
  done

  # Check every output path in every .params file is stripped.
  for o in $(echo $cmd | xargs -d' ' -n 1 | egrep -o  "${bazel_out}[^)]*.params"); do
    bin_relative_path=$(echo $o | sed -r "s|${bazel_out}/bin/||")
    local_path="${bazel_out:0:5}-bin/${bin_relative_path}"
    for k in $(grep "${bazel_out}" $local_path); do
      echo "$k" | grep -v "${bazel_out}/bin" \
        && fail "$local_path: expected all \"${bazel_out}\" paths to start " \
        "with \"${bazel_out}/bin.*\": $k"
      if [[ "$k" == *"$identifying_action_output"* ]]; then
        found_identifying_output=1
      fi
    done
  done

  assert_equals 1 "$found_identifying_output"
}
