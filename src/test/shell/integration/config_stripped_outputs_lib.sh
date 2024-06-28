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
  # Note: keep subshells out of for loop headers as otherwise failures aren't propagated by
  # "set -e" and silently ignored.
  output_paths=$(echo "$cmd" | tr -s ' ' '\n' | xargs -n 1 | grep -E -o "${bazel_out}[^)]*")
  for o in $output_paths; do
    echo "$o" | grep -Ev "${bazel_out}/cfg/bin|${bazel_out}/cfg/genfiles" \
      && fail "expected all \"${bazel_out}\" paths to start with " \
      "\"${bazel_out}/cfg/bin.*\" or \"${bazel_out}/cfg/genfiles.*\": $o"
    if [[ "$o" == *"$identifying_action_output"* ]]; then
      found_identifying_output=1
    fi
  done

  # Check every output path in every .params file is stripped. Do not fail if there are no param
  # files.
  param_files=$(echo "$cmd" | tr -s ' ' '\n' | xargs -n 1 | grep -E -o "${bazel_out}[^)]*.params" || true)
  for o in $param_files; do
    bin_relative_path=$(echo $o | sed -r "s|${bazel_out}/cfg/bin/||")
    local_path="${bazel_out:0:5}-bin/${bin_relative_path}"
    # Don't fail if the file doesn't contain any output paths, but do fail if it doesn't exist.
    if [[ ! -f "$local_path" ]]; then
      fail "expected param file to exist: $local_path"
    fi
    param_file_paths=$(grep "${bazel_out}" $local_path || true)
    for k in $param_file_paths; do
      echo "$k" | grep -Ev "${bazel_out}/cfg/bin|${bazel_out}/cfg/genfiles" \
        && fail "$local_path: expected all \"${bazel_out}\" paths to start " \
        "with \"${bazel_out}/cfg/bin.*\" or \"${bazel_out}/cfg/genfiles.*\": $k"
      if [[ "$k" == *"$identifying_action_output"* ]]; then
        found_identifying_output=1
      fi
    done
  done

  assert_equals 1 "$found_identifying_output"
}

# Asserts a file contains no stripped paths.
#
# Arguments:
#   Path to the file.
function assert_contains_no_stripped_path() {
  # For "bazel-out/x86-fastbuild/bin/...", return "bazel-out".
  output_path=$(bazel info | grep '^output_path:')
  stripped_bin="${output_path##*/}/cfg/bin"

  assert_not_contains "$stripped_bin" "$1" "Stripped path found in $1"
}
