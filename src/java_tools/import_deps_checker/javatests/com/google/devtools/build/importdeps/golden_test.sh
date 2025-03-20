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

set -u

gold_output_file=$1
shift
gold_stderr_file=$1
shift
expected_exit_code=$1
shift
PRINT_JDEPS_PROTO=$1
shift

if [ -d "$TEST_TMPDIR" ]; then
  # Running as part of blaze test
  tmpdir="$TEST_TMPDIR"
else
  # Manual run from command line
  tmpdir="$(mktemp -d)"
fi

if [ -d "$TEST_UNDECLARED_OUTPUTS_DIR" ]; then
  # Running as part of blaze test: capture test output
  output="$TEST_UNDECLARED_OUTPUTS_DIR"
else
  # Manual run from command line: just write into temp dir
  output="${tmpdir}"
fi

output_file="${output}/actual_result.txt"
checker_stderr="${output}/checker_stderr.txt"

# Run the checker command.
$@ --jdeps_output "${output_file}" 2> ${checker_stderr}

checker_ret=$?
if [[ "${checker_ret}" != ${expected_exit_code} ]]; then
  echo "Checker error!!! ${checker_ret}, expected=${expected_exit_code}"
  cat ${checker_stderr}
  exit 1 # Exit with an error.
fi

# Correct for different paths between Blaze and Bazel.
diff <(sed 's|third_party/bazel/||g' "${gold_output_file}") <($PRINT_JDEPS_PROTO "${output_file}" | sed 's|third_party/bazel/||g')
gold_output_ret=$?

if [[ "${gold_output_ret}" != 0 ]] ; then
  echo "============== Actual Output =============="
  $PRINT_JDEPS_PROTO "${output_file}"
  echo "" # New line.
  echo "==========================================="

  echo "============== Expected Output =============="
  cat "${gold_output_file}"
  echo "" # New line.
  echo "==========================================="

  exit 1
fi

# Correct for different paths between Blaze and Bazel.
diff <(sed 's|third_party/bazel/||g' "${gold_stderr_file}") <(sed 's|third_party/bazel/||g' "${checker_stderr}")
gold_stderr_ret=$?

if [[ "${gold_stderr_ret}" != 0 ]]; then
  echo "============== Checker Stderr =============="
  cat "${checker_stderr}"
  echo "" # New line.
  echo "==========================================="

  echo "============== Expected Stderr =============="
  cat "${gold_stderr_file}"
  echo "" # New line.
  echo "==========================================="

  exit 1
fi
