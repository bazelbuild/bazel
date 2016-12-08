#!/bin/bash

# Copyright 2015 The Bazel Authors. All rights reserved.
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

# shift stderr to stdout.
exec 2>&1

no_echo=
if [[ "$1" = "--no_echo" ]]; then
  # Don't print anything to stdout in this special case.
  # Currently needed for persistent test runner.
  no_echo="true"
  shift
else
  echo 'exec ${PAGER:-/usr/bin/less} "$0" || exit 1'
fi

# Bazel sets some environment vars to relative paths, but it's easier to deal
# with absolute paths once we're actually running the test, so let's convert
# them.
if [[ "$TEST_SRCDIR" != /* ]]; then
  export TEST_SRCDIR="$PWD/$TEST_SRCDIR"
fi
if [[ "$JAVA_RUNFILES" != /* ]]; then
  export JAVA_RUNFILES="$PWD/$JAVA_RUNFILES"
fi
if [[ "$PYTHON_RUNFILES" != /* ]]; then
  export PYTHON_RUNFILES="$PWD/$PYTHON_RUNFILES"
fi
if [[ "$TEST_TMPDIR" != /* ]]; then
  export TEST_TMPDIR="$PWD/$TEST_TMPDIR"
fi
if [[ "$XML_OUTPUT_FILE" != /* ]]; then
  export XML_OUTPUT_FILE="$PWD/$XML_OUTPUT_FILE"
fi

# Tell googletest about Bazel sharding.
if [[ -n "${TEST_TOTAL_SHARDS+x}" ]] && ((TEST_TOTAL_SHARDS != 0)); then
  export GTEST_SHARD_INDEX="${TEST_SHARD_INDEX}"
  export GTEST_TOTAL_SHARDS="${TEST_TOTAL_SHARDS}"
fi
export GTEST_TMP_DIR="${TEST_TMPDIR}"

RUNFILES_MANIFEST_FILE="${TEST_SRCDIR}/MANIFEST"

if [ -z "$RUNFILES_MANIFEST_ONLY" ]; then
  function rlocation() {
    if [[ "$1" = /* ]]; then
      echo "$1"
    else
      echo "$(dirname $RUNFILES_MANIFEST_FILE)/$1"
    fi
  }
else
  function rlocation() {
    if [[ "$1" = /* ]]; then
      echo "$1"
    else
      echo $(grep "^$1 " $RUNFILES_MANIFEST_FILE | awk '{ print $2 }')
    fi
  }
fi

export -f rlocation
export RUNFILES_MANIFEST_FILE

DIR="$TEST_SRCDIR"
if [ ! -z "$TEST_WORKSPACE" ]; then
  DIR="$DIR"/"$TEST_WORKSPACE"
fi
[[ -n "$RUNTEST_PRESERVE_CWD" ]] && DIR="$PWD"


# normal commands are run in the exec-root where they have access to
# the entire source tree. By chdir'ing to the runfiles root, tests only
# have direct access to their declared dependencies.
if [ -z "$COVERAGE_DIR" ]; then
  cd "$DIR" || { echo "Could not chdir $DIR"; exit 1; }
fi

# This header marks where --test_output=streamed will start being printed.
if [[ -z "$no_echo" ]]; then
  echo "-----------------------------------------------------------------------------"
fi

# The path of this command-line is usually relative to the exec-root,
# but when using --run_under it can be a "/bin/bash -c" command-line.

# If the test is at the top of the tree, we have to add '.' to $PATH,
PATH=".:$PATH"

if [ -z "$COVERAGE_DIR" ]; then
  TEST_NAME=$1
  shift
else
  TEST_NAME=$2
fi

if [[ "$TEST_NAME" = /* ]]; then
  TEST_PATH="${TEST_NAME}"
else
  TEST_PATH="$(rlocation $TEST_WORKSPACE/$TEST_NAME)"
fi
[[ -n "$RUNTEST_PRESERVE_CWD" ]] && EXE="${TEST_NAME}"

exitCode=0
if [ -z "$COVERAGE_DIR" ]; then
  "${TEST_PATH}" "$@" || exitCode=$?
else
  "$1" "$TEST_PATH" "${@:3}" || exitCode=$?
fi


if [ -n "${XML_OUTPUT_FILE-}" -a ! -f "${XML_OUTPUT_FILE-}" ]; then
  # Create a default XML output file if the test runner hasn't generated it
  if (( $exitCode != 0 )); then
    errors=1
    error_msg="<error message=\"exited with error code $exitCode\"></error>"
  else
    errors=0
    error_msg=
  fi
  cat <<EOF >${XML_OUTPUT_FILE}
<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="$TEST_NAME" tests="1" failures="0" errors="$errors">
    <testcase name="$TEST_NAME" status="run">$error_msg</testcase>
  </testsuite>
</testsuites>
EOF
fi

exit $exitCode
