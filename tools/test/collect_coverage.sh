#!/bin/bash -x

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

# Wrapper script for collecting code coverage during test execution.
#
# Expected environment:
#   COVERAGE_MANIFEST - mandatory, location of the instrumented file manifest
#   LCOV_MERGER - mandatory, location of the LcovMerger
#   COVERAGE_DIR - optional, location of the coverage temp directory
#   COVERAGE_OUTPUT_FILE - optional, location of the final lcov file
#
# Script expects that it will be started in the execution root directory and
# not in the test's runfiles directory.

if [[ -z "$LCOV_MERGER" ]]; then
  echo --
  echo "Coverage collection running in legacy mode."
  echo "Legacy mode only supports C++ and even then, it's very brittle."
  COVERAGE_LEGACY_MODE=1
else
  COVERAGE_LEGACY_MODE=
fi

if [[ -z "$COVERAGE_MANIFEST" ]]; then
  echo --
  echo Coverage runner: \$COVERAGE_MANIFEST is not set
  echo Current environment:
  env | sort
  exit 1
fi

# When collect_coverage.sh is used, test runner must be instructed not to cd
# to the test's runfiles directory.
ROOT="$PWD"

if [[ "$COVERAGE_MANIFEST" != /* ]]; then
  # Canonicalize the path to coverage manifest so that tests can find it.
  export COVERAGE_MANIFEST="$ROOT/$COVERAGE_MANIFEST"
fi

# write coverage data outside of the runfiles tree
export COVERAGE_DIR=${COVERAGE_DIR:-"$ROOT/coverage"}
# make COVERAGE_DIR an absolute path
if ! [[ $COVERAGE_DIR == $ROOT* ]]; then
  COVERAGE_DIR=$ROOT/$COVERAGE_DIR
fi

mkdir -p "$COVERAGE_DIR"
COVERAGE_OUTPUT_FILE=${COVERAGE_OUTPUT_FILE:-"$COVERAGE_DIR/_coverage.dat"}
# make COVERAGE_OUTPUT_FILE an absolute path
if ! [[ $COVERAGE_OUTPUT_FILE == $ROOT* ]]; then
  COVERAGE_OUTPUT_FILE=$ROOT/$COVERAGE_OUTPUT_FILE
fi


# Java
# --------------------------------------
export JAVA_COVERAGE_FILE=$COVERAGE_DIR/jvcov.dat
# Let tests know that it is a coverage run
export COVERAGE=1
export BULK_COVERAGE_RUN=1


# Only check if file exists when LCOV_MERGER is set
if [[ ! "$COVERAGE_LEGACY_MODE" ]]; then
  for name in "$LCOV_MERGER"; do
    if [[ ! -e $name ]]; then
      echo --
      echo Coverage runner: cannot locate file $name
      exit 1
    fi
  done
fi

if [[ "$COVERAGE_LEGACY_MODE" ]]; then
  export GCOV_PREFIX_STRIP=3
  export GCOV_PREFIX="${COVERAGE_DIR}"
fi

cd "$TEST_SRCDIR"
"$@"
TEST_STATUS=$?

# always create output files
touch $COVERAGE_OUTPUT_FILE

if [[ $TEST_STATUS -ne 0 ]]; then
  echo --
  echo Coverage runner: Not collecting coverage for failed test.
  echo The following commands failed with status $TEST_STATUS
  echo "$@"
  exit $TEST_STATUS
fi

cd $ROOT

# If LCOV_MERGER is not set, use the legacy, awful, C++-only method to convert
# coverage files.
# NB: This is here just so that we don't regress. Do not add support for new
# coverage features here. Implement it instead properly in LcovMerger.
if [[ "$COVERAGE_LEGACY_MODE" ]]; then
  cat "${COVERAGE_MANIFEST}" | grep ".gcno$" | while read path; do
    mkdir -p "${COVERAGE_DIR}/$(dirname ${path})"
    cp "${ROOT}/${path}" "${COVERAGE_DIR}/${path}"
  done

  # Unfortunately, lcov messes up the source file names if it can't find the
  # files at their relative paths. Workaround by creating empty source files
  # according to the manifest (i.e., only for files that are supposed to be
  # instrumented).
  cat "${COVERAGE_MANIFEST}" | egrep ".(cc|h)$" | while read path; do
    mkdir -p "${COVERAGE_DIR}/$(dirname ${path})"
    touch "${COVERAGE_DIR}/${path}"
  done

  # Run lcov over the .gcno and .gcda files to generate the lcov tracefile.
  /usr/bin/lcov -c --no-external -d "${COVERAGE_DIR}" -o "${COVERAGE_OUTPUT_FILE}"

  # The paths are all wrong, because they point to /tmp. Fix up the paths to
  # point to the exec root instead (${ROOT}).
  sed -i -e "s*${COVERAGE_DIR}*${ROOT}*g" "${COVERAGE_OUTPUT_FILE}"

  exit $TEST_STATUS
fi

export LCOV_MERGER_CMD="${LCOV_MERGER} --coverage_dir=${COVERAGE_DIR} \
--output_file=${COVERAGE_OUTPUT_FILE}"


if [[ $DISPLAY_LCOV_CMD ]] ; then
  echo "Running lcov_merger"
  echo $LCOV_MERGER_CMD
  echo "-----------------"
fi

# JAVA_RUNFILES is set to the runfiles of the test, which does not necessarily
# contain a JVM (it does only if the test has a Java binary somewhere). So let
# the LCOV merger discover where its own runfiles tree is.
JAVA_RUNFILES= exec $LCOV_MERGER_CMD
