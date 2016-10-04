#!/bin/bash

# Copyright 2014 The Bazel Authors. All rights reserved.
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

# This script bootstraps building a Bazel binary without Bazel then
# use this compiled Bazel to bootstrap Bazel itself. It can also
# be provided with a previous version of Bazel to bootstrap Bazel
# itself.
# The resulting binary can be found at output/bazel.

set -o errexit

cd "$(dirname "$0")"

# Set the default verbose mode in buildenv.sh so that we do not display command
# output unless there is a failure.  We do this conditionally to offer the user
# a chance of overriding this in case they want to do so.
: ${VERBOSE:=no}

source scripts/bootstrap/buildenv.sh

function usage() {
  [ -n "${1:-compile}" ] && echo "Invalid command(s): $1" >&2
  echo "syntax: $0 [command[,command]* [BAZEL_BIN [BAZEL_SUM]]]" >&2
  echo "  General purpose commands:" >&2
  echo "     compile       = compile the bazel binary (default)" >&2
  echo "  Commands for developers:" >&2
  echo "     all         = compile,determinism,test" >&2
  echo "     determinism = test for stability of Bazel builds" >&2
  echo "     srcs        = test that //:srcs contains all the sources" >&2
  echo "     test        = run the full test suite of Bazel" >&2
  exit 1
}

function parse_options() {
  local keywords="(compile|all|determinism|bootstrap|srcs|test)"
  COMMANDS="${1:-compile}"
  [[ "${COMMANDS}" =~ ^$keywords(,$keywords)*$ ]] || usage "$@"
  DO_COMPILE=
  DO_CHECKSUM=
  DO_FULL_CHECKSUM=1
  DO_TESTS=
  DO_SRCS_TEST=
  [[ "${COMMANDS}" =~ (compile|all) ]] && DO_COMPILE=1
  [[ "${COMMANDS}" =~ (bootstrap|determinism|all) ]] && DO_CHECKSUM=1
  [[ "${COMMANDS}" =~ (bootstrap) ]] && DO_FULL_CHECKSUM=
  [[ "${COMMANDS}" =~ (srcs|all) ]] && DO_SRCS_TEST=1
  [[ "${COMMANDS}" =~ (test|all) ]] && DO_TESTS=1

  BAZEL_BIN=${2:-"bazel-bin/src/bazel"}
  BAZEL_SUM=${3:-"x"}
}

parse_options "${@}"

mkdir -p output
: ${BAZEL:=${2-}}

#
# Create an initial binary so we can host ourself
#
if [ ! -x "${BAZEL}" ]; then
  display "$INFO You can skip this first step by providing a path to the bazel binary as second argument:"
  display "$INFO    $0 ${COMMANDS} /path/to/bazel"
  new_step 'Building Bazel from scratch'
  source scripts/bootstrap/compile.sh
  # The DO_COMPILE flow will actually create the bazel binary and set BAZEL.
  DO_COMPILE=1
fi

#
# Bootstrap bazel using the previous bazel binary = release binary
#
if [ "${EMBED_LABEL-x}" = "x" ]; then
  # Add a default label when unspecified
  git_sha1=$(git_sha1)
  EMBED_LABEL="$(get_last_version) (@${git_sha1:-non-git})"
fi

if [[ $PLATFORM == "darwin" ]] && \
    xcodebuild -showsdks 2> /dev/null | grep -q '\-sdk iphonesimulator'; then
  EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS-} --define IPHONE_SDK=1"
fi
source scripts/bootstrap/bootstrap.sh

if [ $DO_COMPILE ]; then
  new_step 'Building Bazel with Bazel'
  display "."
  log "Building output/bazel"
  bazel_build "src:bazel${EXE_EXT}" \
    || fail "Could not build Bazel"
  bazel_bin_path="$(get_bazel_bin_path)/src/bazel${EXE_EXT}"
  [ -e "$bazel_bin_path" ] \
    || fail "Could not find freshly built Bazel binary at '$bazel_bin_path'"
  cp -f "$bazel_bin_path" "output/bazel${EXE_EXT}" \
    || fail "Could not copy '$bazel_bin_path' to 'output/bazel${EXE_EXT}'"
  chmod 0755 "output/bazel${EXE_EXT}"
  BAZEL="$(pwd)/output/bazel${EXE_EXT}"
fi

#
# Output is deterministic between two bootstrapped bazel binary using the actual tools and the
# released binary.
#
if [ $DO_CHECKSUM ]; then
  new_step "Determinism test"
  if [ ! -f ${BAZEL_SUM:-x} ]; then
    BAZEL_SUM=bazel-out/bazel_checksum
    log "First build"
    bootstrap_test ${BAZEL} ${BAZEL_SUM}
  else
    BOOTSTRAP=${BAZEL}
  fi
  if [ "${BAZEL_SUM}" != "${OUTPUT_DIR}/bazel_checksum" ]; then
    cp ${BAZEL_SUM} ${OUTPUT_DIR}/bazel_checksum
  fi
  if [ $DO_FULL_CHECKSUM ]; then
    log "Second build"
    bootstrap_test ${BOOTSTRAP} bazel-out/bazel_checksum
    log "Comparing output"
    (diff -U 0 ${OUTPUT_DIR}/bazel_checksum bazel-out/bazel_checksum >&2) \
        || fail "Differences detected in outputs!"
  fi
fi

#
# Test that //:srcs contains all the sources
#
if [ $DO_SRCS_TEST ]; then
  new_step "Checking that //:srcs contains all the sources"
  log "Querying //:srcs"
  ${BAZEL} query 'kind("source file", deps(//:srcs))' 2>/dev/null \
    | grep -v '^@' \
    | sed -e 's|^//||' | sed 's|^:||' | sed 's|:|/|' \
    | sort -u >"${OUTPUT_DIR}/srcs-query"

  log "Finding all files"
  # SRCS_EXCLUDES can be overriden to adds some more exceptions for the find
  # commands (for CI systems).
  SRCS_EXCLUDES=${SRCS_EXCLUDES-XXXXXXXXXXXXXX1268778dfsdf4}
  # See file BUILD for the list of grep -v exceptions.
  # tools/defaults package is hidden by Bazel so cannot be put in the srcs.
  find . -type f | sed 's|./||' \
    | grep -v '^bazel-' | grep -v '^WORKSPACE.user.bzl' \
    | grep -v '^\.' | grep -v '^out/' | grep -v '^output/' \
    | grep -Ev "${SRCS_EXCLUDES}" \
    | grep -v '^tools/defaults/BUILD' \
    | sort -u >"${OUTPUT_DIR}/srcs-find"

  log "Diffing"
  res="$(diff -U 0 "${OUTPUT_DIR}/srcs-find" "${OUTPUT_DIR}/srcs-query" | sed 's|^-||' | grep -Ev '^(@@|\+\+|--)' || true)"

  if [ -n "${res}" ]; then
    fail "//:srcs filegroup do not contains all the sources, missing:
${res}"
  fi
fi

#
# Tests
#
if [ $DO_TESTS ]; then
  new_step "Running tests"
  display "."

  ndk_target="$(get_bind_target //external:android_ndk_for_testing)"
  sdk_target="$(get_bind_target //external:android_sdk_for_testing)"
  if [ "$ndk_target" = "//:dummy" -o "$sdk_target" = "//:dummy" ]; then
    display "$WARNING Android SDK or NDK are not set in the WORKSPACE file. Android tests will not be run."
  fi

  [ -n "$JAVAC_VERSION" ] || get_java_version
  if [[ ! "${BAZEL_TEST_FILTERS-}" =~ "-jdk8" ]]; then
    if [ "8" -gt ${JAVAC_VERSION#*.} ] || [ "${JAVA_VERSION}" = "1.7" ]; then
      display "$WARNING Your version of Java is lower than 1.8!"
      display "$WARNING Deactivating Java 8 tests, please use a JDK 8 to fully"
      display "$WARNING test Bazel."
      if [ -n "${BAZEL_TEST_FILTERS-}" ]; then
        BAZEL_TEST_FILTERS="${BAZEL_TEST_FILTERS},-jdk8"
      else
        BAZEL_TEST_FILTERS="-jdk8"
      fi
    fi
  fi
  $BAZEL --bazelrc=${BAZELRC} --nomaster_bazelrc \
      ${BAZEL_DIR_STARTUP_OPTIONS} \
      test \
      --test_tag_filters="${BAZEL_TEST_FILTERS-}" \
      --build_tests_only \
      --nolegacy_bazel_java_test \
      --define JAVA_VERSION=${JAVA_VERSION} \
      ${EXTRA_BAZEL_ARGS} \
      -k --test_output=errors //src/... //third_party/ijar/... //scripts/... \
      || fail "Tests failed"
fi

clear_log
display "Build successful! Binary is here: ${BAZEL}"
