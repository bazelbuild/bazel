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

# Check that the bintools can be found, otherwise we would see very confusing
# error messages.
# For example on Windows we would find "FIND.EXE" instead of "/usr/bin/find"
# when running "find".
hash tr >&/dev/null || {
  echo >&2 "ERROR: cannot locate GNU coreutils; check your PATH."
  echo >&2 "       You may need to run the following command:"
  echo >&2 "         export PATH=\"/bin:/usr/bin:\$PATH\""
  exit 1
}

# Ensure Python is on the PATH on Windows,otherwise we would see
# "LAUNCHER ERROR" messages from py_binary exe launchers.
case "$(uname -s | tr "[:upper:]" "[:lower:]")" in
msys*|mingw*|cygwin*)
  # Ensure Python is on the PATH, otherwise the bootstrapping fails later.
  which python.exe >&/dev/null || {
    echo >&2 "ERROR: cannot locate python.exe; check your PATH."
    echo >&2 "       You may need to run the following command, or something"
    echo >&2 "       similar, depending on where you installed Python:"
    echo >&2 "         export PATH=\"/c/Python27:\$PATH\""
    exit 1
  }
  # Ensure TMPDIR uses the user-specified TMPDIR or TMP or TEMP.
  # This is necessary to avoid overly longs paths during bootstrapping, see for
  # example https://github.com/bazelbuild/bazel/issues/4536
  export TMPDIR="${TMPDIR:-${TMP:-${TEMP:-}}}"
esac

cd "$(dirname "$0")"

# Set the default verbose mode in buildenv.sh so that we do not display command
# output unless there is a failure.  We do this conditionally to offer the user
# a chance of overriding this in case they want to do so.
: ${VERBOSE:=no}

source scripts/bootstrap/buildenv.sh

mkdir -p output
: ${BAZEL:=}

#
# Create an initial binary so we can host ourself
#
if [ ! -x "${BAZEL}" ]; then
  new_step 'Building Bazel from scratch'
  source scripts/bootstrap/compile.sh
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

new_step 'Building Bazel with Bazel'
display "."
log "Building output/bazel"
# We set host and target platform directly since the defaults in @bazel_tools
# have not yet been generated.
bazel_build "src:bazel_nojdk${EXE_EXT}" \
  --action_env=PATH \
  --host_platform=@bazel_tools//platforms:host_platform \
  --platforms=@bazel_tools//platforms:target_platform \
  || fail "Could not build Bazel"
bazel_bin_path="$(get_bazel_bin_path)/src/bazel_nojdk${EXE_EXT}"
[ -e "$bazel_bin_path" ] \
  || fail "Could not find freshly built Bazel binary at '$bazel_bin_path'"
cp -f "$bazel_bin_path" "output/bazel${EXE_EXT}" \
  || fail "Could not copy '$bazel_bin_path' to 'output/bazel${EXE_EXT}'"
chmod 0755 "output/bazel${EXE_EXT}"
BAZEL="$(pwd)/output/bazel${EXE_EXT}"

clear_log
display "Build successful! Binary is here: ${BAZEL}"
