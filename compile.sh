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
