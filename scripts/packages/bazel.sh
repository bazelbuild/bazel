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

set -eu

# This is a script which is installed instead of the real Bazel binary.
# It looks for a tools/bazel executable next to the containing WORKSPACE
# file and runs that. If that's not found, it runs the real Bazel binary which
# is installed next to this script as bazel-real.

# `readlink -f` that works on OSX too.
function get_realpath() {
    if [ "$(uname -s)" == "Darwin" ]; then
        local queue="$1"
        if [[ "${queue}" != /* ]] ; then
            # Make sure we start with an absolute path.
            queue="${PWD}/${queue}"
        fi
        local current=""
        while [ -n "${queue}" ]; do
            # Removing a trailing /.
            queue="${queue#/}"
            # Pull the first path segment off of queue.
            local segment="${queue%%/*}"
            # If this is the last segment.
            if [[ "${queue}" != */* ]] ; then
                segment="${queue}"
                queue=""
            else
                # Remove that first segment.
                queue="${queue#*/}"
            fi
            local link="${current}/${segment}"
            if [ -h "${link}" ] ; then
                link="$(readlink "${link}")"
                queue="${link}/${queue}"
                if [[ "${link}" == /* ]] ; then
                    current=""
                fi
            else
                current="${link}"
            fi
        done

        echo "${current}"
    else
        readlink -f "$1"
    fi
}

BAZEL_REAL="$(dirname "$(get_realpath "${BASH_SOURCE[0]}")")/bazel-real"
export BAZEL_REAL

WORKSPACE_DIR="${PWD}"
while [[ "${WORKSPACE_DIR}" != / ]]; do
    if [[ -e "${WORKSPACE_DIR}/WORKSPACE" ]]; then
      break;
    fi
    WORKSPACE_DIR="$(dirname "${WORKSPACE_DIR}")"
done
readonly WORKSPACE_DIR

if [[ -e "${WORKSPACE_DIR}/WORKSPACE" ]]; then
  readonly WRAPPER="${WORKSPACE_DIR}/tools/bazel"

  if [[ -x "${WRAPPER}" ]]; then
    exec -a "$0" "${WRAPPER}" "$@"
  fi
fi

if [[ ! -x "${BAZEL_REAL}" ]]; then
    echo "Failed to find underlying Bazel executable at ${BAZEL_REAL}" >&2
    exit 1
fi

exec -a "$0" "${BAZEL_REAL}" "$@"
