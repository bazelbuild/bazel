#!/bin/bash

# Copyright 2019 The Bazel Authors. All rights reserved.
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

# This is a script which can be installed as your "bazel" binary instead of the
# real Bazel binary. When called, it tries to determine and run the correct
# Bazel version for a given project and forwards all arguments to it.
#
# You can specify which Bazel version to use using these methods:
# 1. Set $USE_BAZEL_VERSION to a version number
#    (e.g. export USE_BAZEL_VERSION=0.29.1).
# 2. Add a .bazelversion file that contains a version number next to your
#    WORKSPACE file.
# 3. Otherwise, the latest Bazel version will be used.
#
# This wrapper only recognizes Bazel versions installed next to itself, thus
# if you install this wrapper as /usr/bin/bazel, you'll have to install binaries
# for individual Bazel binaries as e.g. /usr/bin/bazel-0.29.1.
#
# In addition, if an executable called "tools/bazel" is found in the current
# workspace, this script will not directly execute Bazel, but instead store
# the path to the real Bazel executable in the environment variable BAZEL_REAL
# and then execute the "tools/bazel" wrapper script.
#
# In contrast to Bazelisk, this script does not download anything from the
# internet and instead relies on the local system to provide Bazel binaries.

function color() {
      # Usage: color "31;5" "string"
      # Some valid values for color:
      # - 5 blink, 1 strong, 4 underlined
      # - fg: 31 red,  32 green, 33 yellow, 34 blue, 35 purple, 36 cyan, 37 white
      # - bg: 40 black, 41 red, 44 blue, 45 purple
      printf '\033[%sm%s\033[0m\n' "$@"
}

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

function get_workspace_root() {
  workspace_dir="${PWD}"
  while [[ "${workspace_dir}" != / ]]; do
    if [[ -e "${workspace_dir}/WORKSPACE" ]]; then
      readonly workspace_dir
      return
    fi
    workspace_dir="$(dirname "${workspace_dir}")"
  done
  readonly workspace_dir=""
}

get_workspace_root

readonly wrapper_dir="$(dirname "$(get_realpath "${BASH_SOURCE[0]}")")"

function get_bazel_version() {
  if [[ -n ${USE_BAZEL_VERSION:-} ]]; then
    readonly reason="specified in \$USE_BAZEL_VERSION"
    readonly bazel_version="bazel-${USE_BAZEL_VERSION}"
  elif [[ -e "${workspace_dir}/.bazelversion" ]]; then
    readonly reason="specified in ${workspace_dir}/.bazelversion"
    read -r bazel_version < "${workspace_dir}/.bazelversion"
    readonly bazel_version="bazel-${bazel_version}"
  elif [[ -x "${wrapper_dir}/bazel-real" ]]; then
    readonly reason="automatically selected bazel-real"
    readonly bazel_version="bazel-real"
  else
    readonly reason="automatically selected latest available version"
    readonly bazel_version="$(basename $(find -H "${wrapper_dir}" -maxdepth 1 -name 'bazel-[0-9]*' -type f | sort -V | tail -n 1) 2>/dev/null)"
  fi
}

get_bazel_version

if [[ -z $bazel_version ]]; then
  (color "31" "ERROR: No installed Bazel version found, cannot continue."
  echo ""
  echo "Workspace root is: ${workspace_dir:-<empty>}"
  echo "Selected Bazel version is: ${bazel_version:-<empty>} (${reason:-<empty>})"
  echo ""
  echo "You can specify which Bazel version to use using these methods:"
  echo "1. Set \$USE_BAZEL_VERSION to a version number (e.g. export USE_BAZEL_VERSION=0.29.1)."
  echo "2. Add a .bazelversion file that contains a version number next to your WORKSPACE file."
  echo "3. Otherwise, the latest Bazel version installed as ${wrapper_dir}/bazel-* will be used."
  echo ""
  echo "For case 1 and 2, a binary called \"bazel-\$version\" (e.g. \"bazel-0.29.1\") must be present in ${wrapper_dir}."
  echo ""
  echo "You might be able to install specific Bazel versions via apt-get:"
  echo "  $ apt-get update && apt-get install bazel-0.29.1") >&2
  exit 1
fi

BAZEL_REAL="${wrapper_dir}/${bazel_version}"

if [[ ! -x $BAZEL_REAL ]]; then
  (color "31" "ERROR: Required Bazel binary \"${bazel_version}\" (${reason}) not found in ${wrapper_dir}."
  echo ""
  echo "Workspace root is: ${workspace_dir:-<empty>}"
  echo "Bazel version is: ${bazel_version:-<empty>} (${reason:-<empty>})"
  echo "Bazel binary is: ${BAZEL_REAL:-<empty>}"
  echo ""
  echo "If this is an officially released Bazel version, you might be able to install it:"
  echo "  $ apt-get update && apt-get install ${bazel_version}") >&2
  exit 1
fi

readonly wrapper="${workspace_dir}/tools/bazel"
if [[ -x "$wrapper" ]]; then
  export BAZEL_REAL
  exec -a "$0" "${wrapper}" "$@"
fi

exec -a "$0" "${BAZEL_REAL}" "$@"
