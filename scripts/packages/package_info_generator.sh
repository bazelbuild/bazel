#!/usr/bin/env bash

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

# Generate a README.md for the package from the information provided
# by the build status command.

# Store the build status information we care about
release_name=
git_hash=
url=
built_by=
build_log=
release_notes=

for i in "${@}"; do
  while read line; do
    key=$(echo "$line" | cut -d " " -f 1)
    value="$(echo "$line" | cut -d " " -f 2- | tr '\f' '\n')"
    case $key in
      RELEASE_NAME)
        release_name="$value"
        ;;
      RELEASE_GIT_HASH)
        git_hash="$value"
        ;;
      RELEASE_BUILT_BY)
        built_by="$value"
        ;;
      RELEASE_BUILD_LOG)
        build_log="$value"
        ;;
      RELEASE_NOTES)
        release_notes="$value"
        ;;
      RELEASE_COMMIT_URL)
        commit_url="$value"
        ;;
   esac
  done <<<"$(cat $i)"
done

url="${url:-https://github.com/bazelbuild/bazel/commit/${git_hash}}"

if [ -z "${release_name}" ]; then
  # Not a release
  echo "# Binary package at HEAD (@$git_hash)"
else
  echo "# ${release_notes}"  # Make the first line the header
  # Subsection for environment
  echo
  echo "## Build information"
fi
if [ -n "${built_by-}" ]; then
  if [ -n "${build_log}" ]; then
    echo "   - [Built by ${built_by}](${build_log})"
  else
    echo "   - Built by ${built_by}"
  fi
elif [ -n "${build_log-}" ]; then
  echo "   - [Build log](${build_log})"
fi

echo "   - [Commit](${url})"
