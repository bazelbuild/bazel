#!/bin/bash
#
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
#
# swiftstdlibtoolwrapper runs swift-stdlib-tool and zips up the output.
# This script only runs on darwin and you must have Xcode installed.
#
# --output_zip_path - the path to place the output zip file.
# --bundle_path - the path inside of the archive to where libs will be copied.
# --toolchain - toolchain identifier to use with xcrun.

set -eu

MY_LOCATION=${MY_LOCATION:-"$0.runfiles/bazel_tools/tools/objc"}
REALPATH="${MY_LOCATION}/realpath"
WRAPPER="${MY_LOCATION}/xcrunwrapper.sh"

while [[ $# -gt 0 ]]; do
  ARG="$1"
  shift
  case "${ARG}" in
    --output_zip_path)
      ARG="$1"
      shift
      OUTZIP=$("${REALPATH}" "${ARG}")
      ;;
    --bundle_path)
      ARG="$1"
      shift
      PATH_INSIDE_ZIP="$ARG"
      ;;
    --toolchain)
      ARG="$1"
      shift
      TOOLCHAIN=${ARG}
      ;;
    --scan-executable)
      ARG="$1"
      shift
      BINARY=${ARG}
      ;;
    --platform)
      ARG="$1"
      shift
      PLATFORM=${ARG}
      ;;
    esac
done

# Prepare destination directory
TEMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/swiftstdlibtoolZippingOutput.XXXXXX")
trap "rm -rf \"$TEMPDIR\"" EXIT

FULLPATH="$TEMPDIR/$PATH_INSIDE_ZIP"
mkdir -p "${FULLPATH}"

if [ -n "${TOOLCHAIN:-}" ]; then
  readonly swiftc_dir=$(dirname "$("${WRAPPER}" -f swiftc --toolchain "${TOOLCHAIN}")")
else
  readonly swiftc_dir=$(dirname "$("${WRAPPER}" -f swiftc)")
fi

# Each toolchain has swift libraries directory located at
# /path/to/swiftc/../../lib/swift/<platform>/
# This is the same relative path that Xcode uses and is considered to be stable.
readonly platform_dir="${swiftc_dir}/../lib/swift/${PLATFORM}"

# Always use macOS system Python as it comes with macholib module.
/usr/bin/python "${MY_LOCATION}/swift_stdlib_tool.py" "${BINARY}" "${platform_dir}" "${FULLPATH}"

# Need to push/pop tempdir so it isn't the current working directory
# when we remove it via the EXIT trap.
pushd "$TEMPDIR" > /dev/null
# Reset all dates to Zip Epoch so that two identical zips created at different
# times appear the exact same for comparison purposes.
find . -exec touch -h -t 198001010000 {} \;

# Added include "*" to fix case where we may want an empty zip file because
# there is no data.
zip --compression-method store --symlinks --recurse-paths --quiet "$OUTZIP" . --include "*"
popd > /dev/null
