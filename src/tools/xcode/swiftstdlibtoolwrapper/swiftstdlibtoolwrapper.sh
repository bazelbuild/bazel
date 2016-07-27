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
# $1 OUTZIP - the path to place the output zip file.

set -eu

MY_LOCATION=${MY_LOCATION:-"$0.runfiles/bazel_tools/tools/objc"}
REALPATH="${MY_LOCATION}/realpath"
WRAPPER="${MY_LOCATION}/xcrunwrapper.sh"

OUTZIP=$("${REALPATH}" "$1")
shift 1
TEMPDIR=$(mktemp -d "${TMPDIR:-/tmp}/swiftstdlibtoolZippingOutput.XXXXXX")
trap "rm -rf \"$TEMPDIR\"" EXIT

FULLPATH="$TEMPDIR/Frameworks"

$WRAPPER swift-stdlib-tool --copy --verbose --destination "$FULLPATH" "$@"

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
