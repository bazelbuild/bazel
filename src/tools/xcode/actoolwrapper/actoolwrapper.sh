#!/bin/bash
# Copyright 2015 Google Inc. All rights reserved.
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
# actoolwrapper runs actool and zips up the output.
# This script only runs on darwin and you must have Xcode installed.
#
# $1 OUTZIP - the path to place the output zip file.

set -eu

OUTZIP=$(tools/objc/realpath "$1")
shift 1
TEMPDIR=$(mktemp -d -t ZippingOutput)
trap "rm -rf \"$TEMPDIR\"" EXIT

# actool needs to have absolute paths sent to it, so we call realpaths on
# on all arguments seeing if we can expand them.
# actool and ibtool appear to depend on the same code base.
# --output-partial-info-plist gives actool real troubles
# so we create a file where we expect the plist to be so we can get a full
# path to it.
# Radar 21045660 ibtool has difficulty dealing with relative paths.
TOOLARGS=()
LASTARG=""
for i in $@; do
  if [ "$LASTARG" = "--output-partial-info-plist" ]; then
    PARENTDIR=$(dirname "$i")
    mkdir -p "$PARENTDIR"
    touch "$i"
  fi
  if [ -e "$i" ]; then
    TOOLARGS+=($(tools/objc/realpath "$i"))
  else
    TOOLARGS+=("$i")
  fi
  LASTARG="$i"
done

# If we are running into problems figuring out actool issues, there are a couple
# of env variables that may help. Both of the following must be set to work.
#   IBToolDebugLogFile=<OUTPUT FILE PATH>
#   IBToolDebugLogLevel=4
# you may also see if
#   IBToolNeverDeque=1
# helps.
# Yes IBTOOL appears to be correct here due to actool and ibtool being based
# on the same codebase.
/usr/bin/xcrun actool --errors --warnings --notices \
    --compress-pngs --output-format human-readable-text \
    --compile "$TEMPDIR" "${TOOLARGS[@]}"

# Need to push/pop tempdir so it isn't the current working directory
# when we remove it via the EXIT trap.
pushd "$TEMPDIR" > /dev/null
# Reset all dates to Unix Epoch so that two identical zips created at different
# times appear the exact same for comparison purposes.
find . -exec touch -h -t 197001010000 {} \;
zip --symlinks --recurse-paths --quiet "$OUTZIP" .
popd > /dev/null
