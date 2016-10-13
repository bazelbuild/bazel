#!/bin/bash
#
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

# This script takes in a regular expression and a zip file and writes a file
# containing the names of all files in the zip file that match the regular
# expression with one per line. Names of directories are not included.

if [ "$#" -ne 3 ]; then
  echo "Usage: zip_manifest_creator.sh <regexp> <input zip> <output manifest>"
  exit 1
fi

REGEX="$1"
INPUT_ZIP="$2"
OUTPUT_MANIFEST="$3"

RUNFILES=${RUNFILES:-$0.runfiles}

# For the sh_binary in BUILD.tools, zipper is here.
ZIPPER=$RUNFILES/*/tools/zip/zipper/zipper
if [ ! -x $ZIPPER ]; then
  # For the sh_test in BUILD.oss, zipper is here.
  ZIPPER=$RUNFILES/third_party/ijar/zipper
fi
if [ ! -x $ZIPPER ]; then
  echo "zip_manifest_creator could not find zipper executable"
  exit 1
fi

$ZIPPER v "$INPUT_ZIP" \
  | cut -d ' ' -f3 \
  | grep -v \/$ \
  | grep -x "$REGEX" \
  > "$OUTPUT_MANIFEST"
exit 0
