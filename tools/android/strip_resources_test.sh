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
set -eux

CUT="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/strip_resources"
ZIPPER=$(pwd)/$1


# set up input zip
cd $TEST_TMPDIR
mkdir i
echo MANIFEST > i/AndroidManifest.xml
echo CLASSES.DEX > i/classes.dex
echo ARSC > i/resources.arsc
(cd i; $ZIPPER c i.zip *)

# Invoke code under test
$CUT --input_resource_apk i/i.zip --output_resource_apk o.zip

# Unpack output zip
mkdir o
(cd o; $ZIPPER x ../o.zip)

# Set read permission
chmod u+r o/AndroidManifest.xml

# Check if AndroidManifest.xml is unchanged and that no other files are present
cmp i/AndroidManifest.xml o/AndroidManifest.xml
[[ ! -e o/classes.dex ]]
[[ ! -e o/resources.arsc ]]

exit 0
