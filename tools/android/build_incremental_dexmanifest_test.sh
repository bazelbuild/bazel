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

CUT="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/build_incremental_dexmanifest"
ZIPPER=$(pwd)/$1

cd $TEST_TMPDIR
echo 1 > 1.dex
echo 2 > 2.dex
echo 3 > 3.dex
echo 4 > 4.dex

$ZIPPER c 1.zip 1.dex 2.dex
$ZIPPER c 2.zip 3.dex

$CUT output.manifest 1.zip 2.zip 4.dex
IFS=$'\n' MANIFEST=($(cat output.manifest | cut -d' ' -f1-3))
[[ ${MANIFEST[0]} == "1.zip 1.dex incremental_classes1.dex" ]]
[[ ${MANIFEST[1]} == "1.zip 2.dex incremental_classes2.dex" ]]
[[ ${MANIFEST[2]} == "2.zip 3.dex incremental_classes3.dex" ]]
[[ ${MANIFEST[3]} == "4.dex - incremental_classes4.dex" ]]
