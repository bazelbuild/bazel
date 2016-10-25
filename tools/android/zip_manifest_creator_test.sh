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

export RUNFILES=${RUNFILES:-$($(cd $(dirname ${BASH_SOURCE[0]})); pwd)}
CUT="$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)/zip_manifest_creator"
cd $TEST_TMPDIR

touch classes.jar
touch AndroidManifest.xml
mkdir -p res/values
touch res/values/bar.xml
touch res/values/baz.xml

zip -q foo.zip classes.jar
zip -q foo.zip AndroidManifest.xml
zip -q foo.zip res/values/bar.xml
zip -q foo.zip res/values/baz.xml

$CUT 'res/.*' foo.zip actual.manifest

cat > expected.manifest <<EOT
res/values/bar.xml
res/values/baz.xml
EOT

cmp expected.manifest actual.manifest
