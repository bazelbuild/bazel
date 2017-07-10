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

set -eux

export RUNFILES=$TEST_SRCDIR

IS_WINDOWS=false
case "$(uname | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  IS_WINDOWS=true
esac

if "$IS_WINDOWS"; then
  CUT="$(rlocation "[^/]*/tools/android/zip_manifest_creator")"
  ZIPPER="$(rlocation "[^/]*/$1")"
else
  CUT="$(find "${RUNFILES}" -path "*/tools/android/zip_manifest_creator")"
  ZIPPER="$(find "${RUNFILES}" -path "*/$1")"
fi

cd $TEST_TMPDIR

touch classes.jar
touch AndroidManifest.xml
mkdir -p res/values
touch res/values/bar.xml
touch res/values/baz.xml

"$ZIPPER" c foo.zip classes.jar AndroidManifest.xml res/values/*

"$CUT" 'res/.*' foo.zip actual.manifest

cat > expected.manifest <<EOT
res/values/bar.xml
res/values/baz.xml
EOT

# On Windows: you can install `cmp` using `pacman -Syu diffutils`.
cmp expected.manifest actual.manifest
