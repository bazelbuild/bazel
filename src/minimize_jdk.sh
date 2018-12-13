#!/bin/bash

# Copyright 2018 The Bazel Authors. All rights reserved.
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

# This script creates from the full JDK a minimized version that only contains
# the specified JDK modules.

set -euo pipefail

fulljdk=$1
modules=$(cat "$2" | paste -sd "," - | tr -d '\r')
out=$3

UNAME=$(uname -s | tr 'A-Z' 'a-z')

if [[ "$UNAME" =~ msys_nt* ]]; then
  set -x
  unzip "$fulljdk"
  cd zulu*
  echo -e "MODULES: >>$modules<<\n"
  ./bin/jlink --module-path ./jmods/ --add-modules "$modules" \
    --vm=server --strip-debug --no-man-pages \
    --output reduced
  cp DISCLAIMER readme.txt reduced/
  zip -r -9 ../reduced.zip reduced/
  cd ..
  mv reduced.zip "$out"
else
  tar xf "$fulljdk"
  cd zulu*
  ./bin/jlink --module-path ./jmods/ --add-modules "$modules" \
    --vm=server --strip-debug --no-man-pages \
    --output reduced
  cp DISCLAIMER readme.txt reduced/
  GZIP=-9 tar -zcf ../reduced.tgz reduced
  cd ..
  mv reduced.tgz "$out"
fi
