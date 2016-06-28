#!/bin/bash -eu

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

# This script generates a JavacBootclasspathLocations file from the
# bootclasspath available at build time.

OUT=$1
GENDIR=$2
STRIP_PREFIX=$3

shift 3

BOOTCLASSPATH=$(echo "$*" | \
  tr " " "\n" | \
  sed "s|^${GENDIR}/||" | \
  sed "s|external/||" | \
  sed "s|^|${STRIP_PREFIX}|" | \
  tr "\n" ":" | \
  sed "s/:$//"
)

cat > "$OUT" <<EOF
package com.google.devtools.build.java.bazel;
public class JavacBootclasspathLocations {
  public static final String BOOTCLASSPATH = "${BOOTCLASSPATH}";
}
EOF
