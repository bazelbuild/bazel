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

shift 2

# The Bazel and Google-internal locations of the bootclasspath entries are
# unfortunately subtly different. The "local_jdk" rewrite is for Bazel, and
# the "third_party" rewrite is for Google.
# TODO(ulfjack): Find a way to unify this.
BOOTCLASSPATH=$(echo "$*" | \
  tr " " "\n" | \
  sed "s|^${GENDIR}/||" | \
  sed "s|^.*local_jdk|local_jdk|" | \
  sed "s|^third_party|${PWD##*/}/third_party|" | \
  sed "s|^tools/jdk|${PWD##*/}/tools/jdk|" | \
  tr "\n" ":" | \
  sed "s/:$//"
)

cat > "$OUT" <<EOF
package com.google.devtools.build.java.bazel;
public class JavacBootclasspathLocations {
  public static final String BOOTCLASSPATH = "${BOOTCLASSPATH}";
}
EOF
