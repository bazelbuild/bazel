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

# This script generates a JavaLangtoolsLocation class from the langtools
# available at build time.

OUT=$1
STRIP_PREFIX=$2

shift 2

# We add the current workspace name as a prefix here, and we use the current
# directory name for that. This might be a bit brittle.
FILE="$(echo "$*" | \
    sed "s|^${STRIP_PREFIX}/||" | \
    sed "s|^third_party|${PWD##*/}/third_party|" \
)"
cat > "$OUT" <<EOF
package com.google.devtools.build.java.bazel;
public class JavaLangtoolsLocation {
  public static final String FILE = "${FILE}";
}
EOF
