#!/bin/sh
#
# Copyright 2026 The Bazel Authors. All rights reserved.
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

# Produces the same output as src/tools/launcher/launcher_maker.cc on execution
# platforms without a prebuilt launcher maker.
#
# Usage: launcher_maker.sh <launcher> <launch info params file> <output>

set -eu

cat "$1" >"$3"

# Append each line of the launch info, terminated by a NUL byte.
while IFS= read -r line || [ -n "$line" ]; do
  printf '%s\000' "$line"
done <"$2" >>"$3"

# Append the length of the launch info as a 64-bit little-endian integer. wc
# counts bytes, whereas ${#line} counts characters in some shells.
bytes=$(($(wc -c <"$3") - $(wc -c <"$1")))
size=""
for _ in 1 2 3 4 5 6 7 8; do
  size="$size\\$(printf '%03o' $((bytes % 256)))"
  bytes=$((bytes / 256))
done
printf "$size" >>"$3"
