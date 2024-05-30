#!/bin/bash
# Copyright 2020 The Bazel Authors. All rights reserved.
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

set -eu

echo 'module "crosstool" [system] {'

if [[ "$OSTYPE" == darwin* ]]; then
  for dir in $@; do
    find "$dir" -type f \( -name "*.h" -o -name "*.def" -o -path "*/c++/*" \) \
      | LANG=C sort -u | while read -r header; do
        echo "  textual header \"${header}\""
      done
  done
else
  for dir in $@; do
    find -L "${dir}" -type f 2>/dev/null | LANG=C sort -u | while read -r header; do
      echo "  textual header \"${header}\""
    done
  done
fi

echo "}"
