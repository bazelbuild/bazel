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

set -euo pipefail

directory_structure="$1"; shift
output="$1"; shift

initial_pwd="$(pwd)"

tmp_dir=$(mktemp -d -t)
tmp_zip="$tmp_dir/archive.zip"

if [[ "$directory_structure" == "nodir" ]]; then
  for curr_zip in "$@"
  do
    unzip -q "$curr_zip" -d "$tmp_dir"
  done

  cd "$tmp_dir"
  zip -r -q "$tmp_zip" "."
else
  mkdir -p "$tmp_dir/$directory_structure"
  for curr_zip in "$@"
  do
    unzip -q "$curr_zip" -d "$tmp_dir/$directory_structure"
  done

  cd "$tmp_dir"
  zip -r -q "$tmp_zip" "$directory_structure"
fi

cd "$initial_pwd"
mv -f "$tmp_zip" "$output"
rm -r "$tmp_dir"







