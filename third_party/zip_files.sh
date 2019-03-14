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

# A script that zips all the inputs files under the given directory structure
# in the output zip file.

# Usage: third_party/zip_files.sh directory_structure output_zip [input_files]
#
# For example: third_party_zip_files.sh src/main/cpp my_archive.zip a.cc b.cc
# will create the archive my_archive.zip containing:
# src/main/cpp/a.cc
# src/main/cpp/b.cc

set -euo pipefail

directory_structure="$1"; shift
output="$1"; shift

initial_pwd="$(pwd)"

tmp_dir=$(mktemp -d -t 'tmpdirXXXXX')
tmp_zip="$tmp_dir/archive.zip"

zip -j -q "$tmp_zip" "$@"

mkdir -p "$tmp_dir/$directory_structure"
cd "$tmp_dir/$directory_structure"
unzip -q "$tmp_zip"
rm -f "$tmp_zip"
cd "$tmp_dir"
zip -r -q "$tmp_zip" "$directory_structure"

cd "$initial_pwd"
mv -f "$tmp_zip" "$output"
rm -r "$tmp_dir"

