#!/usr/bin/env bash

# Copyright 2019 The Bazel Authors. All rights reserved.
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

# A script that zips the content of the inputs zip files under a given directory
# structure in the output zip file. "-" can be passed if no top-level directory
# structure is required.

# Usage: third_party/merge_zip_files.sh directory_prefix output_zip [input_zip_files]
#
# For example, if we have the following zips and their content:
# a.zip:
#    dir1/a1.cc
#    a2.cc
# b.zip:
#    dir2/b1.cc
#    b2.cc
#
# third_party_zip_files.sh src/main/cpp my_archive.zip a.zip b.zip
# will create the archive my_archive.zip containing:
# src/main/cpp/a2.cc
# src/main/cpp/b2.cc
# src/main/cpp/dir1/a1.cc
# src/main/cpp/dir2/b1.cc
#
# third_party_zip_files.sh - my_archive.zip a.zip b.zip
# will create the archive my_archive.zip containing:
# a2.cc
# b2.cc
# dir1/a1.cc
# dir2/b1.cc

set -euo pipefail

directory_prefix="$1"; shift
output="$1"; shift

initial_pwd="$(pwd)"

tmp_dir=$(mktemp -d -t 'tmp_bazel_zip_files_XXXXXX')
trap "rm -fr $tmp_dir" EXIT
tmp_zip="$tmp_dir/archive.zip"

if [[ "$directory_prefix" == "-" ]]; then
  for curr_zip in "$@"
  do
    unzip -q -o "$curr_zip" -d "$tmp_dir"
  done

  cd "$tmp_dir"
  zip -9 -r -q "$tmp_zip" "."
else
  mkdir -p "$tmp_dir/$directory_prefix"
  for curr_zip in "$@"
  do
    unzip -q -o "$curr_zip" -d "$tmp_dir/$directory_prefix"
  done

  cd "$tmp_dir"
  zip -9 -r -q "$tmp_zip" "$directory_prefix"
fi

cd "$initial_pwd"
mv -f "$tmp_zip" "$output"
