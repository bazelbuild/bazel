#!/bin/bash
# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

set -eu

android_jar=
binary_jar=
dest=
while [[ "$#" -gt 0 ]]; do
  arg="$1"; shift;
  case "${arg}" in
    --binary) binary_jar="$1"; shift ;;
    --binary=*) binary_jar="${arg:9}" ;;
    --output) dest="$1"; shift ;;
    --output=*) dest="${arg:9}" ;;
    --android_jar) android_jar="$1"; shift ;;
    --android_jar=*) android_jar="${arg:14}" ;;
    *) echo "Unknown flag: ${arg}"; exit 1 ;;
  esac
done

# TODO(b/77339644): Support minification
cp "$(dirname "$0")/java8_legacy.dex.zip" "${dest}"
