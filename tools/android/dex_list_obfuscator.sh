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

# Dummy dex list obfuscator doing nothing
# Should be updated to contain an app, that can obfuscate main dex keep list
# according to the proguard map.

set -eu
input=
output=
while [[ "$#" -gt 0 ]]; do
  arg="$1"; shift;
  case "${arg}" in
    --input) input="$1"; shift ;;
    --output) output="$1"; shift ;;
    ---obfuscation_map=*) shift ;;
    *) echo "Unknown flag: ${arg}"; exit 1 ;;
  esac
done

echo "WARNING: This is just no-op version of the list obfuscator."
echo "It is invoked, because main_dex_list and proguard were both used."
echo "If proguard obfuscates a class, it will not be kept in the main dex even"
echo "if the original name was in the main_dex_list."
echo "The main_dex_list (provided as --input) should be obfuscated using the"
echo "map provided as --obfuscation_map parameter."
echo "If no obfuscation of main dex classes is performed, then noop is OK."

cp $input $output
