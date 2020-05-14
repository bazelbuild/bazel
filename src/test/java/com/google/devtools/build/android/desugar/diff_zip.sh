#!/bin/bash
#
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
#

function zip_content {
  # compare CRC-32, striped out title line, sort by name column name.
  unzip -lv "$1" | grep -v 'META-INF/desugar_log/' | sort -k 8 | tail -n +5
}

diff -y <(zip_content "$1")  <(zip_content "$2")
