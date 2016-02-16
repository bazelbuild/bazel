#!/bin/bash

# Copyright 2016 The Bazel Authors. All rights reserved.
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
# limitations under the License.

set -eu

# This allows the script to be both a binary and a library script. If our binary has defined
# RUNFILES then we use it, otherwise we look for our own runfiles.
RUNFILES=${RUNFILES:-$0.runfiles}

function showfile {
  cat "${RUNFILES}/examples/shell/data/file.txt"
}

