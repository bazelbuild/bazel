#!/bin/bash
#
# Copyright 2017 The Bazel Authors. All rights reserved.
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
# discard_graph_edges_lib.sh: functions needed by discard_graph_edges_test.sh

STARTUP_FLAGS="--batch"
BUILD_FLAGS="--discard_analysis_cache --notrack_incremental_state"

function extract_histogram_count() {
  local histofile="$1"
  local item="$2"
  # We can't use + here because Macs don't recognize it as a special character
  # by default.
  (grep "$item" "$histofile" || echo "") \
      | sed -e 's/^ *[0-9][0-9]*: *\([0-9][0-9]*\) .*$/\1/'
}
