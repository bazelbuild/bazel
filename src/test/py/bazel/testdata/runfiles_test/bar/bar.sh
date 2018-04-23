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
# --- begin runfiles.bash initialization ---
if [[ "${RUNFILES_MANIFEST_ONLY:-}" != 1 && -z "${RUNFILES_DIR:-}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_ONLY=1
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_ONLY=1
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -d "$0.runfiles" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ "${RUNFILES_MANIFEST_ONLY:-}" == 1 && -f "${RUNFILES_MANIFEST_FILE:-}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
elif [[ -n "${RUNFILES_DIR:-}" && -d "${RUNFILES_DIR}" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

echo "Hello Bash Bar!"
echo "rloc=$(rlocation "foo_ws/bar/bar-sh-data.txt")"
