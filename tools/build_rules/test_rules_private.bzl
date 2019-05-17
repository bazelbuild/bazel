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

"""Bash runfiles library init code for test_rules.bzl."""

# Init code to load the runfiles.bash file.
# The runfiles library itself defines rlocation which you would need to look
# up the library's runtime location, thus we have a chicken-and-egg problem.
INIT_BASH_RUNFILES = [
    "# --- begin runfiles.bash initialization ---",
    "# Copy-pasted from Bazel Bash runfiles library (tools/bash/runfiles/runfiles.bash).",
    "set -euo pipefail",
    'if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then',
    '  if [[ -f "$0.runfiles_manifest" ]]; then',
    '    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"',
    '  elif [[ -f "$0.runfiles/MANIFEST" ]]; then',
    '    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"',
    '  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then',
    '    export RUNFILES_DIR="$0.runfiles"',
    "  fi",
    "fi",
    'if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then',
    '  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"',
    'elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then',
    '  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \\',
    '            "$RUNFILES_MANIFEST_FILE" | cut -d " " -f 2-)"',
    "else",
    '  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"',
    "  exit 1",
    "fi",
    "# --- end runfiles.bash initialization ---",
]

# Label of the runfiles library.
BASH_RUNFILES_DEP = "@bazel_tools//tools/bash/runfiles"
