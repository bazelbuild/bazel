#!/usr/bin/env bash
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
# Test that all sources in Bazel are contained in the //:srcs filegroup
# Actually this test just compares the two input (the file list and the
# //:srcs filegroup and show the diff)

# --- begin runfiles.bash initialization ---
# Copy-pasted from Bazel's Bash runfiles library (tools/bash/runfiles/runfiles.bash).
set -euo pipefail
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
    if [[ -f "$0.runfiles_manifest" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
    elif [[ -f "$0.runfiles/MANIFEST" ]]; then
      export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
    elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
      export RUNFILES_DIR="$0.runfiles"
    fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

LIST_SRCS="$(rlocation "local_bazel_source_list/sources.txt")"
SRCS_QUERY="$(mktemp)"

# Rewrite labels to file paths, ignoring all external repos.
cat "$(rlocation "io_bazel/src/test/shell/bazel/srcs_list")" \
  | sed -e '/^@@/d' \
  | sed -e 's|^//||' | sed -e 's|^:||' | sed -e 's|:|/|' \
  | sort -u >"${SRCS_QUERY}"

res="$(diff -U 0 "${LIST_SRCS}" "${SRCS_QUERY}" | sed -e 's|^-||' \
  | grep -Ev '^(@|\+|-)' || true)"

if [ -n "${res}" ]; then
  echo "//:srcs filegroup do not contains all the sources, missing:
${res}"
  exit 1
fi
