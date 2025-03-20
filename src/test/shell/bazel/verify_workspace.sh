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

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

WORKSPACE_FILES=("$(rlocation io_bazel/workspace_deps.bzl)")

# base maven repository URLs can return 404s.
URL_ALLOWLIST=("https://dl.google.com/android/maven2" "https://repo1.maven.org/maven2")

function test_verify_urls() {
  # Find url-shaped lines, skipping jekyll-tree (which isn't a valid URL), and
  # skipping comments.
  invalid_urls=()
  checked_urls=()
  for file in "${WORKSPACE_FILES[@]}"; do
    for url in $(grep -E '"https://|http://' "${file}" | \
      sed -e '/jekyll-tree/d' -e '/^#/d' -r -e  's#^.*"(https?://[^"]+)".*$#\1#g' | \
      sort -u); do
      if [[ " ${URL_ALLOWLIST[*]} " =~ " ${url} " ]]; then
        continue
      fi
      # add only unique url to the array
      if [[ ${#checked_urls[@]} == 0 ]] || [[ ! " ${checked_urls[@]} " =~ " ${url} " ]]; then
        checked_urls+=("${url}")
        echo "Checking ${url} ..."
        if ! curl --head --silent --show-error --fail --output /dev/null --retry 3 "${url}"; then
          invalid_urls+=("${url}")
        fi
      fi
    done
  done

  if [[ ${#invalid_urls[@]} > 0 ]]; then
    fail "Invalid urls: ${invalid_urls[@]}"
  fi
}

run_suite "verify_workspace"
