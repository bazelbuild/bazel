#!/bin/bash

# Copyright 2023 The Bazel Authors. All rights reserved.
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
# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
source "$0.runfiles/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
{ echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_verify_lock_file() {
  rm -f MODULE.bazel
  touch MODULE.bazel
  echo 'common --incompatible_use_plus_in_repo_names' > .bazelrc
  echo "Running: bazel mod deps --lockfile_mode=update to generate the lockfile."
  bazel mod deps --lockfile_mode=update
  diff -u $(rlocation io_bazel/src/test/tools/bzlmod/MODULE.bazel.lock) MODULE.bazel.lock || fail "Default lockfile for empty workspace is no longer in sync with MODULE.tools. Please run \"bazel run //src/test/tools/bzlmod:update_default_lock_file\""

  # Verify if python toolchain version matches Bazel's lock file to ensure it's cached in integration tests.
  # Check strings like `"default_version": "3.11"`` for default python version.
  python_version=$(grep -o '"default_version": "[0-9.]*"' MODULE.bazel.lock | grep -o '[0-9.]*')
  echo "Extracted Python version: $python_version from default lockfile"

  root_lockfile=$(rlocation io_bazel/MODULE.bazel.lock)
  python_version_root=$(grep -o '"default_version": "[0-9.]*"' $root_lockfile | grep -o '[0-9.]*')
  echo "Extracted Python version: $python_version_root from root lockfile"

  if [ "$python_version" != "$python_version_root" ]; then
    fail "Python version in default lockfile does not match root lockfile. Please update Python toolchain version in the root MODULE.bazel file to match."
  fi

  # Verify MODULE.tools with --check_direct_dependencies=error
  echo "Running: bazel mod deps --check_direct_dependencies for MODULE.tools"
  cp $(rlocation io_bazel/src/MODULE.tools) MODULE.bazel
  sed -i.bak '/module(name = "bazel_tools")/d' MODULE.bazel
  bazel mod deps --check_direct_dependencies=error || fail "Please update MODULE.tools to match the resolved versions."
}

run_suite "test_verify_lock_file"
