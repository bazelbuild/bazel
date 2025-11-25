#!/bin/bash
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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

function print_message_and_exit() {
  echo $1 >&2; exit 1;
}

if type rlocation >&/dev/null; then
  # An incomplete rlocation function is defined in Bazel's test-setup.sh script,
  # load the actual Bash runfiles library from @bazel_tools//tools/bash/runfiles
  # to make sure repo mappings is respected.
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

  # Load the unit test framework
  source "$(rlocation io_bazel/src/test/shell/unittest.bash)" \
    || print_message_and_exit "unittest.bash not found!"
  # Load the test environment
  source "$(rlocation io_bazel/src/test/shell/testenv.sh)" \
    || print_message_and_exit "testenv.sh not found!"
else
  # If rlocation is undefined, we are probably running under Blaze.
  # Assume the existence of a runfiles tree.

  CURRENT_SCRIPT=${BASH_SOURCE[0]}
  # Go to the directory where the script is running
  cd "$(dirname ${CURRENT_SCRIPT})" \
    || print_message_and_exit "Unable to access $(dirname ${CURRENT_SCRIPT})"

  DIR=$(pwd)
  # Load the unit test framework
  source "$DIR/unittest.bash" || print_message_and_exit "unittest.bash not found!"
  # Load the test environment
  source "$DIR/testenv.sh" || print_message_and_exit "testenv.sh not found!"
fi

# inplace-sed: a version of sed -i that actually works on Linux and Darwin.
# https://unix.stackexchange.com/questions/92895
# https://stackoverflow.com/questions/5694228
function inplace-sed() {
  if [ $(uname) = "Darwin" ]; then
    sed -i "" "$@"
  else
    sed -i "$@"
  fi
}
