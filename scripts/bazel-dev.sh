#!/usr/bin/env bash
#
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
#

set -eu

USAGE="$0 [<bazel arguments>...]"
DESCRIPTION='
  Rebuilds a development version of Bazel, if necessary, and then runs the
  given Bazel command using that binary.'

function usage() {
  echo "$USAGE" "$DESCRIPTION" >&2
}

# Configuration params. Export these in your bashrc to set personal defaults.

# The source of Bazel code.
BAZEL_REPO=${BAZEL_REPO:-https://github.com/bazelbuild/bazel}
# Where to keep the Bazel repository. If you make changes here, be warned that
# this script may overwrite or lose them.
PARENT_DIR="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"
BAZEL_DIR=${BAZEL_DIR:-${PARENT_DIR}}
# Bazel to use to build local bazel binaries.
BAZEL_BINARY=${BAZEL_BINARY:-$(which bazel)}

# The location of the resulting binary.
BAZEL_DEV="$BAZEL_DIR/bazel-bin/src/bazel-dev"

# First, check whether a rebuild is needed.
REBUILD=0
# If there is no built development Bazel, build it.
if [ ! -x "$BAZEL_DEV" ]; then
  REBUILD=1
fi
# If the current directory isn't the bazel working dir, always try to rebuild.
if [ "$(pwd)" != "$BAZEL_DIR" ]; then
  REBUILD=1
fi

# Perform a rebuild.
if [ "$REBUILD" == 1 ]; then
  echo -e "\033[31mBuilding dev version of bazel...\033[0m"
  (
    cd "$BAZEL_DIR"
    result=0
    ${BAZEL_BINARY} build //src:bazel-dev || result=$?
    if [[ $result != 0 ]]; then
      echo -e "\033[31mError building dev version of bazel.\033[0m"
      exit $result
    fi
  )
fi

# Execute bazel command.
echo -e "\e[31mExecuting bazel-dev...\e[0m"
exec $BAZEL_DEV "$@"

