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

USAGE='bazel-bisect.sh GOOD_COMMIT BAD_COMMIT [<bazel arguments>...]'
DESCRIPTION='
  Downloads a fresh copy of the Bazel source and runs git bisect from the
  specified commits, testing by running the given bazel command in the
  current working directory.'

function usage() {
  echo "$USAGE" "$DESCRIPTION" >&2
}

# Configuration params. Export these in your bashrc to set personal defaults.

# The source of Bazel code.
BAZEL_REPO=${BAZEL_REPO:-https://github.com/bazelbuild/bazel}
# Where to keep the Bazel repository. If you make changes here, be warned that
# this script may overwrite or lose them.
BAZEL_DIR=${BAZEL_DIR:-$HOME/os-bazel-bisect}
# Bazel to use to build local bazel binaries.
BAZEL_BINARY=${BAZEL_BINARY:-$(which bazel)}
# Whether to run 'bazel clean' between invocations.
# Set BAZEL_BISECT_WITH_CLEAN=y to clean build artifacts between runs.
BAZEL_BISECT_WITH_CLEAN=${BAZEL_BISECT_WITH_CLEAN:-n}

# Collect the arguments.
if [ "$#" -lt 3 ]; then
  usage
  exit 1
fi

# Collect the arguments.
#
GOOD_COMMIT="$1"
shift
BAD_COMMIT="$1"
shift
BAZEL_ARGUMENTS="$@"

echo "Bisecting bazel from good $GOOD_COMMIT to bad $BAD_COMMIT: bazel $BAZEL_ARGUMENTS"

# Check out and update Bazel.
if [ ! -d "$BAZEL_DIR" ]; then
  git clone "$BAZEL_REPO" "$BAZEL_DIR"
fi
# Ensure the repository is up to date.
(
  cd "$BAZEL_DIR"
  git fetch --tags
  git checkout master
)

# Run the actual bisect.
WORKING_DIR="$PWD"
(
  BISECT_SCRIPT=/tmp/bisect.sh
  TMP_BIN=/tmp/bazel.bisect

  cat >$BISECT_SCRIPT <<EOF
#!/usr/bin/env bash
set -eu
"$BAZEL_BINARY" build //src:bazel >/dev/null 2>&1 || exit 1
cp -f bazel-bin/src/bazel "$TMP_BIN"
cd "$WORKING_DIR"
# Call the newly built Bazel. If it fails, save the exit code but don't stop
# the script.
result=0
if [[ "${BAZEL_BISECT_WITH_CLEAN}" = "y" ]]; then
  "$TMP_BIN" clean || true
fi
"$TMP_BIN" $BAZEL_ARGUMENTS || result=\$? && true
"$TMP_BIN" shutdown || true
exit \$result
EOF
  chmod +x "$BISECT_SCRIPT"
  cd "$BAZEL_DIR"
  git bisect start
  git bisect good "$GOOD_COMMIT"
  git bisect bad  "$BAD_COMMIT"
  result=0
  git bisect run "$BISECT_SCRIPT" || result=$?
  git bisect reset
  exit $result
)
