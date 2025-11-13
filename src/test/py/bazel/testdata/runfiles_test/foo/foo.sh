#!/usr/bin/env bash
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

# --- begin runfiles.bash initialization v2 ---
# Copy-pasted from the Bazel Bash runfiles library v2.
set -uo pipefail; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
  source "$0.runfiles/$f" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
  { echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v2 ---

if ! type rlocation >&/dev/null; then
  echo >&2 "ERROR: rlocation is undefined"
  exit 1
fi

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  is_windows=true
  ;;
*)
  is_windows=false
  ;;
esac

function child_binary_name() {
  local lang=$1
  if "$is_windows"; then
    echo "_main/bar/bar-${lang}.exe"
  else
    echo "_main/bar/bar-${lang}"
  fi
}

function main() {
  echo "Hello Bash Foo!"
  echo "rloc=$(rlocation "_main/foo/datadep/hello.txt")"

  # Run a subprocess, propagate the runfiles envvar to it. The subprocess will
  # use this process's runfiles manifest or runfiles directory.
  runfiles_export_envvars
  if "$is_windows"; then
    export SYSTEMROOT="${SYSTEMROOT:-}"
  fi
  for lang in py java sh cc; do
    child_bin="$(rlocation "$(child_binary_name $lang)")"
    if ! "$child_bin"; then
      echo >&2 "ERROR: error running bar-$lang"
      exit 1
    fi
  done
}

main
