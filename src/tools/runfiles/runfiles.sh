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

# This script defines utility functions to handle sh_binary runfiles.
#
# On Windows, this script needs $RUNFILES_MANIFEST_FILE to point to the absolute
# path of the runfiles manifest file. If the envvar is undefined or empty, this
# script calls "exit 1".
#
# On Linux/macOS, this script needs $RUNFILES_DIR to point to the absolute path
# of the runfiles directory. If the envvar is undefined or empty, this script
# tries to determine the value by looking for the nearest "*.runfiles" parent
# directory of "$0", and if not found, this script calls "exit 1".

set -eu

# Check that we can find the bintools, otherwise we would see confusing errors.
stat "$0" >&/dev/null || {
  echo >&2 "ERROR[runfiles.sh]: cannot locate GNU coreutils; check your PATH."
  echo >&2 "    You may need to run 'export PATH=/bin:/usr/bin:\$PATH' (on Linux/macOS)"
  echo >&2 "    or 'set PATH=c:\\tools\\msys64\\usr\\bin;%PATH%' (on Windows)."
  exit 1
}

# Now that we have bintools on PATH, determine the current platform and define
# `is_windows` accordingly.
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  function is_windows() {
    true
  }
  ;;
*)
  function is_windows() {
    false
  }
  ;;
esac
export -f is_windows

# Define `is_absolute` unless already defined.
if ! type is_absolute &>/dev/null; then
  function is_absolute() {
    if is_windows; then
      echo "$1" | grep -q "^[a-zA-Z]:[/\\]"
    else
      [[ "$1" = /* ]]
    fi
  }
  export -f is_absolute
fi

# Define `rlocation` unless already defined.
if ! type rlocation &>/dev/null; then
  if is_windows; then
    # If RUNFILES_MANIFEST_FILE is empty/undefined, bail out.
    # On Windows there's no runfiles tree with symlinks like on Linux/macOS, so
    # we cannot locate the runfiles root and the manifest by walking the path
    # of $0.
    if [[ -z "${RUNFILES_MANIFEST_FILE:-}" ]]; then
      echo >&2 "ERROR[runfiles.sh]: RUNFILES_MANIFEST_FILE is empty/undefined"
      exit 1
    fi

    # Read the runfiles manifest to memory, to quicken runfiles lookups.
    # First, read each line of the manifest into `runfiles_lines`. We need to do
    # this while IFS is still the newline character. In the subsequent loop,
    # after we reset IFS, we can construct the `line_split` arrays.
    old_ifs="${IFS:-}"
    IFS=$'\n'
    runfiles_lines=( $(sed -e 's/\r//g' "$RUNFILES_MANIFEST_FILE") )
    IFS="$old_ifs"
    # Now create a dictionary from `runfiles_lines`. Creating `line_split` uses
    # $IFS so we could not have done this without a helper array.
    declare -A runfiles_dict
    for line in "${runfiles_lines[@]}"; do
      line_split=($line)
      runfiles_dict[${line_split[0]}]="${line_split[@]:1}"
    done
  else
    # If RUNFILES_DIR is empty/undefined, try locating the runfiles directory.
    # When the user runs a sh_binary's output directly, it's just a symlink to
    # the main script. There's no launcher like on Windows which would set this
    # environment variable.
    # Walk up the path of $0 looking for a runfiles directory.
    if [[ -z "${RUNFILES_DIR:-}" ]]; then
      RUNFILES_DIR="$(dirname "$0")"
      while [[ "$RUNFILES_DIR" != "/" ]]; do
        if [[ "$RUNFILES_DIR" = *.runfiles ]]; then
          break
        else
          RUNFILES_DIR="$(dirname "$RUNFILES_DIR")"
        fi
      done
      if [[ "$RUNFILES_DIR" = "/" ]]; then
        echo >&2 "ERROR[runfiles.sh]: RUNFILES_DIR is empty/undefined, and cannot find a"
        echo >&2 "    runfiles directory on the path of this script"
        exit 1
      fi
    fi
  fi

  function rlocation() {
    if is_absolute "$1"; then
      echo "$1"
    else
      if is_windows; then
        echo "${runfiles_dict[$1]}"
      else
        echo "${RUNFILES_DIR}/$1"
      fi
    fi
  }
  export -f rlocation
fi
