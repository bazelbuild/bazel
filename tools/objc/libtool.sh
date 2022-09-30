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
# libtool.sh runs the command passed to it using "xcrunwrapper libtool".
#
# It creates symbolic links for all input files with a path-hash appended
# to their original name (foo.o becomes foo_{md5sum}.o). This is to circumvent
# a bug in the original libtool that arises when two input files have the same
# base name (even if they are in different directories).

set -eu

# A trick to allow invoking this script in multiple contexts.
if [ -z ${MY_LOCATION+x} ]; then
  if [ -d "$0.runfiles/" ]; then
    MY_LOCATION="$0.runfiles/bazel_tools/tools/objc"
  else
    MY_LOCATION="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  fi
fi

function invoke_libtool() {
  # Just invoke libtool via xcrunwrapper
  "${MY_LOCATION}/xcrunwrapper.sh" libtool "$@" \
  2> >(grep -v "the table of contents is empty (no object file members in the"`
              `" library define global symbols)$" >&2)
  # ^ Filtering a warning that's unlikely to indicate a real issue
  # ...and not silencable via a flag.
}

if [ ! -f "${MY_LOCATION}"/libtool_check_unique ] ; then
  echo "libtool_check_unique not found. Please file an issue at github.com/bazelbuild/bazel"
  exit 1
elif "${MY_LOCATION}"/libtool_check_unique "$@"; then
  # If there are no duplicate .o basenames,
  # libtool can be invoked with the original arguments.
  invoke_libtool "$@"
  exit
fi

TEMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/libtool.XXXXXXXX")"
trap 'rm -rf "$TEMPDIR"' EXIT

# Creates a symbolic link to the input argument file and returns the symlink
# file path.
function hash_objfile() {
  ORIGINAL_NAME="$1"
  ORIGINAL_HASH="$(/sbin/md5 -qs "${ORIGINAL_NAME}")"
  SYMLINK_NAME="${TEMPDIR}/$(basename "${ORIGINAL_NAME%.o}_${ORIGINAL_HASH}.o")"
  if [[ ! -e "$SYMLINK_NAME" ]]; then
    case "${ORIGINAL_NAME}" in
      /*) ln -sf "$ORIGINAL_NAME" "$SYMLINK_NAME" ;;
      *) ln -sf "$(pwd)/$ORIGINAL_NAME" "$SYMLINK_NAME" ;;
    esac
  fi
  echo "$SYMLINK_NAME"
}

python_executable=/usr/bin/python3
if [[ ! -x "$python_executable" ]]; then
  python_executable=python3
fi

ARGS=()
handle_filelist=0
keep_next=0

function parse_option() {
  local -r ARG="$1"
  if [[ "$handle_filelist" == "1" ]]; then
    handle_filelist=0
    HASHED_FILELIST="${ARG%.objlist}_hashes.objlist"
    rm -f "${HASHED_FILELIST}"
    # Use python helper script for fast md5 calculation of many strings.
    "$python_executable" "${MY_LOCATION}/make_hashed_objlist.py" \
      "${ARG}" "${HASHED_FILELIST}" "${TEMPDIR}"
    ARGS+=("${HASHED_FILELIST}")
  elif [[ "$keep_next" == "1" ]]; then
    keep_next=0
    ARGS+=("$ARG")
  else
    case "${ARG}" in
      # Filelist flag, need to symlink each input in the contents of file and
      # pass a new filelist which contains the symlinks.
      -filelist)
        handle_filelist=1
        ARGS+=("${ARG}")
        ;;
      @*)
        path="${ARG:1}"
        while IFS= read -r opt
        do
          parse_option "$opt"
        done < "$path" || exit 1
        ;;
      # Flags with no args
      -static|-s|-a|-c|-L|-T|-D|-v|-no_warning_for_no_symbols)
        ARGS+=("${ARG}")
        ;;
      # Single-arg flags
      -arch_only|-syslibroot|-o)
        keep_next=1
        ARGS+=("${ARG}")
        ;;
      # Any remaining flags are unexpected and may ruin flag parsing.
      # Add any flags here to libtool_check_unique.cc as well
      -*)
        echo "Unrecognized libtool flag ${ARG}"
        exit 1
        ;;
      # Archive inputs can remain untouched, as they come from other targets.
      *.a)
        ARGS+=("${ARG}")
        ;;
      # Remaining args are input objects
      *)
        ARGS+=("$(hash_objfile "${ARG}")")
        ;;
    esac
  fi
}

for arg in "$@"; do
  parse_option "$arg"
done

printf '%s\n' "${ARGS[@]}" > "$TEMPDIR/processed.params"
invoke_libtool "@$TEMPDIR/processed.params"
