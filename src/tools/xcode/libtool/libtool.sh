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
# It creates symbolic links for all input files with a content-hash appended
# to their original name (foo.o becomes foo_{md5sum}.o). This is to circumvent
# a bug in the original tool that arises when two input files have the same
# base name (even if they are in different directories).

set -eu

MY_LOCATION=${MY_LOCATION:-"$0.runfiles/bazel_tools/tools/objc"}
WRAPPER="${MY_LOCATION}/xcrunwrapper.sh"

# Creates a symbolic link to the input argument file and returns the symlink
# file path.
function hash_objfile() {
  ORIGINAL_NAME="$1"
  ORIGINAL_HASH="$(/sbin/md5 -q "${ORIGINAL_NAME}")"
  SYMLINK_NAME="${ORIGINAL_NAME%.o}_${ORIGINAL_HASH}.o"
  ln -sf "$(basename "$ORIGINAL_NAME")" "$SYMLINK_NAME"
  echo "$SYMLINK_NAME"
}

ARGS=()

while [[ $# -gt 0 ]]; do
  ARG="$1"
  shift

  # Libtool artifact symlinking. Apple's libtool has a bug when there are two
  # input files with the same basename. We thus create symlinks that are named
  # with a hash suffix for each input, and pass them to libtool.
  # See b/28186497.
  case "${ARG}" in
    # Filelist flag, need to symlink each input in the contents of file and
    # pass a new filelist which contains the symlinks.
    -filelist)
      ARGS+=("${ARG}")
      ARG="$1"
      shift
      HASHED_FILELIST="${ARG%.objlist}_hashes.objlist"
      rm -f "${HASHED_FILELIST}"
      while read INPUT_FILE || [ -n "$INPUT_FILE" ]; do
        echo "$(hash_objfile "${INPUT_FILE}")" >> "$HASHED_FILELIST"
      done < "${ARG}"
      ARGS+=("${HASHED_FILELIST}")
      ;;
   # Flags with no args
    -static|-s|-a|-c|-L|-T|-no_warning_for_no_symbols)
      ARGS+=("${ARG}")
      ;;
   # Single-arg flags
   -o|-arch_only|-syslibroot)
     ARGS+=("${ARG}")
     ARG="$1"
     shift
     ARGS+=("${ARG}")
     ;;
   # Any remaining flags are unexpected and may ruin flag parsing.
   -*)
     echo "Unrecognized libtool flag ${ARG}"
     exit 1
     ;;
   # Remaining args are input objects
   *)
     ARGS+=("$(echo "$(hash_objfile "${ARG}")")")
     ;;
   esac
done

"${WRAPPER}" libtool "${ARGS[@]}"
