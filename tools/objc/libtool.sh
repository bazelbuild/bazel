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

WRAPPER="${MY_LOCATION}/xcrunwrapper.sh"

# Ensure 0 timestamping for hermetic results.
export ZERO_AR_DATE=1

if [ ! -f "${MY_LOCATION}"/libtool_check_unique ] ; then
    echo "libtool_check_unique not found. Please file an issue at github.com/bazelbuild/bazel"
elif "${MY_LOCATION}"/libtool_check_unique "$@"; then
  # If there are no duplicate .o basenames,
  # libtool can be invoked with the original arguments.
  "${WRAPPER}" libtool "$@"
  exit
fi

TEMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/libtool.XXXXXXXX")"
trap "rm -rf \"${TEMPDIR}\"" EXIT

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

python_executable=/usr/bin/python2.7
if [[ ! -x "$python_executable" ]]; then
  python_executable=python
fi

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
      # Use python helper script for fast md5 calculation of many strings.
      "$python_executable" "${MY_LOCATION}/make_hashed_objlist.py" \
        "${ARG}" "${HASHED_FILELIST}" "${TEMPDIR}"
      ARGS+=("${HASHED_FILELIST}")
      ;;
   # Output flag
    -o)
     ARGS+=("${ARG}")
     ARG="$1"
     shift
     ARGS+=("${ARG}")
     OUTPUTFILE="${ARG}"
     ;;
   # Flags with no args
    -static|-s|-a|-c|-L|-T|-D|-no_warning_for_no_symbols)
      ARGS+=("${ARG}")
      ;;
   # Single-arg flags
   -arch_only|-syslibroot)
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
   # Archive inputs can remain untouched, as they come from other targets.
   *.a)
     ARGS+=("${ARG}")
     ;;
   # Remaining args are input objects
   *)
     ARGS+=("$(echo "$(hash_objfile "${ARG}")")")
     ;;
   esac
done

"${WRAPPER}" libtool "${ARGS[@]}"
