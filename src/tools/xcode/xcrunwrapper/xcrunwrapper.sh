#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All rights reserved.
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
# xcrunwrapper runs the command passed to it using xcrun. The first arg
# passed is the name of the tool to be invoked via xcrun. (For example, libtool
# or clang).
# xcrunwrapper replaces __BAZEL_XCODE_DEVELOPER_DIR__ with $DEVELOPER_DIR (or
# reasonable default) and __BAZEL_XCODE_SDKROOT__ with a valid path based on
# SDKROOT (or reasonable default).
# These values (__BAZEL_XCODE_*) are a shared secret withIosSdkCommands.java.

set -eu

TOOLNAME=$1
shift

# Creates a symbolic link to the input argument file and returns the symlink
# file path.
function hash_objfile() {
  ORIGINAL_NAME="$1"
  ORIGINAL_HASH="$(/sbin/md5 -q "${ORIGINAL_NAME}")"
  SYMLINK_NAME="${ORIGINAL_NAME%.o}_${ORIGINAL_HASH}.o"
  ln -sf "$(basename "$ORIGINAL_NAME")" "$SYMLINK_NAME"
  echo "$SYMLINK_NAME"
}

# Pick values for DEVELOPER_DIR and SDKROOT as appropriate (if they weren't set)

WRAPPER_DEVDIR="${DEVELOPER_DIR:-}"
if [[ -z "${WRAPPER_DEVDIR}" ]] ; then
  WRAPPER_DEVDIR="$(xcode-select -p)"
fi

# TODO(blaze-team): Remove this once all build environments are setting SDKROOT
# for us.
WRAPPER_SDKROOT="${SDKROOT:-}"
if [[ -z "${WRAPPER_SDKROOT:-}" ]] ; then
  WRAPPER_SDK=iphonesimulator
  for ARG in "$@" ; do
    case "${ARG}" in
      armv6|armv7|armv7s|arm64)
        WRAPPER_SDK=iphoneos
        ;;
      i386|x86_64)
        WRAPPER_SDK=iphonesimulator
        ;;
    esac
  done
  WRAPPER_SDKROOT="$(/usr/bin/xcrun --show-sdk-path --sdk ${WRAPPER_SDK})"
fi

ARGS=("$TOOLNAME")
while [[ $# -gt 0 ]]; do
  ARG="$1"
  shift

  # Libtool artifact symlinking. Apple's libtool has a bug when there are two
  # input files with the same basename. We thus create symlinks that are named
  # with a hash suffix for each input, and pass them to libtool.
  # See b/28186497.
  # TODO(b/28347228): Handle this in a separate wrapper.
  if [ "$TOOLNAME" = "libtool" ] ; then
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
  else
    ARGS+=("${ARG}")
  fi
done

# Subsitute toolkit path placeholders.
UPDATEDARGS=()
for ARG in "${ARGS[@]}" ; do
  ARG="${ARG//__BAZEL_XCODE_DEVELOPER_DIR__/${WRAPPER_DEVDIR}}"
  ARG="${ARG//__BAZEL_XCODE_SDKROOT__/${WRAPPER_SDKROOT}}"
  UPDATEDARGS+=("${ARG}")
done

/usr/bin/xcrun "${UPDATEDARGS[@]}"
