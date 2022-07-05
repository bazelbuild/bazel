#!/bin/bash
# Copyright 2022 The Bazel Authors. All rights reserved.
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
# Wrapper script to invoke D8 class file to class file desugaring.

# exit on errors and uninitialized variables
set -eu

readonly TMPDIR="$(mktemp -d)"
trap "rm -rf ${TMPDIR}" EXIT

readonly DESUGAR_CHM_ONLY_CONFIG=(--desugared_lib_config $0.runfiles/bazel_tools/tools/android/chm_only_desugar_jdk_libs.json)

# Check for params file.  Desugar doesn't accept a mix of params files and flags
# directly on the command line, so we need to build a new params file that adds
# the flags we want.
if [[ "$#" -gt 0 ]]; then
  arg="$1";
  case "${arg}" in
    @*)
      params="${TMPDIR}/desugar.params"
      cat "${arg:1}" > "${params}"  # cp would create file readonly
      for o in "${DESUGAR_CHM_ONLY_CONFIG[@]}"; do
        echo "${o}" >> "${params}"
      done
      "$0.runfiles/bazel_tools/src/tools/android/java/com/google/devtools/build/android/r8/desugar" \
          "@${params}"
      # temp dir deleted by TRAP installed above
      exit 0
    ;;
  esac
fi

# Some unit tests pass an explicit --desugared_lib_config, in that case don't
# add the default one.
has_desugared_lib_config=false
for arg in "$@"; do
  if [[ "$arg" == "--desugared_lib_config" ]]; then
    has_desugared_lib_config=true
  fi
done

if [[ "$has_desugared_lib_config" == "true" ]]; then
  "$0.runfiles/bazel_tools/src/tools/android/java/com/google/devtools/build/android/r8/desugar" \
      "$@"
else
  "$0.runfiles/bazel_tools/src/tools/android/java/com/google/devtools/build/android/r8/desugar" \
      "$@" \
      "${DESUGAR_CHM_ONLY_CONFIG[@]}"
fi
