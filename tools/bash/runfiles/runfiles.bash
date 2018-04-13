#!/bin/bash
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

# This Bash script defines functions to handle sh_binary/sh_test runfiles.
#
# REQUIREMENTS:
# - The RUNFILES_MANIFEST_FILE and/or the RUNFILES_DIR environment variable must
#   be set to the absolute path of the runfiles manifest or the
#   <rulename>.runfiles directory, respectively.
# - If RUNFILES_MANIFEST_ONLY=1 is set, then RUNFILES_MANIFEST_FILE must be set
#   to the absolute path of the runfiles manifest. RUNFILES_DIR may be unset in
#   this case.
# - If RUNFILES_LIB_DEBUG=1 is set, the script will print errors to stderr.

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  # matches an absolute Windows path
  _rlocation_isabs_pattern="^[a-zA-Z]:[/\\]"
  ;;
*)
  # matches an absolute Unix path
  _rlocation_isabs_pattern="^/.*"
  ;;
esac

# Prints to stdout the runtime location of a data-dependency.
function rlocation() {
  if [[ "$1" =~ $_rlocation_isabs_pattern ]]; then
    # If the path is absolute, print it as-is.
    echo $1
  elif [[ "$1" =~ \.\. ]]; then
    if [[ "${RUNFILES_LIB_DEBUG:-}" == 1 ]]; then
      echo >&2 "ERROR[runfiles.bash]: rlocation($1): contains uplevel reference"
    fi
    return 1
  elif [[ "$1" == \\* ]]; then
    if [[ "${RUNFILES_LIB_DEBUG:-}" == 1 ]]; then
      echo >&2 "ERROR[runfiles.bash]: rlocation($1): absolute path without" \
               "drive name"
    fi
    return 1
  else
    if [[ "${RUNFILES_MANIFEST_ONLY:-}" == 1 ]]; then
      if [[ -n "${RUNFILES_MANIFEST_FILE:-}" \
            && -f "${RUNFILES_MANIFEST_FILE}" ]]; then
        grep -m1 "^$1 " "${RUNFILES_MANIFEST_FILE}" | cut -d ' ' -f 2- \
          || return 1
      else
        if [[ "${RUNFILES_LIB_DEBUG:-}" == 1 ]]; then
          echo >&2 "ERROR[runfiles.bash]: trying to use manifest-based" \
                   "runfiles but RUNFILES_MANIFEST_FILE is unset or" \
                   "non-existent" \
                   "(RUNFILES_MANIFEST_ONLY=\"${RUNFILES_MANIFEST_ONLY:-}\"," \
                   "RUNFILES_DIR=\"${RUNFILES_DIR:-}\")"
        fi
        return 1
      fi
    elif [[ -n "${RUNFILES_DIR:-}" && -d "${RUNFILES_DIR}" ]]; then
      echo "${RUNFILES_DIR}/$1"
    else
      if [[ "${RUNFILES_LIB_DEBUG:-}" == 1 ]]; then
        echo >&2 "ERROR[runfiles.bash]: trying to use directory-based" \
                 "runfiles but RUNFILES_DIR is unset or non-existent" \
                 "(RUNFILES_MANIFEST_ONLY=\"${RUNFILES_MANIFEST_ONLY:-}\"," \
                 "RUNFILES_MANIFEST_FILE=\"${RUNFILES_MANIFEST_FILE:-}\")"
      fi
      return 1
    fi
  fi
}
export -f rlocation

# Exports the environment variables that subprocesses may need to use runfiles.
# If a subprocess is a Bazel-built binary rule that also uses the runfiles
# libraries under @bazel_tools//tools/bash/runfiles, then that binary needs
# these envvars in order to initialize its own runfiles library.
function runfiles_export_envvars() {
  if [[ "${RUNFILES_MANIFEST_ONLY:-}" == 1 ]]; then
    if [[ -z "${RUNFILES_MANIFEST_FILE:-}" \
          && ! -f "$RUNFILES_MANIFEST_FILE" ]]; then
      return 1
    fi
    if [[ -z "${RUNFILES_DIR:-}" ]]; then
      if [[ "$RUNFILES_MANIFEST_FILE" == */MANIFEST \
            && -d "${RUNFILES_MANIFEST_FILE%/MANIFEST}" ]]; then
        export RUNFILES_DIR="${RUNFILES_MANIFEST_FILE%/MANIFEST}"
      elif [[ "$RUNFILES_MANIFEST_FILE" == *runfiles_manifest \
            && -d "${RUNFILES_MANIFEST_FILE%_manifest}" ]]; then
        export RUNFILES_DIR="${RUNFILES_MANIFEST_FILE%_manifest}"
      fi
    fi
  fi
  # No need to define anything if RUNFILES_MANIFEST_ONLY is not 1: it makes no
  # difference whether RUNFILES_DIR is defined or not.

  export RUNFILES_MANIFEST_FILE="${RUNFILES_MANIFEST_FILE:-}"
  export RUNFILES_DIR="${RUNFILES_DIR:-}"
  export JAVA_RUNFILES="${RUNFILES_DIR:-}"
}
export -f runfiles_export_envvars
