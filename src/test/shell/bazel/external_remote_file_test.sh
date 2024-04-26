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
#
# Tests the patching functionality of external repositories.

set -euo pipefail
# --- begin runfiles.bash initialization ---
if [[ ! -d "${RUNFILES_DIR:-/dev/null}" && ! -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  if [[ -f "$0.runfiles_manifest" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles_manifest"
  elif [[ -f "$0.runfiles/MANIFEST" ]]; then
    export RUNFILES_MANIFEST_FILE="$0.runfiles/MANIFEST"
  elif [[ -f "$0.runfiles/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
    export RUNFILES_DIR="$0.runfiles"
  fi
fi
if [[ -f "${RUNFILES_DIR:-/dev/null}/bazel_tools/tools/bash/runfiles/runfiles.bash" ]]; then
  source "${RUNFILES_DIR}/bazel_tools/tools/bash/runfiles/runfiles.bash"
elif [[ -f "${RUNFILES_MANIFEST_FILE:-/dev/null}" ]]; then
  source "$(grep -m1 "^bazel_tools/tools/bash/runfiles/runfiles.bash " \
            "$RUNFILES_MANIFEST_FILE" | cut -d ' ' -f 2-)"
else
  echo >&2 "ERROR: cannot find @bazel_tools//tools/bash/runfiles:runfiles.bash"
  exit 1
fi
# --- end runfiles.bash initialization ---

source "$(rlocation "io_bazel/src/test/shell/integration_test_setup.sh")" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# `uname` returns the current platform, e.g "MSYS_NT-10.0" or "Linux".
# `tr` converts all upper case letters to lower case.
# `case` matches the result if the `uname | tr` expression to string prefixes
# that use the same wildcards as names do in Bash, i.e. "msys*" matches strings
# starting with "msys", and "*" matches everything (it's the default case).
case "$(uname -s | tr [:upper:] [:lower:])" in
msys*)
  # As of 2019-01-15, Bazel on Windows only supports MSYS Bash.
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

if "$is_windows"; then
  # Disable MSYS path conversion that converts path-looking command arguments to
  # Windows paths (even if they arguments are not in fact paths).
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi


if $is_windows; then
  export MSYS_NO_PATHCONV=1
  export MSYS2_ARG_CONV_EXCL="*"
fi

set_up() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  write_default_lockfile "MODULE.bazel.lock"
  # create an archive file with files interesting for patching
  mkdir hello_world-0.1.2
  cat > hello_world-0.1.2/hello_world.c <<'EOF'
#include <stdio.h>
int main() {
  printf("Hello, world!\n");
  return 0;
}
EOF
  zip hello_world.zip hello_world-0.1.2/*
  rm -rf hello_world-0.1.2
}

function get_extrepourl() {
  if $is_windows; then
    echo "file:///$(cygpath -m $1)"
  else
    echo "file://$1"
  fi
}


test_overlay_remote_file_with_integrity() {
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"

  # Generate the remote files to overlay
  # this could be like the Bazel Central Repository
  # Generate the remote patch file
  cat > BUILD.bazel <<'EOF'
load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "hello_world",
    srcs = ["hello_world.c"],
)
EOF
  touch WORKSPACE

  # Test that the patches attribute of http_archive is honored
  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="hello_world",
  strip_prefix="hello_world-0.1.2",
  urls=["${EXTREPOURL}/hello_world.zip"],
  remote_file_urls={
    "WORKSPACE": ["${EXTREPOURL}/WORKSPACE"],
    "BUILD.bazel": ["${EXTREPOURL}/BUILD.bazel"],
  },
  remote_file_integrity={
    "WORKSPACE": "sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=",
    "BUILD.bazel": "sha256-0bs+dwSOzHTbNAgDS02I3giLAZu2/NLn7BJWwQGN/Pk=",
  },
)
EOF
  write_default_lockfile "MODULE.bazel.lock"

  # TODO(fzakaria): I found this command to fail since it tries to spawn sandbox
  # within a sandbox. I'm not sure how the other tests that build cc_binary get
  # around this.
  bazel build @hello_world//:hello_world --spawn_strategy=local
}

run_suite "external remote file tests"
