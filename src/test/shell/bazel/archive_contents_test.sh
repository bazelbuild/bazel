#!/bin/bash
#
# Copyright 2019 The Bazel Authors. All rights reserved.
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
# Verify that archives can be unpacked, even if they contain strangely named
# files.

# --- begin runfiles.bash initialization ---
set -euo pipefail
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

case "$(uname -s | tr [:upper:] [:lower:])" in
msys*|mingw*|cygwin*)
  declare -r is_windows=true
  ;;
*)
  declare -r is_windows=false
  ;;
esac

#### SETUP #############################################################

test_tar_utf8() {
    # Archive contains a file that is valid UTF8, but is not 7-bit clean.
    WRKDIR=`pwd`
    mkdir ext
    echo 'Hello World' > ext/data.txt
    touch ext/$'unrelated\xF0\x9F\x8D\x83.txt'
    touch ext/$'cyrillic\xD0\x90\xD0\x91\xD0\x92\xD0\x93\xD0\x94...'
    touch ext/$'umlauts\x41\xCC\x88\x4F\xCC\x88\x55\xCC\x88\x61\xCC\x88\x6F\xCC\x88\x75\xCC\x88'
    # TODO(philwo) - figure out why we get an "invalid characters" error on
    # macOS Catalina when using this form (NFC normalized).
    #touch ext/$'umlauts\xC3\x84\xC3\x96\xC3\x9C\xC3\xA4\xC3\xB6\xC3\xBC'

    tar cf ext.tar ext
    rm -rf ext

    mkdir main
    cd main
    if $is_windows; then
        # Windows needs "file:///c:/foo/bar".
        FILE_URL="file:///$(cygpath -m "$WRKDIR")/ext.tar"
    else
        # Non-Windows needs "file:///foo/bar".
        FILE_URL="file://${WRKDIR}/ext.tar"
    fi

    cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
  name = "ext",
  url = "${FILE_URL}",
  build_file_content = "exports_files(['data.txt'])",
  strip_prefix="ext",
)
EOF
    cat > BUILD <<'EOF'
genrule(
  name = "it",
  outs = ["it.txt"],
  srcs = ["@ext//:data.txt"],
  cmd = "cp $< $@",
)
EOF

    bazel build //:it || fail "Build should succeed"
}


run_suite "Tests for extracting archives containing strangely-named files"
