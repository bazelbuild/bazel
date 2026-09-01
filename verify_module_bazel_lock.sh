#!/usr/bin/env bash

# Copyright 2026 The Bazel Authors. All rights reserved.
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

set -euo pipefail
# --- begin runfiles.bash initialization v3 ---
# Copy-pasted from the Bazel Bash runfiles library v3.
set -uo pipefail; set +e; f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f2- -d' ')" 2>/dev/null || \
source "$0.runfiles/$f" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
source "$(grep -sm1 "^$f " "$0.exe.runfiles_manifest" | cut -f2- -d' ')" 2>/dev/null || \
{ echo>&2 "ERROR: cannot find $f"; exit 1; }; f=; set -e
# --- end runfiles.bash initialization v3 ---

function die() {
  echo "$@" >&2
  exit 1
}

# Verifies that Bazel's own MODULE.bazel.lock is up-to-date, i.e., that running
# `bazel mod deps --lockfile_mode=update` in the workspace would not modify it.
#
# `--lockfile_mode=error` on CI only reports lockfile information that is
# missing or outdated, but tolerates extra information such as entries for
# registry files, yanked versions or module extensions that are no longer
# referenced after a change to MODULE.bazel (this tolerance is intentional,
# e.g. to keep automated git merges of lockfiles working). Such leftovers are
# only cleaned up by an update, which makes local builds dirty the tree. This
# test replays module resolution against the checked-in lockfile and fails if
# an update would modify the file.

# The lockfile is maintained by the Bazel version in .bazelversion, which
# developers and CI use via bazelisk and which resolves modules slightly
# differently than a Bazel built at HEAD, e.g. due to the bazel_dep versions in
# its embedded MODULE.tools. Download and run exactly that version.
bazel_version="$(tr -d '[:space:]' < "$(rlocation io_bazel/.bazelversion)")"
case "$(uname -s)" in
  Linux) os=linux ;;
  Darwin) os=darwin ;;
  MSYS*|MINGW*|CYGWIN*) os=windows ;;
  *) die "Unsupported operating system: $(uname -s)" ;;
esac
case "$(uname -m)" in
  x86_64|amd64) arch=x86_64 ;;
  arm64|aarch64) arch=arm64 ;;
  *) die "Unsupported architecture: $(uname -m)" ;;
esac
suffix=""
if [[ "$os" == "windows" ]]; then
  suffix=".exe"
fi
url="https://releases.bazel.build/${bazel_version}/release/bazel-${bazel_version}-${os}-${arch}${suffix}"
bazel="$TEST_TMPDIR/bazel${suffix}"
echo "Downloading $url"
curl -fsSL -o "$bazel" "$url" || die "Failed to download Bazel $bazel_version from $url."
chmod +x "$bazel"

# Set up a workspace that contains everything that module resolution (but not
# module extension evaluation, which requires fetching repos) needs: the root
# module file, the current lockfile, and the files referenced by the overrides
# in MODULE.bazel. This mirrors the workspace set up by the
# //:generate_dist_lockfile genrule.
checked_in_lockfile="$(rlocation io_bazel/MODULE.bazel.lock)"
mkdir -p "$TEST_TMPDIR/workspace"
cd "$TEST_TMPDIR/workspace"
touch BUILD.bazel
cp "$(rlocation io_bazel/MODULE.bazel)" MODULE.bazel
cp "$checked_in_lockfile" MODULE.bazel.lock
chmod u+w MODULE.bazel MODULE.bazel.lock

# Patches referenced by overrides in MODULE.bazel are read during module
# resolution and thus have to exist in the workspace.
for label in $(grep -o '"//[^"]*\.patch"' MODULE.bazel | tr -d '"' | sort -u); do
  patch_path="${label#//}"
  patch_path="${patch_path/://}"
  mkdir -p "$(dirname "$patch_path")"
  cp "$(rlocation "io_bazel/$patch_path")" "$patch_path" \
    || die "Patch file $label referenced by an override in MODULE.bazel is not available to this test. Please add it to the data of //:verify_module_bazel_lock in BUILD."
  # The package containing a patch has to exist for its label to resolve, but
  # its BUILD file is never loaded.
  touch "$(dirname "$patch_path")/BUILD"
done

# MODULE.bazel files of modules with a local_path_override are read during
# module resolution. If this list is incomplete, the bazel invocation below
# fails with a clear error message.
mkdir -p third_party/remoteapis
cp "$(rlocation io_bazel/third_party/remoteapis/MODULE.bazel)" \
  third_party/remoteapis/MODULE.bazel

# `bazel query :all` triggers module resolution, but no extension evaluation,
# and in update mode rewrites MODULE.bazel.lock if and only if its contents
# are no longer up-to-date.
echo "Running: bazel query --lockfile_mode=update to verify the lockfile."
"$bazel" --batch --ignore_all_rc_files \
    --output_user_root="$TEST_TMPDIR/output_user_root" \
    query --check_direct_dependencies=error --lockfile_mode=update :all \
  || die "Module resolution failed. If the error above mentions a file that does not exist in this test's workspace, please add it to the data of //:verify_module_bazel_lock in BUILD and copy it into place above."

diff -u "$checked_in_lockfile" MODULE.bazel.lock \
  || die "MODULE.bazel.lock is not up-to-date (see the diff above; '-' lines are checked in, '+' lines are what Bazel would generate). Please run \"bazel mod deps --lockfile_mode=update\" in your workspace and commit the resulting changes. This typically happens when MODULE.bazel is changed without updating the lockfile."

echo "PASS"
