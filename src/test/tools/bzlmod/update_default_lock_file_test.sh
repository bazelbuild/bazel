#!/usr/bin/env bash
#
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

source "$(rlocation io_bazel/src/test/shell/unittest.bash)" || exit 1
updater="$(rlocation io_bazel/src/test/tools/bzlmod/update_default_lock_file.sh)"
runfiles_library="$(rlocation bazel_tools/tools/bash/runfiles/runfiles.bash)"

function check_update() {
  local root="$(mktemp -d "${TEST_TMPDIR}/lock-updater.XXXXXXXX")"
  local runfiles="${root}/$1" workspace="${root}/$2" tmp="${root}/$3"
  local expected_status="${4:-0}" status=0
  local destination="${workspace}/src/test/tools/bzlmod/MODULE.bazel.lock"
  mkdir -p "${runfiles}/bazel_tools/tools/bash/runfiles" \
    "${runfiles}/io_bazel/src" "$(dirname "$destination")" "$tmp"
  cp "$runfiles_library" "${runfiles}/bazel_tools/tools/bash/runfiles/runfiles.bash"
  printf 'original lockfile\n' > "$destination"
  printf '{"lockFileVersion": 10}\n' > "${root}/generated.lock"
  # Use the real runfiles library, with a small Bazel stand-in to avoid registry access.
  cat > "${runfiles}/io_bazel/src/bazel" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\0' "$@" > "${FIXTURE_DIR}/args"
printf '%s\n' "$PWD" > "${FIXTURE_DIR}/generated.workspace"
[[ -f MODULE.bazel && -f BUILD ]]
if [[ "$FIXTURE_STATUS" != 0 ]]; then
  exit "$FIXTURE_STATUS"
fi
cp "${FIXTURE_DIR}/generated.lock" MODULE.bazel.lock
EOF
  chmod +x "${runfiles}/io_bazel/src/bazel"
  env -u RUNFILES_MANIFEST_FILE -u RUNFILES_MANIFEST_ONLY \
    -u RUNFILES_REPO_MAPPING RUNFILES_DIR="$runfiles" \
    BUILD_WORKSPACE_DIRECTORY="$workspace" TMPDIR="$tmp" \
    FIXTURE_DIR="$root" FIXTURE_STATUS="$expected_status" \
    bash "$updater" '--registry=file:///registry with spaces' '' '*.literal' \
    > "$TEST_log" 2>&1 || status=$?
  assert_equals "$expected_status" "$status" "Unexpected updater exit status"
  printf '%s\0' --batch --ignore_all_rc_files mod deps \
    '--registry=file:///registry with spaces' '' '*.literal' > "${root}/expected.args"
  cmp "${root}/expected.args" "${root}/args" || fail "Arguments changed"
  if [[ "$expected_status" == 0 ]]; then
    cmp "${root}/generated.lock" "$destination" || fail "Lockfile not updated"
  else
    assert_equals 'original lockfile' "$(cat "$destination")"
  fi
  local generated_workspace="$(cat "${root}/generated.workspace")"
  [[ ! -d "$generated_workspace" ]] \
    || fail "Temporary workspace was not removed: ${generated_workspace}"
}

function test_plain_paths() {
  check_update runfiles workspace tmp
}

function test_runfiles_path_with_spaces() {
  check_update 'runfiles with spaces' workspace tmp
}

function test_workspace_path_with_spaces() {
  check_update runfiles 'workspace with spaces' tmp
}

function test_temporary_path_with_spaces() {
  # Some platforms' mktemp -t ignores TMPDIR. Check the actual generated path.
  check_update runfiles workspace 'tmp with spaces'
}

function test_all_paths_with_spaces() {
  check_update 'runfiles with spaces' 'workspace with spaces' 'tmp with spaces'
}

function test_bazel_failure_preserves_lockfile() {
  check_update runfiles workspace tmp 42
}

run_suite "Default lockfile updater path tests"
