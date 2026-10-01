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
template="$(rlocation io_bazel/scripts/packages/template_bin.sh)"

function set_up() {
  test_dir="$(mktemp -d "${TEST_TMPDIR}/installer.XXXXXXXX")"
  mkdir -p "${test_dir}/payload" "${test_dir}/cwd"
  for file in bazel bazel-real bazel-complete.bash _bazel bazel.fish; do
    printf '#!/usr/bin/env bash\n# %s\nexit 0\n' "$file" > "${test_dir}/payload/${file}"
  done
  (
    cd "${test_dir}/payload"
    zip -q "${test_dir}/payload.zip" ./*
  )
  # Match self_extract_binary.bzl: prepend the launcher and adjust ZIP offsets.
  cat "$template" "${test_dir}/payload.zip" > "${test_dir}/installer.sh"
  zip -qA "${test_dir}/installer.sh"
  cd "${test_dir}/cwd"
}

function check_install() {
  local bin="$1" base="$2"
  shift 2
  # Skip executing a real Bazel binary; extraction and filesystem operations are real.
  bash "${test_dir}/installer.sh" --skip-platform-check --skip-uncompress "$@" \
    > "$TEST_log" 2>&1 || fail "Installation failed"
  [[ -d "${base}/etc" ]] || fail "Missing base/etc directory"
  [[ -L "${bin}/bazel" ]] || fail "Missing bazel symlink"
  assert_equals "${base}/bin/bazel" "$(readlink "${bin}/bazel")"
  [[ -x "${bin}/bazel" ]] || fail "Installed bazel is not executable"
  [[ -x "${base}/bin/bazel-real" ]] || fail "Installed bazel-real is not executable"
  local file
  for file in bazel bazel-real bazel-complete.bash _bazel bazel.fish; do
    cmp "${test_dir}/payload/${file}" "${base}/bin/${file}" \
      || fail "Incorrect installed contents: ${file}"
  done
  assert_equals "" "$(ls -A "${test_dir}/cwd")" "Unexpected directories in working directory"
}

function test_install_plain_prefix() {
  local prefix="${test_dir}/install"
  check_install "${prefix}/bin" "${prefix}/lib/bazel" "--prefix=${prefix}"
}

function test_install_prefix_with_spaces() {
  local prefix="${test_dir}/install prefix"
  check_install "${prefix}/bin" "${prefix}/lib/bazel" "--prefix=${prefix}"
}

function test_install_bin_with_spaces() {
  local bin="${test_dir}/custom bin" base="${test_dir}/base"
  check_install "$bin" "$base" "--bin=${bin}" "--base=${base}"
}

function test_install_base_with_spaces() {
  local bin="${test_dir}/bin" base="${test_dir}/custom base"
  check_install "$bin" "$base" "--bin=${bin}" "--base=${base}"
}

function test_install_user_with_spaces() {
  export HOME="${test_dir}/user home"
  check_install "${HOME}/bin" "${HOME}/.bazel" --user
}

run_suite "Self-extracting installer path tests"
