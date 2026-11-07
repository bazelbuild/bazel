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
build_script="$(rlocation io_bazel/scripts/ci/build.sh)"

function set_up() {
  local test_dir tool
  test_dir="$(mktemp -d "${TEST_TMPDIR}/release_index.XXXXXXXX")"
  cd "${test_dir}"
  mkdir bin artifacts
  export PATH="${test_dir}/bin:${PATH}"
  # Sourcing build.sh checks these tools and reads a gcloud secret.
  for tool in gcloud gh debsign reprepro gpg; do
    printf '#!/usr/bin/env bash\nexit 0\n' > "bin/${tool}"
    chmod +x "bin/${tool}"
  done
  cat > bin/pandoc <<'EOF'
#!/usr/bin/env bash
[[ "$*" == "-f markdown -t html" ]] || exit 99
cat
exit "${PANDOC_STATUS:-0}"
EOF
  chmod +x bin/pandoc
  export PANDOC_STATUS=0
  echo 'get_release_page() { echo "Release notes"; }' > setup.sh
  echo checksum > artifacts/bazel.tar.gz.sha256
}

function run_index() {
  local status=0
  # Start without pipefail so the test runner cannot mask the original bug.
  bash -c 'set +o pipefail; source "$1"; source "$2"; eval "$3"' \
    _ "${build_script}" "${PWD}/setup.sh" "$1" > "$TEST_log" 2>&1 || status=$?
  assert_equals "$2" "${status}"
}

function fail_producer() {
  echo 'create_index_md() { echo partial; return 17; }' > setup.sh
}

function fail_release_page() {
  echo 'get_release_page() { echo partial; return 17; }' > setup.sh
}

function assert_index_output() {
  cat > expected <<'EOF'
Release notes

## Index of files

 - [bazel.tar.gz](bazel.tar.gz) [[SHA-256](bazel.tar.gz.sha256)] [[SIG](bazel.tar.gz.sig)]
EOF
  diff -u expected "$TEST_log" || fail "Unexpected index contents"
}

function test_markdown_failure() {
  fail_producer
  run_index 'create_index_html artifacts' 17
}

function test_markdown_failure_in_conditional() {
  fail_producer
  run_index 'if create_index_html artifacts; then exit 0; else exit $?; fi' 17
}

function test_release_page_failure() {
  fail_release_page
  run_index 'create_index_html artifacts' 17
  expect_not_log 'Index of files'
}

function test_release_page_failure_in_conditional() {
  fail_release_page
  run_index 'if create_index_html artifacts; then exit 0; else exit $?; fi' 17
  expect_not_log 'Index of files'
}

function test_pandoc_failure() {
  export PANDOC_STATUS=29
  run_index 'create_index_html artifacts' 29
}

function test_pandoc_failure_in_conditional() {
  export PANDOC_STATUS=29
  run_index 'if create_index_html artifacts; then exit 0; else exit $?; fi' 29
}

function test_both_commands_fail() {
  fail_producer
  export PANDOC_STATUS=29
  run_index 'if create_index_html artifacts; then exit 0; else exit $?; fi' 29
}

function test_success() {
  run_index 'create_index_html artifacts' 0
  assert_index_output
}

function test_success_in_conditional() {
  run_index 'if create_index_html artifacts; then exit 0; else exit $?; fi' 0
  assert_index_output
}

function test_pipefail_stays_disabled_after_success() {
  run_index 'create_index_html artifacts; [[ ! -o pipefail ]]' 0
}

function test_pipefail_stays_disabled_after_failure() {
  fail_producer
  run_index 'if create_index_html artifacts; then exit 1; fi; [[ ! -o pipefail ]]' 0
}

function test_pipefail_stays_enabled() {
  run_index 'set -o pipefail; create_index_html artifacts; [[ -o pipefail ]]' 0
}

run_suite "Release index generation tests"
