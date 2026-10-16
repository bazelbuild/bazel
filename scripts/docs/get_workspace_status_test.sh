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
status_script="$(rlocation io_bazel/scripts/docs/get_workspace_status.sh)"
release_common="$(rlocation io_bazel/scripts/release/common.sh)"

function set_up() {
  local test_dir
  test_dir="$(mktemp -d "${TEST_TMPDIR}/workspace_status.XXXXXXXX")"
  cd "${test_dir}"
  mkdir -p scripts/release bin
  unset BUILDKITE_BRANCH GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE
  export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null
  export GIT_AUTHOR_NAME="Test" GIT_AUTHOR_EMAIL="test@example.com"
  export GIT_COMMITTER_NAME="Test" GIT_COMMITTER_EMAIL="test@example.com"
  export REAL_GIT="$(command -v git)"
  export PATH="${test_dir}/bin:${PATH}"
}

function create_repository() {
  cp "${release_common}" scripts/release/common.sh
  git init -q
  git symbolic-ref HEAD refs/heads/main
  git -c commit.gpgsign=false commit --allow-empty -qm "Initial commit"
}

function assert_status() {
  bash "${status_script}" > stdout 2> "$TEST_log" || fail "Status script failed"
  assert_equals "BUILD_SCM_REVISION $1" "$(cat stdout)"
}

function assert_status_fails() {
  local status=0
  bash "${status_script}" > stdout 2> "$TEST_log" || status=$?
  assert_not_equals 0 "${status}"
  assert_empty_file stdout
}

function test_missing_setup() {
  cat > bin/git <<'EOF'
#!/usr/bin/env bash
echo main
EOF
  chmod +x bin/git
  assert_status_fails
}

function test_failed_setup() {
  cat > scripts/release/common.sh <<'EOF'
get_release_name() { echo 9.3.0; }
return 23
EOF
  assert_status_fails
}

function test_failed_release_lookup() {
  cat > scripts/release/common.sh <<'EOF'
get_release_name() { echo 9.3.0; return 19; }
EOF
  assert_status_fails
}

function test_failed_revision_lookup() {
  create_repository
  cat > bin/git <<'EOF'
#!/usr/bin/env bash
if [[ "$1" == rev-parse ]]; then
  echo partial-revision
  exit 17
fi
exec "${REAL_GIT}" "$@"
EOF
  chmod +x bin/git
  assert_status_fails
}

function test_outside_repository() {
  cp "${release_common}" scripts/release/common.sh
  assert_status_fails
}

function test_branch_without_tags() {
  create_repository
  assert_status UNSAFE_main
}

function test_release_branch() {
  create_repository
  git checkout -qb release-9.3.0rc1
  assert_status 9.3.0
}

function test_rolling_release_branch() {
  create_repository
  git checkout -qb release-10.0.0-pre.20261001.1rc1
  assert_status 10.0.0-pre.20261001.1
}

function test_release_tag() {
  create_repository
  git -c tag.gpgsign=false tag -a 9.3.0 -m "Release"
  git checkout --detach -q
  assert_status 9.3.0
}

function test_rolling_release_tag() {
  create_repository
  git -c tag.gpgsign=false tag 10.0.0-pre.20261001.1
  git checkout --detach -q
  assert_status 10.0.0-pre.20261001.1
}

function test_branch_after_release_tag() {
  create_repository
  git -c tag.gpgsign=false tag 9.3.0
  git -c commit.gpgsign=false commit --allow-empty -qm "Next commit"
  assert_status UNSAFE_main
}

function test_detached_head_without_tags() {
  create_repository
  git checkout --detach -q
  assert_status UNSAFE_HEAD
}

function test_detached_head_with_buildkite_branch() {
  create_repository
  git checkout --detach -q
  export BUILDKITE_BRANCH=release-9.3.0rc1
  assert_status 9.3.0
}

run_suite "Documentation workspace status tests"
