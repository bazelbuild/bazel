#!/bin/bash

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

# Tests release notes generation (relnotes.sh)
set -eu

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source ${SCRIPT_DIR}/testenv.sh || { echo "testenv.sh not found!" >&2; exit 1; }

source ${SCRIPT_DIR}/common.sh || { echo "common.sh not found!" >&2; exit 1; }

RELEASE_SCRIPT=${SCRIPT_DIR}/release.sh

GERRIT_ROOT=${TEST_TMPDIR}/git/gerrit
GITHUB_ROOT=${TEST_TMPDIR}/git/github
WORKSPACE=${TEST_TMPDIR}/git/workspace
export RELEASE_REPOSITORIES="${GITHUB_ROOT}"
export MASTER_REPOSITORIES="${GITHUB_ROOT} ${GERRIT_ROOT}"

setup_git_repository

function set_up() {
  # Clean previous clones
  rm -fr ${GERRIT_ROOT} ${GITHUB_ROOT} ${WORKSPACE}
  # Now creates the clones
  git clone -l --bare -q ${MASTER_ROOT} ${GERRIT_ROOT}
  git clone -l --bare -q ${MASTER_ROOT} ${GITHUB_ROOT}
  # And the working copy
  git clone -l -q ${GERRIT_ROOT} ${WORKSPACE}
  cd ${WORKSPACE}
  # Avoid committer message
  cat >>.git/config <<EOF
[user]
    name = Bazel tests
    email = noreply@google.com
EOF
}

function create() {
  local name="$1"
  local commit="$2"
  if [[ "$1" =~ ^--force_rc=([0-9]*)$ ]]; then
    name="$2"
    commit="$3"
  fi
  local old_branch=$(git_get_branch)
  ${RELEASE_SCRIPT} create $@ &> $TEST_log \
    || fail "Failed to cut release $name at commit $commit"
  local new_branch=$(git_get_branch)
  assert_equals "release-$name" "$new_branch"
  assert_contains "Created $name.* on branch release-$name." $TEST_log
  get_full_release_notes "release-$name" >$TEST_log
}

function push() {
  local branch="release-$1"
  git checkout "$branch"
  ${RELEASE_SCRIPT} push || fail "Failed to push release branch $branch"
  git --git-dir=${GITHUB_ROOT} branch >$TEST_log
  expect_log "$branch"
  git --git-dir=${GERRIT_ROOT} branch >$TEST_log
  expect_not_log "$branch"
  assert_equals "$(git show -s --pretty=format:%B $branch)" \
      "$(git --git-dir=${GITHUB_ROOT} show -s --pretty=format:%B $branch)"
}

function release() {
  local tag=$1
  local branch=$(git_get_branch)
  echo y | ${RELEASE_SCRIPT} release || fail "Failed to release ${branch}"
  assert_equals master "$(git_get_branch)"
  git tag >$TEST_log
  expect_log $tag
  git --git-dir=${GITHUB_ROOT} tag >$TEST_log
  expect_log $tag
  git --git-dir=${GERRIT_ROOT} tag >$TEST_log
  expect_not_log $tag
  # Test commit is everywhere
  local commit="$(git show -s --pretty=format:%B $tag)"
  assert_equals "$commit" "$(git show -s --pretty=format:%B master)"
  assert_equals "$commit" \
      "$(git --git-dir=${GITHUB_ROOT} show -s --pretty=format:%B $tag)"
  assert_equals "$commit" \
      "$(git --git-dir=${GITHUB_ROOT} show -s --pretty=format:%B master)"
  assert_equals "$commit" \
      "$(git --git-dir=${GERRIT_ROOT} show -s --pretty=format:%B master)"

  # Now test for CHANGELOG.md file in master branch
  local changelog="$(git show $tag:CHANGELOG.md)"
  assert_equals "$changelog" "$(git show master:CHANGELOG.md)"
  assert_equals "$changelog" \
      "$(git --git-dir=${GITHUB_ROOT} show $tag:CHANGELOG.md)"
  assert_equals "$changelog" \
      "$(git --git-dir=${GITHUB_ROOT} show master:CHANGELOG.md)"
  assert_equals "$changelog" \
      "$(git --git-dir=${GERRIT_ROOT} show master:CHANGELOG.md)"

}

function abandon() {
  local tag="$1"
  local branch="release-$tag"
  git checkout "$branch"
  local changelog="$(git show master:CHANGELOG.md || true)"
  local master_sha1=$(git rev-parse master)
  echo y | ${RELEASE_SCRIPT} abandon || fail "Failed to abandon release ${branch}"
  assert_equals master "$(git_get_branch)"

  # test release was not tagged
  git tag >$TEST_log
  expect_not_log $tag
  git --git-dir=${GITHUB_ROOT} tag >$TEST_log
  expect_not_log $tag
  git --git-dir=${GERRIT_ROOT} tag >$TEST_log
  expect_not_log $tag

  # Test branch was deleted
  git branch >$TEST_log
  expect_not_log $branch
  git --git-dir=${GITHUB_ROOT} branch >$TEST_log
  expect_not_log $branch

  # Test the master branch commit hasn't changed
  assert_equals "$(git rev-parse master)" "${master_sha1}"

  # Now test for CHANGELOG.md file in master branch hasn't changed
  assert_equals "$changelog" "$(git show master:CHANGELOG.md)"
  assert_equals "$changelog" \
      "$(git --git-dir=${GITHUB_ROOT} show master:CHANGELOG.md)"
  assert_equals "$changelog" \
      "$(git --git-dir=${GERRIT_ROOT} show master:CHANGELOG.md)"

}

function test_merge_release_notes() {
  local RELNOTES='Incompatible changes:

  - Remove deprecated "make var" INCDIR

Important changes:

  - Use a default implementation of a progress message, rather than
    defaulting to null for all SpawnActions.'
  local NEW_RELNOTES="${RELNOTES}"'
  - Attribute error messages related to Android resources are easier
    to understand now.'
  local REPLACEMENT="Test replacement."

  assert_equals '<<<<<<< HEAD
Incompatible changes:

  - Remove deprecated "make var" INCDIR

Important changes:

  - Use a default implementation of a progress message, rather than
    defaulting to null for all SpawnActions.
  - Attribute error messages related to Android resources are easier
    to understand now.
=======
Test replacement.
>>>>>>> master-merge-notes-1' "$(merge_release_notes master "${NEW_RELNOTES}" "${RELNOTES}" "${REPLACEMENT}")"

  assert_equals "${NEW_RELNOTES}" \
    "$(merge_release_notes master "${NEW_RELNOTES}" "${RELNOTES}" "${RELNOTES}")"

  assert_equals "${RELNOTES}"'
<<<<<<< HEAD
  - Attribute error messages related to Android resources are easier
    to understand now.
=======
  - Merge conflict.
>>>>>>> master-merge-notes-1' "$(merge_release_notes master "${NEW_RELNOTES}" "${RELNOTES}" "${RELNOTES}
  - Merge conflict.")"

}

function test_release_workflow() {
  export EDITOR=true
  # Initial release
  create v0 965c392
  expect_log "Release v0rc1"
  expect_log "Initial release"
  # Push the release branch
  push v0
  # Do the initial release
  release v0

  CHANGELOG='## Release v0 ('$(date +%Y-%m-%d)')

```
Baseline: 965c392
```

Initial release.'
  assert_equals "${CHANGELOG}" "$(<CHANGELOG.md)"

  # Second release.

  # First we need to edit the logs
  export EDITOR=${TEST_TMPDIR}/editor.sh
  local RELNOTES='Incompatible changes:

  - Remove deprecated "make var" INCDIR

Important changes:

  - Use a default implementation of a progress message, rather than
    defaulting to null for all SpawnActions.'

  cat >${TEST_TMPDIR}/expected.log <<EOF
# Editing release notes
# Modify the release notes to make them suitable for the release.
# Every line starting with a # will be removed as well as every
# empty line at the start and at the end.

# Release v1rc1 ($(date +%Y-%m-%d))

${RELNOTES}

EOF

  echo "Test replacement" >${TEST_TMPDIR}/replacement.log

  cat >${EDITOR} <<EOF
#!/bin/bash

# 1. Assert the file is correct
if [ "\$(cat \$1)" != "\$(cat ${TEST_TMPDIR}/expected.log)" ]; then
  echo "Expected:" >&2
  cat ${TEST_TMPDIR}/expected.log >&2
  echo "Got:" >&2
  cat \$1 >&2
  exit 1
fi

# 2. write the replacement in the input file
cat ${TEST_TMPDIR}/replacement.log >\$1
EOF
  chmod +x ${EDITOR}
  create v1 1170dc6 0540fde
  local header='Release v1rc1 ('$(date +%Y-%m-%d)')

Baseline: 1170dc6

Cherry picks:
   + 0540fde: Extract version numbers that look like "..._1.2.3_..."
              from BUILD_EMBED_LABEL into Info.plist.

'
  assert_equals "${header}Test replacement" "$(cat ${TEST_log})"
  assert_equals "Test replacement" "$(get_release_notes release-v1)"
  assert_equals 1 "$(get_release_candidate release-v1)"
  push v1

  # Test creating a second candidate
  RELNOTES="${RELNOTES}"'
  - Attribute error messages related to Android resources are easier
    to understand now.'

  # There should be a merge conflict
  cat >${TEST_TMPDIR}/expected.log <<EOF
# Editing release notes
# Modify the release notes to make them suitable for the release.
# Every line starting with a # will be removed as well as every
# empty line at the start and at the end.

# Release v1rc2 ($(date +%Y-%m-%d))

<<<<<<< HEAD
${RELNOTES}
=======
Test replacement
>>>>>>> release-v1-merge-notes-1
EOF
  echo "${RELNOTES}" >${TEST_TMPDIR}/replacement.log

  create v1 1170dc6 0540fde cef25c4
  title='Release v1rc2 ('$(date +%Y-%m-%d)')'
  revision_info='Baseline: 1170dc6

Cherry picks:
   + 0540fde: Extract version numbers that look like "..._1.2.3_..."
              from BUILD_EMBED_LABEL into Info.plist.
   + cef25c4: RELNOTES: Attribute error messages related to Android
              resources are easier to understand now.'
  header="${title}

${revision_info}

"
  assert_equals "${header}${RELNOTES}" "$(cat ${TEST_log})"
  assert_equals "${RELNOTES}" "$(get_release_notes release-v1)"
  assert_equals 2 "$(get_release_candidate release-v1)"

  # Push the release
  push v1
  release v1
  title='Release v1 ('$(date +%Y-%m-%d)')'
  CHANGELOG='## '"${title}"'

```
'"${revision_info}"'
```

'"${RELNOTES}"'

'"${CHANGELOG}"
  assert_equals "${CHANGELOG}" "$(<CHANGELOG.md)"

  # Third release to test abandon
  cat >${EDITOR} <<EOF
#!/bin/bash
# Make sure we have release notes or the release will be cancelled.
echo 'Dummy release' >\$1
EOF
  # Create release
  create --force_rc=2 v2 2464526
  expect_log "Release v2rc2"
  expect_log "Baseline: 2464526"
  assert_equals 2 "$(get_release_candidate release-v2)"
  # Abandon it
  abandon v2
  # Add a commit hook to test if it is ignored
  cat <<'EOF' >.git/hooks/commit-msg
echo HOOK-SHOULD-BE-IGNORED >>$1
EOF
  chmod +x .git/hooks/commit-msg
  # Re-create release
  create v2 2464526
  expect_log "Release v2rc1"
  expect_log "Baseline: 2464526"
  expect_not_log "HOOK-SHOULD-BE-IGNORED"
  # Push
  push v2
  # Abandon it
  abandon v2
}

function generate_rc() {
  local force_rc=
  if [[ "$1" =~ ^--force_rc=[0-9]+$ ]]; then
    force_rc="$1"
    shift
  fi
  local name="$1"
  shift
  if (git rev-parse --verify "release-$name" &>/dev/null); then
    git checkout release-"$name"
  else
    git checkout -b release-"$name" $1
    shift
  fi
  for i in "$@"; do
    git cherry-pick $i
  done
  ${RELEASE_SCRIPT} generate-rc $force_rc &> $TEST_log \
    || fail "Failed to cut release $name"
  get_full_release_notes "release-$name" >$TEST_log
  git checkout master
}

# Same test as before with the workflow for the git user
function test_git_release_workflow() {
  export EDITOR=true
  # Initial release
  generate_rc v0 965c392

  expect_log "Release v0rc1"
  expect_log "Initial release"
  # Push the release branch
  push v0
  # Do the initial release
  release v0

  # Second release.

  # First we need to edit the logs
  export EDITOR=${TEST_TMPDIR}/editor.sh
  local RELNOTES='Incompatible changes:

  - Remove deprecated "make var" INCDIR

Important changes:

  - Use a default implementation of a progress message, rather than
    defaulting to null for all SpawnActions.'

  cat >${TEST_TMPDIR}/expected.log <<EOF
# Editing release notes
# Modify the release notes to make them suitable for the release.
# Every line starting with a # will be removed as well as every
# empty line at the start and at the end.

# Release v1rc1 ($(date +%Y-%m-%d))

${RELNOTES}

EOF

  echo "Test replacement" >${TEST_TMPDIR}/replacement.log

  cat >${EDITOR} <<EOF
#!/bin/bash

# 1. Assert the file is correct
if [ "\$(cat \$1)" != "\$(cat ${TEST_TMPDIR}/expected.log)" ]; then
  echo "Expected:" >&2
  cat ${TEST_TMPDIR}/expected.log >&2
  echo "Got:" >&2
  cat \$1 >&2
  exit 1
fi

# 2. write the replacement in the input file
cat ${TEST_TMPDIR}/replacement.log >\$1
EOF
  chmod +x ${EDITOR}
  generate_rc v1 1170dc6 0540fde
  local header='Release v1rc1 ('$(date +%Y-%m-%d)')

Baseline: 1170dc6

Cherry picks:
   + 0540fde: Extract version numbers that look like "..._1.2.3_..."
              from BUILD_EMBED_LABEL into Info.plist.

'
  assert_equals "${header}Test replacement" "$(cat ${TEST_log})"
  assert_equals "Test replacement" "$(get_release_notes release-v1)"
  assert_equals 1 "$(get_release_candidate release-v1)"
  push v1

  # Test creating a second candidate
  RELNOTES="${RELNOTES}"'
  - Attribute error messages related to Android resources are easier
    to understand now.'

  # There should be a merge conflict
  cat >${TEST_TMPDIR}/expected.log <<EOF
# Editing release notes
# Modify the release notes to make them suitable for the release.
# Every line starting with a # will be removed as well as every
# empty line at the start and at the end.

# Release v1rc2 ($(date +%Y-%m-%d))

<<<<<<< HEAD
${RELNOTES}
=======
Test replacement
>>>>>>> release-v1-merge-notes-1
EOF
  echo "${RELNOTES}" >${TEST_TMPDIR}/replacement.log

  generate_rc v1 cef25c4
  header='Release v1rc2 ('$(date +%Y-%m-%d)')

Baseline: 1170dc6

Cherry picks:
   + 0540fde: Extract version numbers that look like "..._1.2.3_..."
              from BUILD_EMBED_LABEL into Info.plist.
   + cef25c4: RELNOTES: Attribute error messages related to Android
              resources are easier to understand now.

'
  assert_equals "${header}${RELNOTES}" "$(cat ${TEST_log})"
  assert_equals "${RELNOTES}" "$(get_release_notes release-v1)"
  assert_equals 2 "$(get_release_candidate release-v1)"

  # Push the release
  push v1
  release v1

  # Third release to test abandon
  cat >${EDITOR} <<EOF
#!/bin/bash
# Make sure we have release notes or the release will be cancelled.
echo 'Dummy release' >\$1
EOF
  # Create release
  generate_rc --force_rc=2 v2 2464526
  expect_log "Release v2rc2"
  expect_log "Baseline: 2464526"
  assert_equals 2 "$(get_release_candidate release-v2)"
  # Abandon it
  abandon v2
  # Add a commit hook to test if it is ignored
  cat <<'EOF' >.git/hooks/commit-msg
echo HOOK-SHOULD-BE-IGNORED >>$1
EOF
  chmod +x .git/hooks/commit-msg
  # Re-create release
  generate_rc v2 2464526
  expect_log "Release v2rc1"
  expect_log "Baseline: 2464526"
  expect_not_log "HOOK-SHOULD-BE-IGNORED"
  # Push
  push v2
  # Abandon it
  abandon v2
}

run_suite "Release tests"
