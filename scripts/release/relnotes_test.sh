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

### Setup a git repository
setup_git_repository

### Load the relnotes script
source ${SCRIPT_DIR}/relnotes.sh || { echo "relnotes.sh not found!" >&2; exit 1; }

### Tests method

function set_up() {
  cd ${MASTER_ROOT}
}

function test_format_release_notes() {
  local expected='  - Lorem ipsus I do not know more of latin than that but I need to
    type random text that spans multiple line so we can test that the
    wrapping of lines works as intended.
  - Another thing I must type.
  - Yet another test that spans across multiple lines so I must type
    some random stuff to test wrapping.'
  local input=("Lorem ipsus I do not know more of latin \
than that but I need to type random text that spans multiple line so we \
can test that the wrapping of lines works as intended."
"Another thing I must type."
"Yet another test that spans across multiple lines so I must type \
some random stuff to test wrapping.")
  assert_equals "${expected}" "$(format_release_notes "${input[@]}")"
}

function test_get_release_notes_commits() {
  # Generated with git log --grep RELNOTES.
  # Only 6d98f6c 53c0748 are removed (rollback).
  commits="0188971 957934c 7a99c7f b5ba24a c9041bf 8232d9b 422c731 e9029d4 \
cc44636 06b09ce 29b05c8 67944d8 e8f6647 6d9fb36 f7c9922 5c0e4b2 9e387dd \
98c9274 db4d861 a689f29 db487ce 965c392 bb59d88 d3461db cef25c4 14d905b"
  assert_equals "$commits" "$(get_release_notes_commits 00d7223 | xargs)"
  assert_equals "$(echo "$commits" | sed 's/957934c //')" \
      "$(get_release_notes_commits 00d7223 957934c | xargs)"
}

TEST_INC_CHANGE='Incompatible changes:

  - Remove deprecated "make var" INCDIR

'
TEST_NEW_CHANGE='New features:

  - added --with_aspect_deps to blaze query, that prints additional
    information about aspects of target when --output is set to {xml,
    proto, record}.

'
TEST_CHANGE='Important changes:

  - Use a default implementation of a progress message, rather than
    defaulting to null for all SpawnActions.
  - Attribute error messages related to Android resources are easier
    to understand now.'

function test_release_notes() {
  assert_equals "$TEST_INC_CHANGE$(echo)$TEST_NEW_CHANGE$(echo)$TEST_CHANGE" \
      "$(release_notes 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e)"
  assert_equals "$TEST_NEW_CHANGE$(echo)$TEST_CHANGE" \
      "$(release_notes 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e bb59d88)"
}

function test_get_last_release() {
  rm -f ${TEST_TMPDIR}/CHANGELOG.md
  if (get_last_release "${TEST_TMPDIR}/CHANGELOG.md"); then
    fail "Should have returned false for initial release"
  fi
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## No release
EOF
  if (get_last_release "${TEST_TMPDIR}/CHANGELOG.md"); then
    fail "Should have returned false when no release exists"
  fi
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## New release

Baseline: 965c392

Initial release without cherry-picks

EOF
  assert_equals "965c392" \
      "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"


  mv ${TEST_TMPDIR}/CHANGELOG.md ${TEST_TMPDIR}/CHANGELOG.md.bak
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88

Baseline: 965c392

Cherry picks:
   + bb59d88: RELNOTES[INC]: Remove deprecated "make var" INCDIR

$TEST_INC_CHANGE
EOF
  cat ${TEST_TMPDIR}/CHANGELOG.md.bak >>${TEST_TMPDIR}/CHANGELOG.md
  rm ${TEST_TMPDIR}/CHANGELOG.md.bak
  assert_equals "965c392 bb59d88" \
      "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"

  mv ${TEST_TMPDIR}/CHANGELOG.md ${TEST_TMPDIR}/CHANGELOG.md.bak
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88 and 14d905b

Baseline: 965c392

Cherry picks:
   + bb59d88: RELNOTES[INC]: Remove deprecated "make var" INCDIR
   + 14d905b: Add --with_aspect_deps flag to blaze query. This flag
              should produce additional information about aspect
              dependencies when --output is set to {xml, proto}.

$TEST_INC_CHANGE
$TEST_NEW_CHANGE
EOF
  cat ${TEST_TMPDIR}/CHANGELOG.md.bak >>${TEST_TMPDIR}/CHANGELOG.md
  rm ${TEST_TMPDIR}/CHANGELOG.md.bak
  assert_equals "965c392 bb59d88 14d905b" \
      "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"

}

function test_create_release_notes() {
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## New release

Baseline: 965c392

Initial release without cherry-picks

EOF
  assert_equals "$TEST_INC_CHANGE$(echo)$TEST_NEW_CHANGE$(echo)$TEST_CHANGE" \
      "$(create_release_notes ${TEST_TMPDIR}/CHANGELOG.md)"

  cat <<'EOF' >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88

```
Baseline: 965c392

Cherry picks:
   + bb59d88: RELNOTES[INC]: Remove deprecated "make var" INCDIR
```

EOF
  cat <<EOF >>${TEST_TMPDIR}/CHANGELOG.md
$TEST_INC_CHANGE
EOF
  assert_equals "$TEST_NEW_CHANGE$(echo)$TEST_CHANGE" \
      "$(create_release_notes ${TEST_TMPDIR}/CHANGELOG.md)"
  assert_equals "965c392 bb59d88" \
      "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"

  cat <<'EOF' >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88 and 14d905b

```
Baseline: 965c392

Cherry picks:
   + bb59d88: RELNOTES[INC]: Remove deprecated "make var" INCDIR
   + 14d905b: Add --with_aspect_deps flag to blaze query. This flag
              should produce additional information about aspect
              dependencies when --output is set to {xml, proto}.
```

EOF
  cat <<EOF >>${TEST_TMPDIR}/CHANGELOG.md
$TEST_INC_CHANGE
$TEST_NEW_CHANGE
EOF
  assert_equals "$TEST_CHANGE" \
      "$(create_release_notes ${TEST_TMPDIR}/CHANGELOG.md)"
}

function test_create_revision_information() {
  expected='Baseline: 965c392

Cherry picks:
   + bb59d88: RELNOTES[INC]: Remove deprecated "make var" INCDIR
   + 14d905b: Add --with_aspect_deps flag to blaze query. This flag
              should produce additional information about aspect
              dependencies when --output is set to {xml, proto}.'
   assert_equals "$expected" \
              "$(create_revision_information 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e bb59d88 14d905b5cce9a1bbc2911331809b03679b23dad1)"
}

run_suite "Release notes generation tests"
