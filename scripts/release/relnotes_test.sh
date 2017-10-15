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
source ${SCRIPT_DIR}/common.sh || { echo "common.sh not found!" >&2; exit 1; }
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
  commits='01889715e70b55b9d197e546593878f16cdc0f26 957934c40f73e96a4414c6a9efbc165367459b4b 7a99c7f47705bbb4ff8617f4876bc0298093a556 b5ba24a3f3ee0e7da718bf8becac96d691ae2074 c9041bf6f629b1441b6131ca495d8e6d0fb84f42 8232d9ba85b26cb4d10588a39d7a7adafeb5c4af 422c731fbefb098962813b3e0914a9192c72e549 e9029d4613d98c17e05236a0058164bb8787f94b cc44636d2d538bc91e7291ed4607f2bdce356827 06b09ce978eb984bee3a83ed446aab2dce60fa43 29b05c8e6c48b4028a06cd788d833506cce090eb 67944d866d4b74b9c4af51d5097a51fed5a6c30e e8f664780e3089b0af8b267effdec0f3242843ad 6d9fb360b79ec040e423b20b72a9cc3c4bac5b54 f7c992263610c9246a2c81b4e015b9c7f216fd50 5c0e4b2c64e9c9ccf80607ce4d8855ad032c302f 9e387ddc2fbeb6c88400e8b9fcf4e1d1fc600be7 98c92744557330d844ff5c38a28e5419d153ed1f db4d8619023693c97e5afb467737084ccd30b311 a689f2900911039d2c10e6de7d41fbf1bdf31f44 db487ce72207a340589182bbd85b84d1a9375bd1 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e bb59d88448d3365ff3ec168c1431cd86c5a5f02c d3461dba46b50719e238939946048cd1ca12755a cef25c44bc6c2ae8e5bd649228a9a9c39f057576 14d905b5cce9a1bbc2911331809b03679b23dad1'
  assert_equals "$commits" "$(get_release_notes_commits 00d7223 | xargs)"
  assert_equals "$(echo "$commits" | sed 's/957934c[0-9a-f]* //')" \
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
      "$(release_notes 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e \
      bb59d88)"
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

Baseline: 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e

Initial release without cherry-picks

EOF
  assert_equals "965c392ab1d68d5bc23fdef3d86d635ec9d2da8e" \
      "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"


  mv ${TEST_TMPDIR}/CHANGELOG.md ${TEST_TMPDIR}/CHANGELOG.md.bak
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88448d3365ff3ec168c1431cd86c5a5f02c

Baseline: 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e

Cherry picks:
   + bb59d88448d3365ff3ec168c1431cd86c5a5f02c: RELNOTES[INC]: Remove deprecated "make var" INCDIR

$TEST_INC_CHANGE
EOF
  cat ${TEST_TMPDIR}/CHANGELOG.md.bak >>${TEST_TMPDIR}/CHANGELOG.md
  rm ${TEST_TMPDIR}/CHANGELOG.md.bak
  assert_equals "965c392ab1d68d5bc23fdef3d86d635ec9d2da8e bb59d88448d3365ff3ec168c1431cd86c5a5f02c" \
      "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"

  mv ${TEST_TMPDIR}/CHANGELOG.md ${TEST_TMPDIR}/CHANGELOG.md.bak
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88448d3365ff3ec168c1431cd86c5a5f02c and
## 14d905b5cce9a1bbc2911331809b03679b23dad1:

Baseline: 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e

Cherry picks:
   + bb59d88448d3365ff3ec168c1431cd86c5a5f02c:
     RELNOTES[INC]: Remove deprecated "make var" INCDIR
   + 14d905b5cce9a1bbc2911331809b03679b23dad1:
     Add --with_aspect_deps flag to blaze query. This flag should
     produce additional information about aspect dependencies when
     --output is set to {xml, proto}.

$TEST_INC_CHANGE
$TEST_NEW_CHANGE
EOF
  cat ${TEST_TMPDIR}/CHANGELOG.md.bak >>${TEST_TMPDIR}/CHANGELOG.md
  rm ${TEST_TMPDIR}/CHANGELOG.md.bak
  assert_equals \
    "965c392ab1d68d5bc23fdef3d86d635ec9d2da8e bb59d88448d3365ff3ec168c1431cd86c5a5f02c 14d905b5cce9a1bbc2911331809b03679b23dad1" \
    "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"

}

function test_create_release_notes() {
  cat <<EOF >${TEST_TMPDIR}/CHANGELOG.md
## New release

Baseline: 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e

Initial release without cherry-picks

EOF
  assert_equals "$TEST_INC_CHANGE$(echo)$TEST_NEW_CHANGE$(echo)$TEST_CHANGE" \
      "$(create_release_notes ${TEST_TMPDIR}/CHANGELOG.md)"

  cat <<'EOF' >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88448d3365ff3ec168c1431cd86c5a5f02c

```
Baseline: 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e

Cherry picks:
   + bb59d88448d3365ff3ec168c1431cd86c5a5f02c:
     RELNOTES[INC]: Remove deprecated "make var" INCDIR
```

EOF
  cat <<EOF >>${TEST_TMPDIR}/CHANGELOG.md
$TEST_INC_CHANGE
EOF
  assert_equals "$TEST_NEW_CHANGE$(echo)$TEST_CHANGE" \
      "$(create_release_notes ${TEST_TMPDIR}/CHANGELOG.md)"
  assert_equals "965c392ab1d68d5bc23fdef3d86d635ec9d2da8e bb59d88448d3365ff3ec168c1431cd86c5a5f02c" \
      "$(get_last_release "${TEST_TMPDIR}/CHANGELOG.md")"

  cat <<'EOF' >${TEST_TMPDIR}/CHANGELOG.md
## Cherry-picking bb59d88448d3365ff3ec168c1431cd86c5a5f02c and
## 14d905b5cce9a1bbc2911331809b03679b23dad1:

```
Baseline: 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e

Cherry picks:
   + bb59d88448d3365ff3ec168c1431cd86c5a5f02c:
     RELNOTES[INC]: Remove deprecated "make var" INCDIR
   + 14d905b5cce9a1bbc2911331809b03679b23dad1:
     Add --with_aspect_deps flag to blaze query. This flag should
     produce additional information about aspect dependencies when
     --output is set to {xml, proto}.
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
  expected='Baseline: 965c392ab1d68d5bc23fdef3d86d635ec9d2da8e

Cherry picks:
   + bb59d88448d3365ff3ec168c1431cd86c5a5f02c:
     RELNOTES[INC]: Remove deprecated "make var" INCDIR
   + 14d905b5cce9a1bbc2911331809b03679b23dad1:
     Add --with_aspect_deps flag to blaze query. This flag should
     produce additional information about aspect dependencies when
     --output is set to {xml, proto}.'
  actual="$(create_revision_information \
    965c392ab1d68d5bc23fdef3d86d635ec9d2da8e \
    bb59d88448d3365ff3ec168c1431cd86c5a5f02c \
    14d905b5cce9a1bbc2911331809b03679b23dad1)"
  assert_equals "$expected" "$actual"
}

function test_extract_release_note_for_pre_copybara_commits() {
  local expected='added --with_aspect_deps to blaze query, that prints additional information about aspects of target when --output is set to {xml, proto, record}.'
  extract_release_note 14d905b5cce9a1bbc2911331809b03679b23dad1
  local actual=$(printf "%s\n"  "${RELNOTES_NEW[@]}")
  assert_equals "${expected}" "${actual}"
}

function test_extract_release_note_for_post_copybara_commits() {
  local expected="'output_groups' and 'instrumented_files' cannot be specified in DefaultInfo."
  extract_release_note e788964a6ebc2c4966456ac74044f4f44a126fe5
  local actual=$(printf "%s\n"  "${RELNOTES_[@]}")
  assert_equals "${expected}" "${actual}"
}

run_suite "Release notes generation tests"
