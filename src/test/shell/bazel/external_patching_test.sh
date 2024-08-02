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
  setup_module_dot_bazel "MODULE.bazel"
  # create an archive file with files interesting for patching
  mkdir ext-0.1.2
  cat > ext-0.1.2/foo.sh <<'EOF'
#!/usr/bin/env sh

echo Here be dragons...
EOF
  zip ext.zip ext-0.1.2/*
  rm -rf ext-0.1.2
}

function get_extrepourl() {
  if $is_windows; then
    echo "file:///$(cygpath -m $1)"
  else
    echo "file://$1"
  fi
}

test_patch_file() {
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"

  # Test that the patches attribute of http_archive is honored
  mkdir main
  cd main
  cat > patch_foo.sh <<'EOF'
--- foo.sh.orig	2018-01-15 10:39:20.183909147 +0100
+++ foo.sh	2018-01-15 10:43:35.331566052 +0100
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo There are dragons...
EOF
  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["${EXTREPOURL}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
  patches = ["//:patch_foo.sh"],
  patch_cmds = ["find . -name '*.sh' -exec sed -i.orig '1s|#!/usr/bin/env sh\$|/bin/sh\$|' {} +"],
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.sh"],
  srcs = ["@ext//:foo.sh"],
  cmd = "cp $< $@; chmod u+x $@",
  executable = True,
)
EOF
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'There are' $foopath || fail "expected patch to be applied"
  grep env $foopath && fail "expected patch commands to be executed" || :

  # Verify that changes to the patch files trigger enough rebuilding
  cat > patch_foo.sh <<'EOF'
--- foo.sh.orig	2018-01-15 10:39:20.183909147 +0100
+++ foo.sh	2018-01-15 10:43:35.331566052 +0100
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo completely differently patched
EOF
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'differently patched' $foopath \
      || fail "expected the new patch to be applied"

  # Verify that changes to the patches attribute trigger enough rebuilding
  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["${EXTREPOURL}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'Here be' $foopath || fail "expected unpatched file"
  grep -q 'env' $foopath || fail "expected unpatched file"
}

test_patch_failed() {
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"

  cat > my_patch_tool <<'EOF'
#!/bin/sh

echo Helpful message
exit 1
EOF
  chmod u+x my_patch_tool

  # Test that the patches attribute of http_archive is honored
  mkdir main
  cd main
  echo "ignored anyway" > patch_foo.sh
  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["${EXTREPOURL}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
  patches = ["//:patch_foo.sh"],
  patch_tool = "${EXTREPODIR}/my_patch_tool",
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  touch BUILD

  bazel build @ext//... >"${TEST_log}" 2>&1 && fail "expected failure" || :
  expect_log 'Helpful message'
}

test_remote_patch_on_top_of_local_patch() {
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"
  # Generate the remote patch file
  cat > remote.patch <<'EOF'
--- a/foo.sh	2018-01-15 10:39:20.183909147 +0100
+++ b/foo.sh	2018-01-15 10:43:35.331566052 +0100
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo There are dragons...
EOF
  integrity="sha256-$(cat remote.patch | openssl dgst -sha256 -binary | openssl base64 -A)"

  mkdir main
  cd main

  # Generate the local patch file
  cat > local.patch <<'EOF'
--- foo.sh.orig	2021-07-05 15:16:49.000000000 +0200
+++ foo.sh	2021-07-05 15:17:15.000000000 +0200
@@ -1,3 +1,3 @@
-#!/usr/bin/env sh
+#!/bin/sh

 echo There are dragons...
EOF

  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["${EXTREPOURL}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
  remote_patches = {"${EXTREPOURL}/remote.patch": "${integrity}"},
  remote_patch_strip = 1,
  patches = ["//:local.patch"],
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.sh"],
  srcs = ["@ext//:foo.sh"],
  cmd = "cp $< $@; chmod u+x $@",
  executable = True,
)
EOF

  bazel build :foo.sh
  foopath=`bazel info bazel-bin`/foo.sh
  grep -q 'There are' $foopath || fail "expected remote patch to be applied"
  grep -q '/bin/sh' $foopath || fail "expected local patch to be applied"
}

test_remote_patch_integrity_incorrect() {
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"
  # Generate the remote patch file
  cat > remote.patch <<'EOF'
--- a/foo.sh	2018-01-15 10:39:20.183909147 +0100
+++ b/foo.sh	2018-01-15 10:43:35.331566052 +0100
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo There are dragons...
EOF

  mkdir main
  cd main

  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["${EXTREPOURL}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
  remote_patches = {"${EXTREPOURL}/remote.patch": "sha256-Yab3Yqr2BlLL8zKHm43MLP2BviEpoGHalX0Dnq538LA="},
  remote_patch_strip = 1,
  patches = ["//:local.patch"],
)
EOF
  setup_module_dot_bazel "MODULE.bazel"

  bazel build @ext//... &> $TEST_log 2>&1 && fail "Expected to fail"
  expect_log "Error downloading \\[.*/remote.patch\\] to"
  expect_log "but wanted sha256-Yab3Yqr2BlLL8zKHm43MLP2BviEpoGHalX0Dnq538LA="
}

test_remote_patches_with_same_base_name() {
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"

  mkdir a
  # Generate a remote patch file
  cat > a/remote.patch <<'EOF'
--- a/foo.sh	2018-01-15 10:39:20.183909147 +0100
+++ b/foo.sh	2018-01-15 10:43:35.331566052 +0100
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo There are dragons...
EOF
  integrity_a="sha256-$(cat a/remote.patch | openssl dgst -sha256 -binary | openssl base64 -A)"

  mkdir b
  # Generate another remote patch file with the same base name
  cat > b/remote.patch <<'EOF'
--- a/foo.sh	2021-07-05 15:16:49.000000000 +0200
+++ b/foo.sh	2021-07-05 15:17:15.000000000 +0200
@@ -1,3 +1,3 @@
-#!/usr/bin/env sh
+#!/bin/sh

 echo There are dragons...
EOF
  integrity_b="sha256-$(cat b/remote.patch | openssl dgst -sha256 -binary | openssl base64 -A)"

  mkdir main
  cd main

  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["${EXTREPOURL}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
  remote_patches = {
    "${EXTREPOURL}/a/remote.patch": "$integrity_a",
    "${EXTREPOURL}/b/remote.patch": "$integrity_b",
  },
  remote_patch_strip = 1,
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.sh"],
  srcs = ["@ext//:foo.sh"],
  cmd = "cp $< $@; chmod u+x $@",
  executable = True,
)
EOF

  bazel build :foo.sh
  foopath=`bazel info bazel-bin`/foo.sh
  grep -q 'There are' $foopath || fail "expected a/remote.patch to be applied"
  grep -q '/bin/sh' $foopath || fail "expected b/remote.patch to be applied"
}

test_patch_git() {
  EXTREPODIR=`pwd`
  if $is_windows; then
    EXTREPODIR="$(cygpath -m ${EXTREPODIR})"
  fi

  export GIT_CONFIG_NOSYSTEM=YES

  mkdir extgit
  (cd extgit && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  cat > extgit/foo.sh <<'EOF'
#!/usr/bin/env sh

echo Here be dragons...
EOF
  (cd extgit
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'initial commit'
   git tag mytag)

  # Test that the patches attribute of git_repository is honored
  mkdir main
  cd main
  cat > patch_foo.sh <<'EOF'
--- foo.sh.orig	2018-01-15 10:39:20.183909147 +0100
+++ foo.sh	2018-01-15 10:43:35.331566052 +0100
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo There are dragons...
EOF
  cat > MODULE.bazel <<EOF
new_git_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="${EXTREPODIR}/extgit/.git",
  tag="mytag",
  build_file_content="exports_files([\"foo.sh\"])",
  patches = ["//:patch_foo.sh"],
  patch_cmds = ["find . -name '*.sh' -exec sed -i.orig '1s|#!/usr/bin/env sh\$|/bin/sh\$|' {} +"],
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.sh"],
  srcs = ["@ext//:foo.sh"],
  cmd = "cp $< $@; chmod u+x $@",
  executable = True,
)
EOF
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'There are' $foopath || fail "expected patch to be applied"
  grep env $foopath && fail "expected patch commands to be executed" || :

  # Verify that changes to the patch files trigger enough rebuilding
  cat > patch_foo.sh <<'EOF'
--- foo.sh.orig	2018-01-15 10:39:20.183909147 +0100
+++ foo.sh	2018-01-15 10:43:35.331566052 +0100
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo completely differently patched
EOF
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'differently patched' $foopath \
      || fail "expected the new patch to be applied"

  # Verify that changes to the patches attribute trigger enough rebuilding
  cat > MODULE.bazel <<EOF
new_git_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="${EXTREPODIR}/extgit/.git",
  tag="mytag",
  build_file_content="exports_files([\"foo.sh\"])",
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'Here be' $foopath || fail "expected unpatched file"
  grep -q 'env' $foopath || fail "expected unpatched file"
}

test_override_buildfile() {
  ## Verify that the BUILD file of an external repository can be overridden
  ## via the http_archive rule.
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"

  mkdir withbuild
  cat > withbuild/BUILD <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo BAD >> $@",
)
EOF
  cat > withbuild/file.txt <<'EOF'
from external repo
EOF
  zip withbuild.zip withbuild/*
  rm -rf withbuild

  mkdir main
  cd main
  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="withbuild",
  strip_prefix="withbuild",
  urls=["${EXTREPOURL}/withbuild.zip"],
  build_file="@//:ext.BUILD",
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  outs = ["local.txt"],
  srcs = ["@withbuild//:target"],
  cmd = "cp $< $@",
)
EOF
  cat > ext.BUILD <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo GOOD >> $@",
  visibility=["//visibility:public"],
)
EOF

  bazel build //:local || fail "Expected success"

  cat `bazel info bazel-genfiles`/local.txt > "${TEST_log}"
  expect_log "from external repo"
  expect_log "GOOD"
  expect_not_log "BAD"
}

test_override_buildfile_content() {
  ## Verify that the BUILD file of an external repository can be overridden
  ## via specified content in the http_archive rule.
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"

  mkdir withbuild
  cat > withbuild/BUILD <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo BAD >> $@",
)
EOF
  cat > withbuild/file.txt <<'EOF'
from external repo
EOF
  zip withbuild.zip withbuild/*
  rm -rf withbuild

  mkdir main
  cd main
  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="withbuild",
  strip_prefix="withbuild",
  urls=["${EXTREPOURL}/withbuild.zip"],
  build_file_content="""
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp \$< \$@ && echo GOOD >> \$@",
  visibility=["//visibility:public"],
)
  """,
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  outs = ["local.txt"],
  srcs = ["@withbuild//:target"],
  cmd = "cp $< $@",
)
EOF

  bazel build //:local || fail "Expected success"

  cat `bazel info bazel-genfiles`/local.txt > "${TEST_log}"
  expect_log "from external repo"
  expect_log "GOOD"
  expect_not_log "BAD"
}

test_override_buildfile_git() {
  ## Verify that the BUILD file of an external repository can be overridden
  ## via the git_repository rule.
  EXTREPODIR=`pwd`
  if $is_windows; then
    EXTREPODIR="$(cygpath -m ${EXTREPODIR})"
  fi

  export GIT_CONFIG_NOSYSTEM=YES

  mkdir withbuild
  (cd withbuild && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  cat > withbuild/BUILD.bazel <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo BAD >> $@",
  visibility=["//visibility:public"],
)
EOF
  cat > withbuild/file.txt <<'EOF'
from external repo
EOF
  (cd withbuild
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'initial commit'
   git tag mytag)

  mkdir main
  cd main
  cat > MODULE.bazel <<EOF
new_git_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="withbuild",
  remote="${EXTREPODIR}/withbuild/.git",
  tag="mytag",
  build_file="@//:ext.BUILD",
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  outs = ["local.txt"],
  srcs = ["@withbuild//:target"],
  cmd = "cp $< $@",
)
EOF
  cat > ext.BUILD <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo GOOD >> $@",
  visibility=["//visibility:public"],
)
EOF

  bazel build //:local || fail "Expected success"

  cat `bazel info bazel-genfiles`/local.txt > "${TEST_log}"
  expect_log "from external repo"
  expect_log "GOOD"
  expect_not_log "BAD"
}

test_override_buildfilecontents_git() {
  ## Verify that the BUILD file of an external repository can be overridden
  ## via specified content in the git_repository rule.
  EXTREPODIR=`pwd`
  if $is_windows; then
    EXTREPODIR="$(cygpath -m ${EXTREPODIR})"
  fi

  export GIT_CONFIG_NOSYSTEM=YES

  mkdir withbuild
  (cd withbuild && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  cat > withbuild/BUILD.bazel <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo BAD >> $@",
  visibility=["//visibility:public"],
)
EOF
  cat > withbuild/file.txt <<'EOF'
from external repo
EOF
  (cd withbuild
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'initial commit'
   git tag mytag)

  mkdir main
  cd main
  cat > MODULE.bazel <<EOF
new_git_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="withbuild",
  remote="${EXTREPODIR}/withbuild/.git",
  tag="mytag",
  build_file_content="""
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp \$< \$@ && echo GOOD >> \$@",
  visibility=["//visibility:public"],
)
  """,
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  outs = ["local.txt"],
  srcs = ["@withbuild//:target"],
  cmd = "cp $< $@",
)
EOF

  bazel build //:local || fail "Expected success"

  cat `bazel info bazel-genfiles`/local.txt > "${TEST_log}"
  expect_log "from external repo"
  expect_log "GOOD"
  expect_not_log "BAD"
}

test_build_file_build_bazel() {
  ## Verify that the BUILD file of an external repository can be overridden
  ## via the http_archive rule.
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"
  mkdir withbuild
  cat > withbuild/BUILD.bazel <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo BAD >> $@",
)
EOF
  cat > withbuild/file.txt <<'EOF'
from external repo
EOF
  zip withbuild.zip withbuild/*
  rm -rf withbuild

  mkdir main
  cd main
  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="withbuild",
  strip_prefix="withbuild",
  urls=["${EXTREPOURL}/withbuild.zip"],
  build_file="@//:ext.BUILD",
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  outs = ["local.txt"],
  srcs = ["@withbuild//:target"],
  cmd = "cp $< $@",
)
EOF
  cat > ext.BUILD <<'EOF'
genrule(
  name="target",
  srcs=["file.txt"],
  outs=["target.txt"],
  cmd="cp $< $@ && echo GOOD >> $@",
  visibility=["//visibility:public"],
)
EOF

  bazel build //:local || fail "Expected success"

  cat `bazel info bazel-genfiles`/local.txt > "${TEST_log}"
  expect_log "from external repo"
  expect_log "GOOD"
  expect_not_log "BAD"
}

test_git_format_patch() {
  EXTREPODIR=`pwd`
  EXTREPOURL="$(get_extrepourl ${EXTREPODIR})"

  # Verify that a patch in the style of git-format-patch(1) can be handled.
  mkdir main
  cd main
  ls -al
  cat > 0001-foo.sh-remove-dragons.patch <<'EOF'
From a8c0f9248dd85feac9d08b017776b5aedd1e7be8 Mon Sep 17 00:00:00 2001
From: Klaus Aehlig <aehlig@google.com>
Date: Wed, 13 Jun 2018 11:32:39 +0200
Subject: [PATCH] foo.sh: remove dragons

---
 foo.sh | 2 +-
 1 file changed, 1 insertion(+), 1 deletion(-)

diff --git a/foo.sh b/foo.sh
index 1f4c41e..9d548ff 100644
--- a/foo.sh
+++ b/foo.sh
@@ -1,3 +1,3 @@
 #!/usr/bin/env sh

-echo Here be dragons...
+echo New version of foo.sh, no more dangerous animals...
--
2.18.0.rc1.244.gcf134e6275-goog

EOF
  cat > MODULE.bazel <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["${EXTREPOURL}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
  patches = ["//:0001-foo.sh-remove-dragons.patch"],
  patch_args = ["-p1"],
  patch_cmds = ["find . -name '*.sh' -exec sed -i.orig '1s|#!/usr/bin/env sh\$|/bin/sh\$|' {} +"],
)
EOF
  setup_module_dot_bazel "MODULE.bazel"
  cat > BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.sh"],
  srcs = ["@ext//:foo.sh"],
  cmd = "cp $< $@; chmod u+x $@",
  executable = True,
)
EOF
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'New version' $foopath || fail "expected patch to be applied"
}

run_suite "external patching tests"
