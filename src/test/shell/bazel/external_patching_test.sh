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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }


set_up() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"
  # create an archive file with files interesting for patching
  mkdir ext-0.1.2
  cat > ext-0.1.2/foo.sh <<'EOF'
#!/usr/bin/env sh

echo Here be dragons...
EOF
  zip ext.zip ext-0.1.2/*
  rm -rf ext-0.1.2
}


test_patch_file() {
  EXTREPODIR=`pwd`

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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["file://${EXTREPODIR}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
  patches = ["//:patch_foo.sh"],
  patch_cmds = ["find . -name '*.sh' -exec sed -i.orig '1s|#!/usr/bin/env sh\$|/bin/sh\$|' {} +"],
)
EOF
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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext-0.1.2",
  urls=["file://${EXTREPODIR}/ext.zip"],
  build_file_content="exports_files([\"foo.sh\"])",
)
EOF
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'Here be' $foopath || fail "expected unpatched file"
  grep -q 'env' $foopath || fail "expected unpatched file"
}

test_patch_git() {
  EXTREPODIR=`pwd`
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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="file://${EXTREPODIR}/extgit/.git",
  tag="mytag",
  build_file_content="exports_files([\"foo.sh\"])",
  patches = ["//:patch_foo.sh"],
  patch_cmds = ["find . -name '*.sh' -exec sed -i.orig '1s|#!/usr/bin/env sh\$|/bin/sh\$|' {} +"],
)
EOF
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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="file://${EXTREPODIR}/extgit/.git",
  tag="mytag",
  build_file_content="exports_files([\"foo.sh\"])",
)
EOF
  bazel build :foo.sh
  foopath=`bazel info bazel-genfiles`/foo.sh
  grep -q 'Here be' $foopath || fail "expected unpatched file"
  grep -q 'env' $foopath || fail "expected unpatched file"
}

test_override_buildfile() {
  ## Verify that the BUILD file of an external repository can be overriden
  ## via the http_archive rule.
  EXTREPODIR=`pwd`

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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="withbuild",
  strip_prefix="withbuild",
  urls=["file://${EXTREPODIR}/withbuild.zip"],
  build_file="@//:ext.BUILD",
)
EOF
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
  ## Verify that the BUILD file of an external repository can be overriden
  ## via specified content in the http_archive rule.
  EXTREPODIR=`pwd`

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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="withbuild",
  strip_prefix="withbuild",
  urls=["file://${EXTREPODIR}/withbuild.zip"],
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
  ## Verify that the BUILD file of an external repository can be overriden
  ## via the git_repository rule.
  EXTREPODIR=`pwd`
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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="withbuild",
  remote="file://${EXTREPODIR}/withbuild/.git",
  tag="mytag",
  build_file="@//:ext.BUILD",
)
EOF
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
  ## Verify that the BUILD file of an external repository can be overriden
  ## via specified content in the git_repository rule.
  EXTREPODIR=`pwd`
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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="withbuild",
  remote="file://${EXTREPODIR}/withbuild/.git",
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
  ## Verify that the BUILD file of an external repository can be overriden
  ## via the http_archive rule.
  EXTREPODIR=`pwd`

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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="withbuild",
  strip_prefix="withbuild",
  urls=["file://${EXTREPODIR}/withbuild.zip"],
  build_file="@//:ext.BUILD",
)
EOF
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

run_suite "external patching tests"
