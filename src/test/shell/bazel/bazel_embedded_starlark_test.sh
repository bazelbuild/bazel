#!/usr/bin/env bash
#
# Copyright 2018 The Bazel Authors. All rights reserved.
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
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }


test_http_archive() {
  mkdir ext
  cat > ext/foo.sh <<'EOF'
#!/usr/bin/env sh

echo Here be dragons...
EOF
  zip ext.zip ext/*
  rm -rf ext

  EXTREPODIR=`pwd`
  mkdir main
  cd main
  cat > $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  urls=["file://${EXTREPODIR}/ext.zip"],
  strip_prefix="ext",
  build_file_content="exports_files([\"foo.sh\"])",
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
  bazel build :foo &> $TEST_log \
    || fail "Expected to build even with incompatible changes"
  bazel run :foo | grep -q dragons || fail "wrong output"
}

test_git_repository() {
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

  mkdir main
  cd main
  cat > $(setup_module_dot_bazel) <<EOF
git_repository = use_repo_rule("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
  name="ext",
  remote="file://${EXTREPODIR}/extgit/.git",
  tag="mytag",
  build_file_content="exports_files([\"foo.sh\"])",
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
  bazel build :foo &> $TEST_log \
    || fail "Expected to build even with incompatible changes"
  bazel run :foo | grep -q dragons || fail "wrong output"
}

run_suite "embedded Starlark"
