#!/bin/bash
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
# Test that the embedded skylark code is compliant with --all_incompatible_changes.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }


test_pkg_tar() {
  rm -rf main
  mkdir main
  cd main
  touch WORKSPACE
  echo Hello World > foo.txt
  echo Hello World, again > bar.txt
  cat > BUILD <<'EOF'
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "data",
    srcs = glob(["*.txt"]),
)
EOF
  bazel build --all_incompatible_changes ... \
    || fail "Expect success, even with all upcoming Skylark changes"
  grep -q 'Hello World' `bazel info bazel-bin`/data.tar \
    || fail "Output not generated correctly"
}

test_pkg_tar_quoting() {
  # Verify that pkg_tar can handle file names that are allowed as lablels
  # but contain characters that could mess up options.
  rm -rf main out
  mkdir main
  cd main
  touch WORKSPACE
  mkdir data
  echo 'with equal' > data/'foo=bar'
  echo 'like an option' > data/--foo
  cat > BUILD <<'EOF'
load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
  name = "fancy",
  srcs = glob(["data/**/*"]),
  symlinks = {"link_with_colons" : "some:dangling:link"},
)
EOF
  bazel build --all_incompatible_changes :fancy || fail "Expected success"
  mkdir ../out
  tar -C ../out -x -v -f `bazel info bazel-bin`/fancy.tar

  grep equal ../out/foo=bar || fail "file with equal sign not packed correctly"
  grep option ../out/--foo || fail "file with double minus not packed correctly"
  readlink ../out/link_with_colons | grep -- 'some:dangling:link' \
      || fail "symlink not packed"
}

test_pkg_tar_strip_directory() {
  # Verify that pkg_tar's strip_prefix permits stripping off the name of
  # directory output targets.
  rm -rf main out
  mkdir main
  cd main
  touch WORKSPACE
  cat > BUILD <<'EOF'
load(":apple.bzl", "create_banana_directory")

create_banana_directory(
    name = "banana_directory",
)

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "banana_tarball",
    srcs = [":banana_directory"],
    strip_prefix = "banana",
)
EOF
  cat > apple.bzl <<'EOF'
def _create_banana_directory_impl(ctx):
    out = ctx.actions.declare_directory("banana")
    ctx.actions.run(
        executable = "bash",
        arguments = ["-c", "mkdir -p %s/pear && touch %s/pear/grape" % (out.path, out.path)],
        outputs = [out],
    )
    return [
        DefaultInfo(
            files = depset([out]),
        ),
    ]

create_banana_directory = rule(
    implementation = _create_banana_directory_impl,
)
EOF
  bazel build --all_incompatible_changes :banana_tarball || fail "Expected success"
  mkdir ../out
  tar -C ../out -x -v -f `bazel info bazel-bin`/banana_tarball.tar

  test -f ../out/pear/grape || fail "expected file to be present"
}

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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
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
  bazel build --all_incompatible_changes :foo \
    || fail "Expected to build even with incompatible changes"
  bazel run :foo | grep -q dragons || fail "wrong output"
}

test_new_git_repository() {
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
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
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
  bazel build --all_incompatible_changes :foo \
    || fail "Expected to build even with incompatible changes"
  bazel run :foo | grep -q dragons || fail "wrong output"
}

run_suite "embedded skylark"
