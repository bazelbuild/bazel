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

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

test_result_recorded() {
  mkdir result_recorded && cd result_recorded
  rm -rf fetchrepo
  mkdir fetchrepo
  cd fetchrepo
  cat > rule.bzl <<'EOF'
def _rule_impl(ctx):
  ctx.symlink(ctx.attr.build_file, "BUILD")
  return {"build_file": ctx.attr.build_file, "extra_arg": "foobar"}

trivial_rule = repository_rule(
  implementation = _rule_impl,
  attrs = { "build_file" : attr.label() },
)

EOF
  cat > ext.BUILD <<'EOF'
genrule(
  name = "foo",
  outs = ["foo.txt"],
  cmd = "echo bar > $@",
)
EOF
  touch BUILD
  cat  > WORKSPACE <<'EOF'
load("//:rule.bzl", "trivial_rule")
trivial_rule(
  name = "ext",
  build_file = "//:ext.BUILD",
)
EOF

  bazel clean --expunge
  bazel build --experimental_repository_resolved_file=../repo.bzl @ext//... \
      || fail "Expected success"
  # some of the file systems on our test machines are really slow to
  # notice the creation of a file---even after the call to sync(1).
  bazel shutdown; sync; sleep 10

  # Verify that bazel can read the generated repo.bzl file and that it contains
  # the expected information
  cd ..
  echo; cat repo.bzl; echo; echo
  mkdir analysisrepo
  mv repo.bzl analysisrepo
  cd analysisrepo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

[ genrule(
    name = "out",
    outs = ["out.txt"],
    cmd = "echo %s > $@" % entry["repositories"][0]["attributes"]["extra_arg"],
  ) for entry in resolved if entry["original_rule_class"] == "//:rule.bzl%trivial_rule"
]

[ genrule(
    name = "origcount",
    outs = ["origcount.txt"],
    cmd = "echo %s > $@" % len(entry["original_attributes"])
  ) for entry in resolved if entry["original_rule_class"] == "//:rule.bzl%trivial_rule"
]
EOF
  bazel build :out :origcount || fail "Expected success"
  grep "foobar" `bazel info bazel-genfiles`/out.txt \
      || fail "Did not find the expected value"
  [ $(cat `bazel info bazel-genfiles`/origcount.txt) -eq 2 ] \
      || fail "Not the correct number of original attributes"
}

test_git_return_value() {
  EXTREPODIR=`pwd`
  export GIT_CONFIG_NOSYSTEM=YES

  mkdir extgit
  (cd extgit && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  echo Hello World > extgit/hello.txt
  (cd extgit
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'initial commit'
   git tag mytag)

  # Check out the external git repository at the given tag, and record
  # the return value of the git rule.
  mkdir tagcheckout
  cd tagcheckout
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="file://${EXTREPODIR}/extgit/.git",
  tag="mytag",
  build_file_content="exports_files([\"hello.txt\"])",
)
EOF
  bazel sync --experimental_repository_resolved_file=../repo.bzl
  # some of the file systems on our test machines are really slow to
  # notice the creation of a file---even after the call to sync(1).
  bazel shutdown; sync; sleep 10

  cd ..
  echo; cat repo.bzl; echo

  # Now add an additional commit to the upstream repository and
  # force update the tag
  echo CHANGED > extgit/hello.txt
  (cd extgit
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'change hello.txt'
   git tag -f mytag)

  # Verify that the recorded resolved information is what we expect. In
  # particular, verify that we don't get the new upstream commit.
  mkdir analysisrepo
  cd analysisrepo
  cp ../repo.bzl .
  cat > workspace.bzl <<'EOF'
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("//:repo.bzl", "resolved")

def repo():
    for entry in resolved:
        if entry["original_attributes"]["name"] == "ext":
            new_git_repository(**(entry["repositories"][0]["attributes"]))
EOF
  cat > WORKSPACE <<'EOF'
load("//:workspace.bzl", "repo")
repo()
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "out",
  outs = ["out.txt"],
  srcs = ["@ext//:hello.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build //:out
  grep "Hello World" `bazel info bazel-genfiles`/out.txt \
       || fail "ext not taken at the right commit"
  grep "CHANGED" `bazel info bazel-genfiles`/out.txt  \
       && fail "not taking the frozen commit" || :
}

test_git_follow_branch() {
  EXTREPODIR=`pwd`
  export GIT_CONFIG_NOSYSTEM=YES

  mkdir extgit
  (cd extgit && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  echo Hello World > extgit/hello.txt
  (cd extgit
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'initial commit')
  # Check out the external git repository at the given branch, and record
  # the return value of the git rule.
  mkdir branchcheckout
  cd branchcheckout
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="file://${EXTREPODIR}/extgit/.git",
  branch="master",
  build_file_content="exports_files([\"hello.txt\"])",
)
EOF
  bazel sync --experimental_repository_resolved_file=../repo.bzl
  # some of the file systems on our test machines are really slow to
  # notice the creation of a file---even after the call to sync(1).
  bazel shutdown; sync; sleep 10

  cd ..
  echo; cat repo.bzl; echo

  # Now add an additional commit to the upstream repository
  echo CHANGED > extgit/hello.txt
  (cd extgit
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'change hello.txt')

  # Verify that the recorded resolved information is what we expect. In
  # particular, verify that we don't get the new upstream commit.
  mkdir analysisrepo
  cd analysisrepo
  cp ../repo.bzl .
  cat > workspace.bzl <<'EOF'
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("//:repo.bzl", "resolved")

def repo():
    for entry in resolved:
        if entry["original_attributes"]["name"] == "ext":
            new_git_repository(**(entry["repositories"][0]["attributes"]))
EOF
  cat > WORKSPACE <<'EOF'
load("//:workspace.bzl", "repo")
repo()
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "out",
  outs = ["out.txt"],
  srcs = ["@ext//:hello.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel build //:out
  grep "Hello World" `bazel info bazel-genfiles`/out.txt \
       || fail "ext not taken at the right commit"
  grep "CHANGED" `bazel info bazel-genfiles`/out.txt  \
       && fail "not taking the frozen commit" || :
}


test_sync_calls_all() {
  mkdir sync_calls_all && cd sync_calls_all
  rm -rf fetchrepo
  mkdir fetchrepo
  rm -f repo.bzl
  cd fetchrepo
  cat > rule.bzl <<'EOF'
def _rule_impl(ctx):
  ctx.file("foo.bzl", """
it = "foo"
other = "bar"
""")
  ctx.file("BUILD", "")
  return {"comment" : ctx.attr.comment }

trivial_rule = repository_rule(
  implementation = _rule_impl,
  attrs = { "comment" : attr.string() },
)
EOF
  touch BUILD
  cat  > WORKSPACE <<'EOF'
load("//:rule.bzl", "trivial_rule")
trivial_rule(name = "a", comment = "bootstrap")
load("@a//:foo.bzl", "it")
trivial_rule(name = "b", comment = it)
trivial_rule(name = "c", comment = it)
load("@c//:foo.bzl", "other")
trivial_rule(name = "d", comment = other)
EOF

  bazel clean --expunge
  bazel sync --experimental_repository_resolved_file=../repo.bzl
  # some of the file systems on our test machines are really slow to
  # notice the creation of a file---even after the call to sync(1).
  bazel shutdown; sync; sleep 10

  cd ..
  echo; cat repo.bzl; echo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

names = [entry["original_attributes"]["name"] for entry in resolved]

[
  genrule(
   name = name,
   outs = [ "%s.txt" % (name,) ],
   cmd = "echo %s > $@" % (name,),
  ) for name in names
]
EOF
  bazel build :a :b :c :d || fail "Expected all 4 repositories to be present"
}

test_sync_call_invalidates() {
  mkdir sync_call_invalidates && cd sync_call_invalidates
  rm -rf fetchrepo
  mkdir fetchrepo
  rm -f repo.bzl
  touch BUILD
  cat > rule.bzl <<'EOF'
def _rule_impl(ctx):
  ctx.file("BUILD", """
genrule(
  name = "it",
  outs = ["it.txt"],
  cmd = "echo hello world > $@",
)
""")
  ctx.file("WORKSPACE", "")

trivial_rule = repository_rule(
  implementation = _rule_impl,
  attrs = {},
)
EOF
  cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "trivial_rule")

trivial_rule(name = "a")
trivial_rule(name = "b")
EOF

  bazel build @a//... @b//...
  echo; echo sync run; echo
  bazel sync --experimental_repository_resolved_file=../repo.bzl
  # some of the file systems on our test machines are really slow to
  # notice the creation of a file---even after the call to sync(1).
  bazel shutdown; sync; sleep 10

  cd ..
  echo; cat repo.bzl; echo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

names = [entry["original_attributes"]["name"] for entry in resolved]

[
  genrule(
   name = name,
   outs = [ "%s.txt" % (name,) ],
   cmd = "echo %s > $@" % (name,),
  ) for name in names
]
EOF
  bazel build :a :b || fail "Expected both repositories to be present"
}

test_sync_load_errors_reported() {
  rm -rf fetchrepo
  mkdir fetchrepo
  cd fetchrepo
  cat > WORKSPACE <<'EOF'
load("//does/not:exist.bzl", "randomfunction")

radomfunction(name="foo")
EOF
  bazel sync > "${TEST_log}" 2>&1 && fail "Expected failure" || :
  expect_log '//does/not:exist.bzl'
}

test_sync_debug_and_errors_printed() {
  rm -rf fetchrepo
  mkdir fetchrepo
  cd fetchrepo
  cat > rule.bzl <<'EOF'
def _broken_rule_impl(ctx):
  print("DEBUG-message")
  fail("Failure-message")

broken_rule = repository_rule(
  implementation = _broken_rule_impl,
  attrs = {},
)
EOF
  touch BUILD
  cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "broken_rule")

broken_rule(name = "broken")
EOF
  bazel sync > "${TEST_log}" 2>&1 && fail "expected failure" || :
  expect_log "DEBUG-message"
  expect_log "Failure-message"
}

run_suite "workspace_resolved_test tests"
