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
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

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


  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=../repo.bzl
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
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

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
  cat > BUILD <<'EOF'
genrule(
  name = "out",
  outs = ["out.txt"],
  srcs = ["@ext//:hello.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=../repo.bzl
  bazel build :out
  grep "CHANGED" `bazel info bazel-genfiles`/out.txt  \
       && fail "Unexpected content in out.txt" || :
  cd ..
  echo; cat repo.bzl; echo

  # Now add an additional commit to the upstream repository
  echo CHANGED > extgit/hello.txt
  (cd extgit
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'change hello.txt')


  # First verify that `bazel sync` sees the new commit (we don't record it).
  cd branchcheckout
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir
  bazel build :out
  grep "CHANGED" `bazel info bazel-genfiles`/out.txt  \
       || fail "sync did not update the external repository"
  bazel shutdown; sync; sleep 10
  cd ..
  echo

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

test_sync_follows_git_branch() {
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

  export GIT_CONFIG_NOSYSTEM=YES

  rm -f gitdir
  mkdir gitdir
  (cd gitdir && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  echo Hello World > gitdir/hello.txt
  (cd gitdir
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'initial commit')
  echo Hello Stable World > gitdir/hello.txt
  (cd gitdir
   git checkout -b stable
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'stable commit')

  # Follow the stable branch of the git repository
  mkdir followbranch
  cat > followbranch/WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="file://${EXTREPODIR}/gitdir/.git",
  branch="stable",
  build_file_content="exports_files([\"hello.txt\"])",
)
EOF
  cat > followbranch/BUILD <<'EOF'
genrule(
  name = "out",
  outs = ["out.txt"],
  srcs = ["@ext//:hello.txt"],
  cmd = "cp $< $@",
)
EOF
  (cd followbranch && bazel build :out \
       && cat `bazel info bazel-genfiles`/out.txt > "${TEST_log}")
  expect_log 'Hello Stable World'

  # New upstream commits on the branch followed
  echo CHANGED > gitdir/hello.txt
  (cd gitdir
   git checkout stable
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'stable commit')

  # Verify that sync followed by build gets the correct version
  (cd followbranch && bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir && bazel build :out \
       && cat `bazel info bazel-genfiles`/out.txt > "${TEST_log}")
  expect_log 'CHANGED'
  expect_not_log 'Hello Stable World'
}


test_sync_calls_all() {
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

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
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=../repo.bzl
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
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

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
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=../repo.bzl
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
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

  rm -rf fetchrepo
  mkdir fetchrepo
  cd fetchrepo
  cat > WORKSPACE <<'EOF'
load("//does/not:exist.bzl", "randomfunction")

radomfunction(name="foo")
EOF
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir > "${TEST_log}" 2>&1 && fail "Expected failure" || :
  expect_log '//does/not:exist.bzl'
}

test_sync_reporting() {
  # Verify that debug and error messages in starlark functions are reported.
  # Also verify that the fact that the repository is fetched is reported as well.
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

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
  bazel sync --curses=yes --experimental_ui_actions_shown=100 --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir > "${TEST_log}" 2>&1 && fail "expected failure" || :
  expect_log 'Fetching @broken'
  expect_log "DEBUG-message"
  expect_log "Failure-message"
}

test_indirect_call() {
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

  rm -rf fetchrepo
  mkdir fetchrepo
  cd fetchrepo
  touch BUILD
  cat > rule.bzl <<'EOF'
def _trivial_rule_impl(ctx):
  ctx.file("BUILD","genrule(name='hello', outs=['hello.txt'], cmd=' echo hello world > $@')")

trivial_rule = repository_rule(
  implementation = _trivial_rule_impl,
  attrs = {},
)
EOF
  cat > indirect.bzl <<'EOF'
def call(fn_name, **args):
  fn_name(**args)
EOF
  cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "trivial_rule")
load("//:indirect.bzl", "call")

call(trivial_rule, name="foo")
EOF
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=../repo.bzl
  bazel shutdown; sync; sleep 10

  cd ..
  echo; cat repo.bzl; echo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

ruleclass = "".join([entry["original_rule_class"] for entry in resolved if entry["original_attributes"]["name"]=="foo"])

genrule(
  name = "ruleclass",
  outs = ["ruleclass.txt"],
  cmd = "echo %s > $@" % (ruleclass,)
)
EOF
  bazel build //:ruleclass
  cat `bazel info bazel-genfiles`/ruleclass.txt > ${TEST_log}
  expect_log '//:rule.bzl%trivial_rule'
  expect_not_log 'fn_name'
}

test_resolved_file_reading() {
  # Verify that the option to read a resolved file instead of the WORKSPACE
  # file works as expected.
  EXTREPODIR=`pwd`
  export GIT_CONFIG_NOSYSTEM=YES
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

  mkdir extgit
  (cd extgit && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  echo Hello World > extgit/hello.txt
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
  branch="master",
  build_file_content="exports_files([\"hello.txt\"])",
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "out",
  outs = ["out.txt"],
  srcs = ["@ext//:hello.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=resolved.bzl
  echo; cat resolved.bzl; echo

  bazel clean --expunge
  echo 'Do not use any more' > WORKSPACE
  bazel build \
        --experimental_resolved_file_instead_of_workspace=`pwd`/resolved.bzl \
        :out || fail "Expected success with resolved file replacing WORKSPACE"
  rm WORKSPACE && touch WORKSPACE # bazel info needs a valid WORKSPACE
  grep 'Hello World' `bazel info bazel-genfiles`/out.txt \
      || fail "Did not find the expected output"
}

test_label_resolved_value() {
  # Verify that label arguments in a repository rule end up in the resolved
  # file in a parsable form.
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar
  mkdir ext
  echo Hello World > ext/file.txt
  zip ext.zip ext/*

  mkdir main
  cd main
  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${EXTREPODIR}/ext.zip"],
  build_file="@//:exit.BUILD",
)
EOF
  echo 'exports_files(["file.txt"])' > exit.BUILD
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  outs = ["local.txt"],
  srcs = ["@ext//:file.txt"],
  cmd = "cp $< $@",
)
EOF

  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=resolved.bzl
  rm WORKSPACE; touch WORKSPACE
  echo; cat resolved.bzl; echo

  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl \
        //:local || fail "Expected success"
  grep World `bazel info bazel-genfiles`/local.txt \
      || fail "target not built correctly"
}

test_resolved_file_not_remembered() {
  # Verify that the --experimental_resolved_file_instead_of_workspace option
  # does not leak into a subsequent sync
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

  export GIT_CONFIG_NOSYSTEM=YES

  rm -f gitdir
  mkdir gitdir
  (cd gitdir && git init \
       && git config user.email 'me@example.com' \
       && git config user.name 'E X Ample' )
  echo Hello Stable World > gitdir/hello.txt
  (cd gitdir
   git checkout -b stable
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'stable commit')

  # The project follows the stable branch of the git repository
  mkdir followbranch
  cat > followbranch/WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
new_git_repository(
  name="ext",
  remote="file://${EXTREPODIR}/gitdir/.git",
  branch="stable",
  build_file_content="exports_files([\"hello.txt\"])",
)
EOF
  cat > followbranch/BUILD <<'EOF'
genrule(
  name = "out",
  outs = ["out.txt"],
  srcs = ["@ext//:hello.txt"],
  cmd = "cp $< $@",
)
EOF
  (cd followbranch \
    && bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=resolved.bzl)
  # New upstream commits on the branch followed
  echo CHANGED > gitdir/hello.txt
  (cd gitdir
   git checkout stable
   git add .
   git commit --author="A U Thor <author@example.com>" -m 'stable commit')

  cd followbranch
  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl :out
  cat `bazel info bazel-genfiles`/out.txt > "${TEST_log}"
  expect_log 'Hello Stable World'
  expect_not_log 'CHANGED'
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=resolved.bzl
  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl :out
  cat `bazel info bazel-genfiles`/out.txt > "${TEST_log}"
  expect_log 'CHANGED'
  expect_not_log 'Hello Stable World'
}

create_sample_repository() {
  # Create, in the current direcotry, a repository that creates an external
  # repository `foo` containing
  # - file with fixed data, generated by ctx.file,
  # - a BUILD file linked from the main repository
  # - a symlink to ., and
  # - danling absolute and reproducible symlink.
  touch BUILD
  cat > rule.bzl <<'EOF'
def _trivial_rule_impl(ctx):
  ctx.symlink(ctx.attr.build_file, "BUILD")
  ctx.file("data.txt", "some data")
  ctx.execute(["ln", "-s", ".", "self_link"])
  ctx.execute(["ln", "-s", "/does/not/exist", "dangling"])

trivial_rule = repository_rule(
  implementation = _trivial_rule_impl,
  attrs = { "build_file" : attr.label() },
)
EOF
  echo '# fixed contents' > BUILD.remote
  cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "trivial_rule")

trivial_rule(name="foo", build_file="@//:BUILD.remote")
EOF
}

test_hash_included_and_reproducible() {
  # Verify that a hash of the output directory is included, that
  # the hash is invariant under
  # - change of the working directory, and
  # - and current time.
  EXTREPODIR=`pwd`
  tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar

  rm -rf fetchrepoA
  mkdir fetchrepoA
  cd fetchrepoA
  create_sample_repository
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=../repo.bzl
  bazel shutdown; sync; sleep 10

  cd ..
  echo; cat repo.bzl; echo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")
hashes = [entry["repositories"][0]["output_tree_hash"]
         for entry in resolved if entry["original_attributes"]["name"]=="foo"]

[genrule(
  name="hash",
  outs=["hash.txt"],
  cmd="echo '%s' > $@" % (hash,),
) for hash in hashes]
EOF
  bazel build //:hash
  cp `bazel info bazel-genfiles`/hash.txt hashA.txt
  cat hashA.txt > "${TEST_log}"
  [ `cat hashA.txt | wc -c` -gt 2 ] \
      || fail "A hash of reasonable length expected"
  bazel clean --expunge
  rm repo.bzl


  rm -rf fetchrepoB
  mkdir fetchrepoB
  cd fetchrepoB
  create_sample_repository
  bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=../repo.bzl
  bazel shutdown; sync; sleep 10

  cd ..
  echo; cat repo.bzl; echo
  bazel build //:hash
  cp `bazel info bazel-genfiles`/hash.txt hashB.txt
  cat hashB.txt > "${TEST_log}"
  diff hashA.txt hashB.txt || fail "Expected hash to be reproducible"
}

test_non_reproducibility_detected() {
    EXTREPODIR=`pwd`
    tar xvf ${TEST_SRCDIR}/jdk_WORKSPACE_files/archives.tar
    # Verify that a non-reproducible rule is detected by hash verification
    mkdir repo
    cd repo
    touch BUILD
    cat > rule.bzl <<'EOF'
def _time_rule_impl(ctx):
  ctx.execute(["bash", "-c", "date +%s > timestamp"])

time_rule = repository_rule(
  implementation = _time_rule_impl,
  attrs = {},
)
EOF
    cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "time_rule")

time_rule(name="timestamprepo")
EOF

    bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_resolved_file=resolved.bzl
    sync; sleep 10
    bazel sync --distdir=${EXTREPODIR}/jdk_WORKSPACE/distdir --experimental_repository_hash_file=`pwd`/resolved.bzl \
          --experimental_verify_repository_rules='//:rule.bzl%time_rule' \
          > "${TEST_log}" 2>&1 && fail "expected failure" || :
    expect_log "timestamprepo.*hash"
}

run_suite "workspace_resolved_test tests"
