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

mock_rules_java_to_avoid_downloading

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
        > "${TEST_log}" 2>&1 || fail "Expected success"
  inplace-sed -e "s?$(pwd)?PWD?g" "$TEST_log"
  bazel shutdown

  # We expect the additional argument to be reported to the user...
  expect_log 'extra_arg.*foobar'
  # ...as well as the location of the rule instantiation and definition.
  expect_log 'Repository ext instantiated at:'
  expect_log '  PWD/WORKSPACE:2:'
  expect_log 'Repository rule trivial_rule defined at:'
  expect_log '  PWD/rule.bzl:5:'

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
  bazel shutdown

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
  cat > BUILD <<'EOF'
genrule(
  name = "out",
  outs = ["out.txt"],
  srcs = ["@ext//:hello.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel sync --experimental_repository_resolved_file=../repo.bzl
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
  bazel sync
  bazel build :out
  grep "CHANGED" `bazel info bazel-genfiles`/out.txt  \
       || fail "sync did not update the external repository"
  bazel shutdown
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
  (cd followbranch && bazel sync && bazel build :out \
       && cat `bazel info bazel-genfiles`/out.txt > "${TEST_log}")
  expect_log 'CHANGED'
  expect_not_log 'Hello Stable World'
}

test_http_return_value() {
  EXTREPODIR=`pwd`

  mkdir -p a
  touch a/WORKSPACE
  touch a/BUILD
  touch a/f.txt

  zip a.zip a/*
  expected_sha256="$(sha256sum "${EXTREPODIR}/a.zip" | head -c 64)"
  rm -rf a

  # http_archive rule doesn't specify the sha256 attribute
  mkdir -p main
  cat > main/WORKSPACE <<EOF
workspace(name = "main")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="a",
  strip_prefix="a",
  urls=["file://${EXTREPODIR}/a.zip"],
)
EOF
  touch main/BUILD

  cd main
  bazel sync \
      --experimental_repository_resolved_file=../repo.bzl

  grep ${expected_sha256} ../repo.bzl || fail "didn't return commit"
}

test_sync_calls_all() {
  EXTREPODIR=`pwd`

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
  bazel shutdown

  cd ..
  echo; cat repo.bzl; echo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

names = [entry["original_attributes"]["name"]
         for entry in resolved
         if "native" not in entry]

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
  bazel shutdown

  cd ..
  echo; cat repo.bzl; echo
  touch WORKSPACE
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

names = [entry["original_attributes"]["name"]
         for entry in resolved
         if "native" not in entry]
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

test_sync_reporting() {
  # Verify that debug and error messages in starlark functions are reported.
  # Also verify that the fact that the repository is fetched is reported as well.
  EXTREPODIR=`pwd`

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
  cat >> $(create_workspace_with_default_repos WORKSPACE) <<'EOF'
load("//:rule.bzl", "broken_rule")

broken_rule(name = "broken")
EOF
  bazel sync --curses=yes --experimental_ui_actions_shown=100 > "${TEST_log}" 2>&1 && fail "expected failure" || :
  expect_log 'Fetching repository @broken'
  expect_log "DEBUG-message"
  expect_log "Failure-message"
}

test_indirect_call() {
  EXTREPODIR=`pwd`

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
  bazel sync --experimental_repository_resolved_file=../repo.bzl
  bazel shutdown

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
  bazel sync --experimental_repository_resolved_file=resolved.bzl
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

  bazel sync --experimental_repository_resolved_file=resolved.bzl
  rm WORKSPACE
  touch WORKSPACE
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
    && bazel sync --experimental_repository_resolved_file=resolved.bzl)
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
  bazel sync --experimental_repository_resolved_file=resolved.bzl
  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl :out
  cat `bazel info bazel-genfiles`/out.txt > "${TEST_log}"
  expect_log 'CHANGED'
  expect_not_log 'Hello Stable World'
}

create_sample_repository() {
  # Create, in the current directory, a repository that creates an external
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

  rm -rf fetchrepoA
  mkdir fetchrepoA
  cd fetchrepoA
  create_sample_repository
  bazel sync --experimental_repository_resolved_file=../repo.bzl
  bazel shutdown

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
  bazel sync --experimental_repository_resolved_file=../repo.bzl
  bazel shutdown

  cd ..
  echo; cat repo.bzl; echo
  bazel build //:hash
  cp `bazel info bazel-genfiles`/hash.txt hashB.txt
  cat hashB.txt > "${TEST_log}"
  diff hashA.txt hashB.txt || fail "Expected hash to be reproducible"
}

test_non_reproducibility_detected() {
    EXTREPODIR=`pwd`
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

    bazel sync --experimental_repository_resolved_file=resolved.bzl
    cat resolved.bzl > /dev/null || fail "resolved.bzl should exist"
    bazel sync --experimental_repository_hash_file=`pwd`/resolved.bzl \
          --experimental_verify_repository_rules='//:rule.bzl%time_rule' \
          > "${TEST_log}" 2>&1 && fail "expected failure" || :
    expect_log "timestamprepo.*hash"
}

test_chain_resolved() {
  # Verify that a cahin of dependencies in external repositories is reflected
  # in the resolved file in such a way, that the resolved file can be used.
  EXTREPODIR=`pwd`

  mkdir rulerepo
  cat > rulerepo/rule.bzl <<'EOF'
def _rule_impl(ctx):
  ctx.file("data.txt", "Hello World")
  ctx.file("BUILD", "exports_files(['data.txt'])")

trivial_rule = repository_rule(
  implementation = _rule_impl,
  attrs = {},
)
EOF
  touch rulerepo/BUILD
  zip rule.zip rulerepo/*
  rm -rf rulerepo

  cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="rulerepo",
  strip_prefix="rulerepo",
  urls=["file://${EXTREPODIR}/rule.zip"],
)
load("@rulerepo//:rule.bzl", "trivial_rule")
trivial_rule(name="a")
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs =  ["@a//:data.txt"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel sync --experimental_repository_resolved_file=resolved.bzl
  bazel clean --expunge
  echo; cat resolved.bzl; echo

  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl \
        //:local || fail "Expected success"
}

test_usage_order_respected() {
   # Verify that if one rules uses a file from another (without any load
   # statement between), then still the resolved file is such that it can
   # be used as a workspace replacement.
   EXTREPODIR=`pwd`

   mkdir datarepo
   echo 'Pure data' > datarepo/data.txt
   zip datarepo.zip datarepo/*
   rm -rf datarepo

   mkdir metadatarepo
   echo 'exports_files(["data.txt"])' > metadatarepo/datarepo.BUILD
   touch metadatarepo/BUILD
   zip metadatarepo.zip metadatarepo/*
   rm -rf metadatarepo

   mkdir main
   cd main
   cat > WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="datarepo",
  strip_prefix="datarepo",
  urls=["file://${EXTREPODIR}/datarepo.zip"],
  build_file="@metadatarepo//:datarepo.BUILD",
)
http_archive(
  name="metadatarepo",
  strip_prefix="metadatarepo",
  urls=["file://${EXTREPODIR}/metadatarepo.zip"],
)
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "local",
  srcs =  ["@datarepo//:data.txt"],
  outs = ["local.txt"],
  cmd = "cp $< $@",
)
EOF
  bazel sync \
     --experimental_repository_resolved_file=resolved.bzl
  bazel clean --expunge
  echo; cat resolved.bzl; echo


  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl \
        //:local || fail "Expected success"
}

test_order_reproducible() {
  # Verify that the order of repositories in the resolved file is reproducible
  # and does not depend on the parameters or timing of the actual rules.
  EXTREPODIR=`pwd`

  mkdir main
  cd main

  cat > rule.bzl <<'EOF'
def _impl(ctx):
  ctx.execute(["/bin/sh", "-c", "sleep %s" % (ctx.attr.sleep,)])
  ctx.file("data", "some test data")
  ctx.file("BUILD", "exports_files(['data'])")

sleep_rule = repository_rule(
  implementation = _impl,
  attrs = {"sleep": attr.int()},
)
EOF
  cat > BUILD <<'EOF'
load("//:repo.bzl", "resolved")

genrule(
  name = "order",
  outs = ["order.txt"],
  cmd = ("echo '%s' > $@" %
    ([entry["original_attributes"]["name"] for entry in resolved],)),
)
EOF
  cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "sleep_rule")

sleep_rule(name="a", sleep=1)
sleep_rule(name="c", sleep=3)
sleep_rule(name="b", sleep=5)
EOF
  bazel sync \
        --experimental_repository_resolved_file=repo.bzl
  bazel build //:order
  cp `bazel info bazel-genfiles`/order.txt order-first.txt
  bazel clean --expunge

  cat > WORKSPACE <<'EOF'
load("//:rule.bzl", "sleep_rule")

sleep_rule(name="a", sleep=5)
sleep_rule(name="c", sleep=3)
sleep_rule(name="b", sleep=1)
EOF
  bazel sync \
        --experimental_repository_resolved_file=repo.bzl
  bazel build //:order
  cp `bazel info bazel-genfiles`/order.txt order-second.txt

  echo; cat order-first.txt; echo; cat order-second.txt; echo

  diff order-first.txt order-second.txt \
      || fail "expected order to be reproducible"
}

test_non_starlarkrepo() {
  # Verify that entries in the WORKSPACE that are not starlark repositoires
  # are correctly reported in the resolved file.
  EXTREPODIR=`pwd`

  mkdir local
  touch local/WORKSPACE
  echo Hello World > local/data.txt
  echo 'exports_files(["data.txt"])' > local/BUILD

  mkdir newlocal
  echo Pure data > newlocal/data.txt

  mkdir main
  cd main
  mkdir target_to_be_bound
  echo More data > target_to_be_bound/data.txt
  echo 'exports_files(["data.txt"])' > target_to_be_bound/BUILD
  cat > WORKSPACE <<'EOF'
local_repository(name="thisislocal", path="../local")
new_local_repository(name="newlocal", path="../newlocal",
                     build_file_content='exports_files(["data.txt"])')
bind(name="bound", actual="//target_to_be_bound:data.txt")
EOF
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@thisislocal//:data.txt", "@newlocal//:data.txt",
          "//external:bound"],
  outs = ["it.txt"],
  cmd = "cat $(SRCS) > $@",
)
EOF

  bazel build //:it || fail "Expected success"

  bazel sync --experimental_repository_resolved_file=resolved.bzl
  echo > WORKSPACE # remove workspace, only work from the resolved file
  bazel clean --expunge
  echo; cat resolved.bzl; echo
  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl \
        //:it || fail "Expected success"
}

test_hidden_symbols() {
  # Verify that the resolved file can be used for building, even if it
  # legitimately contains a private symbol
  mkdir main
  cd main
  cat > BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["@foo//:data.txt"],
  outs = ["it.txt"],
  cmd = "cp $< $@",
)
EOF

  cat > repo.bzl <<'EOF'
_THE_DATA="42"

def _data_impl(ctx):
  ctx.file("BUILD", "exports_files(['data.txt'])")
  ctx.file("data.txt", ctx.attr.data)

_repo = repository_rule(
  implementation = _data_impl,
  attrs = { "data" : attr.string() },
)

def data_repo(name):
  _repo(name=name, data=_THE_DATA)

EOF
  cat > WORKSPACE <<'EOF'
load("//:repo.bzl", "data_repo")

data_repo("foo")
EOF

  bazel build --experimental_repository_resolved_file=resolved.bzl //:it
  echo > WORKSPACE # remove workspace, only work from the resolved file
  bazel clean --expunge
  echo; cat resolved.bzl; echo

  bazel build --experimental_resolved_file_instead_of_workspace=resolved.bzl \
        //:it || fail "Expected success"
}

test_toolchain_recorded() {
  # Verify that the registration of toolchains and execution platforms is
  # recorded in the resolved file
  EXTREPODIR=`pwd`

  mkdir ext
  touch ext/BUILD
  cat > ext/toolchains.bzl <<'EOF'
def ext_toolchains():
  native.register_toolchains("@ext//:toolchain")
  native.register_execution_platforms("@ext//:platform")
EOF
  tar cvf ext.tar ext
  rm -rf ext

  mkdir main
  cd main
  cat >> WORKSPACE <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="ext",
  strip_prefix="ext",
  urls=["file://${EXTREPODIR}/ext.tar"],
)
load("@ext//:toolchains.bzl", "ext_toolchains")
ext_toolchains()
EOF
  touch BUILD
  bazel sync --experimental_repository_resolved_file=resolved.bzl
  echo; cat resolved.bzl; echo

  grep 'register_toolchains.*ext//:toolchain' resolved.bzl \
      || fail "tool chain not registered in resolved file"
  grep 'register_execution_platforms.*ext//:platform' resolved.bzl \
      || fail "execution platform not registered in resolved file"
}

test_local_config_platform_recorded() {
  EXTREPODIR=`pwd`

  # Verify that the auto-generated local_config_platform repo is
  # recorded in the resolved file
  mkdir main
  cd main
  # Clear out the WORKSPACE.
  cat >> WORKSPACE <<EOF
EOF
  touch BUILD
  bazel sync --experimental_repository_resolved_file=resolved.bzl
  echo; cat resolved.bzl; echo

  grep 'local_config_platform' resolved.bzl \
      || fail "local_config_platform in resolved file"
}

test_definition_location_recorded() {
  # Verify that for Starlark repositories the location of the definition
  # is recorded in the resolved file.
  EXTREPODIR=`pwd`

  mkdir ext
  touch ext/BUILD

  tar cvf ext.tar ext
  rm -rf ext

  mkdir main
  cd main
  touch BUILD
  mkdir -p first/path
  cat > first/path/foo.bzl <<'EOF'
load("//:another/directory/bar.bzl", "bar")

def foo():
  bar()
EOF
  mkdir -p another/directory
  cat > another/directory/bar.bzl <<EOF
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def bar():
  http_archive(
    name = "ext",
    url = "file://${EXTREPODIR}/ext.tar",
  )
EOF
  cat > WORKSPACE <<'EOF'
load("//:first/path/foo.bzl", "foo")

foo()
EOF

  bazel sync --experimental_repository_resolved_file=resolved.bzl

  echo; cat resolved.bzl; echo

  cat > BUILD <<'EOF'
load("//:finddef.bzl", "finddef")

genrule(
  name = "ext_def",
  outs = ["ext_def.txt"],
  cmd = "echo '%s' > $@" % (finddef("ext"),),
)
EOF
  cat > finddef.bzl <<'EOF'
load("//:resolved.bzl", "resolved")

def finddef(name):
  for repo in resolved:
    if repo["original_attributes"]["name"] == name:
      return repo["definition_information"]
EOF

  bazel build //:ext_def

  cat `bazel info bazel-genfiles`/ext_def.txt > "${TEST_log}"
  inplace-sed -e "s?$(pwd)/?PWD/?g" -e "s?$TEST_TMPDIR/?TEST_TMPDIR/?g" "${TEST_log}"
  expect_log "Repository ext instantiated at:"
  expect_log "  PWD/WORKSPACE:3"
  expect_log "  PWD/first/path/foo.bzl:4"
  expect_log "  PWD/another/directory/bar.bzl:4"
  expect_log "Repository rule http_archive defined at:"
  expect_log "  TEST_TMPDIR/.*/external/bazel_tools/tools/build_defs/repo/http.bzl:"
}

# Regression test for #11040.
#
# Test that a canonical repo warning is generated for explicitly specified
# attributes whose values differ, and that it is never generated for implicitly
# created attributes (in particular, the generator_* attributes).
test_canonical_warning() {
  EXTREPODIR=`pwd`

  mkdir main
  touch main/BUILD
  cat > main/reporule.bzl <<EOF
def _impl(repository_ctx):
    repository_ctx.file("a.txt", "A")
    repository_ctx.file("BUILD.bazel", "exports_files(['a.txt'])")
    # Don't include "name", test that we warn about it below.
    return {"myattr": "bar"}

reporule = repository_rule(
    implementation = _impl,
    attrs = {
        "myattr": attr.string()
    })

# We need to use a macro for the generator_* attributes to be defined.
def instantiate_reporule(name, **kwargs):
    reporule(name=name, **kwargs)
EOF
  cat > main/WORKSPACE <<EOF
workspace(name = "main")
load("//:reporule.bzl", "instantiate_reporule")
instantiate_reporule(
  name = "myrepo",
  myattr = "foo"
)
EOF

  cd main
  # We should get a warning for "myattr" having a changed value and for "name"
  # being dropped, but not for the generator_* attributes.
  bazel sync >/dev/null 2>$TEST_log
  bazel clean --expunge
  expect_log "Rule 'myrepo' indicated that a canonical reproducible form \
can be obtained by modifying arguments myattr = \"bar\" and dropping \
\[.*\"name\".*\]"
  expect_not_log "Rule 'myrepo' indicated .*generator"
}

run_suite "workspace_resolved_test tests"
