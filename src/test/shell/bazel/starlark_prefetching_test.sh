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
# Verify that declared arguments of a repository rule are present before
# the first execution attempt of the rule is done.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

test_label_arg() {
  # Verify that a repository rule does not get restarted, if accessing
  # one of its label arguments as file.
  WRKDIR=`pwd`
  rm -rf repo
  rm -rf log
  mkdir repo
  cd repo
  touch BUILD
  cat > rule.bzl <<EOF
def _impl(ctx):
  # Access the build file late
  ctx.execute(["/bin/sh", "-c", "date +%s >> ${WRKDIR}/log"])
  ctx.file("REPO.bazel")
  ctx.symlink(ctx.attr.build_file, "BUILD")

myrule=repository_rule(implementation=_impl,
 attrs={
   "build_file" : attr.label(),
 })
EOF
  cat > ext.BUILD <<'EOF'
genrule(
  name="foo",
  outs=["foo.txt"],
  cmd = "echo foo > $@",
)
EOF
  cat > MODULE.bazel <<'EOF'
myrule = use_repo_rule("//:rule.bzl", "myrule")
myrule(name="ext", build_file="//:ext.BUILD")
EOF
  setup_module_dot_bazel "MODULE.bazel"
  bazel build @ext//:foo || fail "expected success"
  [ `cat "${WRKDIR}/log" | wc -l` -eq 1 ] \
      || fail "did not find precisely one invocation of the action"
}

test_unused_invalid_label_arg() {
  # Verify that we preserve the behavior of allowing to pass labels that
  # do referring to an non-existing path, if they are never used.
  WRKDIR=`pwd`
  rm -rf repo
  mkdir repo
  cd repo
  touch BUILD
  cat > rule.bzl <<'EOF'
def _impl(ctx):
  ctx.file("REPO.bazel")
  ctx.file("BUILD",
           "genrule(name=\"foo\", outs=[\"foo.txt\"], cmd = \"echo foo > $@\")")

myrule=repository_rule(implementation=_impl,
 attrs={
   "unused" : attr.label(),
 })
EOF
  cat > MODULE.bazel <<'EOF'
myrule = use_repo_rule("//:rule.bzl", "myrule")
myrule(name="ext", unused="//does/not/exist:file")
EOF
  setup_module_dot_bazel "MODULE.bazel"
  bazel build @ext//:foo || fail "expected success"
}


test_label_list_arg() {
  # Verify that a repository rule does not get restarted, if accessing
  # the entries of a label list as files.
  WRKDIR=`pwd`
  rm -rf repo
  rm -rf log
  mkdir repo
  cd repo
  touch BUILD
  cat > rule.bzl <<EOF
def _impl(ctx):
  ctx.execute(["/bin/sh", "-c", "date +%s >> ${WRKDIR}/log"])
  ctx.file("REPO.bazel")
  ctx.file("BUILD",  """
genrule(
  name="foo",
  srcs= ["src.txt"],
  outs=["foo.txt"],
  cmd = "cp \$< \$@",
)
""")
  for f in ctx.attr.data:
    ctx.execute(["/bin/sh", "-c", "cat %s >> src.txt" % ctx.path(f)])

myrule=repository_rule(implementation=_impl,
 attrs={
   "data" : attr.label_list(),
 })
EOF
  cat > MODULE.bazel <<'EOF'
myrule = use_repo_rule("//:rule.bzl", "myrule")
myrule(name="ext", data = ["//:a.txt", "//:b.txt"])
EOF
  setup_module_dot_bazel "MODULE.bazel"
  echo Hello > a.txt
  echo World > b.txt
  bazel build @ext//:foo || fail "expected success"
  [ `cat "${WRKDIR}/log" | wc -l` -eq 1 ] \
      || fail "did not find precisely one invocation of the action"
}

test_unused_invalid_label_list_arg() {
  # Verify that we preserve the behavior of allowing to pass labels that
  # do referring to an non-existing path, if they are never used.
  # Here, test it if such labels are passed in a label list.
  WRKDIR=`pwd`
  rm -rf repo
  mkdir repo
  cd repo
  touch BUILD
  cat > rule.bzl <<'EOF'
def _impl(ctx):
  ctx.file("REPO.bazel")
  ctx.file("BUILD",
           "genrule(name=\"foo\", outs=[\"foo.txt\"], cmd = \"echo foo > $@\")")

myrule=repository_rule(implementation=_impl,
 attrs={
   "unused_list" : attr.label_list(),
 })
EOF
  cat > MODULE.bazel <<'EOF'
myrule = use_repo_rule("//:rule.bzl", "myrule")
myrule(name="ext", unused_list=["//does/not/exist:file1",
                                "//does/not/exists:file2"])
EOF
  setup_module_dot_bazel "MODULE.bazel"
  bazel build @ext//:foo || fail "expected success"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/10515
test_label_keyed_string_dict_arg() {
  # Verify that Bazel preloads Labels from label_keyed_string_dict, and as a
  # result, it runs the repository's implementation only once (i.e. it won't
  # restart the corresponding SkyFunction).
  WRKDIR=`pwd`
  rm -rf repo
  rm -rf log
  mkdir repo
  cd repo
  touch BUILD
  cat > rule.bzl <<EOF
def _impl(ctx):
    ctx.execute(["/bin/sh", "-c", "date +%s >> ${WRKDIR}/log"])
    ctx.file("REPO.bazel")
    ctx.file("BUILD", """
genrule(
    name = "foo",
    srcs = ["src.txt"],
    outs = ["foo.txt"],
    cmd = "cp \$< \$@",
)
""")
    for f in ctx.attr.data:
        # ctx.path(f) shouldn't trigger a restart since we've prefetched the value.
        ctx.execute(["/bin/sh", "-c", "cat %s >> src.txt" % ctx.path(f)])

myrule = repository_rule(
    implementation = _impl,
    attrs = {
        "data": attr.label_keyed_string_dict(),
    },
)
EOF
  cat > MODULE.bazel <<'EOF'
myrule = use_repo_rule("//:rule.bzl", "myrule")
myrule(name="ext", data = {"//:a.txt": "a", "//:b.txt": "b"})
EOF
  setup_module_dot_bazel "MODULE.bazel"
  echo Hello > a.txt
  echo World > b.txt
  bazel build @ext//:foo || fail "expected success"
  [ `cat "${WRKDIR}/log" | wc -l` -eq 1 ] \
      || fail "did not find precisely one invocation of the action"
}

test_unused_invalid_label_keyed_string_dict_arg() {
  # Verify that we preserve the behavior of allowing to pass labels that
  # do referring to an non-existing path, if they are never used.
  # Here, test it if such labels are passed in a label_keyed_string_dict.
  WRKDIR=`pwd`
  rm -rf repo
  mkdir repo
  cd repo
  touch BUILD
  cat > rule.bzl <<'EOF'
def _impl(ctx):
  ctx.file("REPO.bazel")
  ctx.file("BUILD",
           "genrule(name=\"foo\", outs=[\"foo.txt\"], cmd = \"echo foo > $@\")")

myrule=repository_rule(implementation=_impl,
 attrs={
   "unused_dict" : attr.label_keyed_string_dict(),
 })
EOF
  cat > MODULE.bazel <<'EOF'
myrule = use_repo_rule("//:rule.bzl", "myrule")
myrule(name="ext", unused_dict={"//does/not/exist:file1": "file1",
                                "//does/not/exists:file2": "file2"})
EOF
  setup_module_dot_bazel "MODULE.bazel"
  bazel build @ext//:foo || fail "expected success"
}

# Regression test for https://github.com/bazelbuild/bazel/issues/13441
function test_files_tracked_with_non_existing_files() {
  cat > rules.bzl <<'EOF'
def _repo_impl(ctx):
    ctx.symlink(ctx.path(Label("@//:MODULE.bazel")).dirname, "link")
    print("b.txt: " + ctx.read("link/b.txt"))
    print("c.txt: " + ctx.read("link/c.txt"))

    ctx.file("BUILD")
    ctx.file("REPO.bazel")

repo = repository_rule(
    _repo_impl,
    attrs = {"_files": attr.label_list(
        default = [
            Label("@//:a.txt"),
            Label("@//:b.txt"),
            Label("@//:c.txt"),
        ],
    )},
)
EOF

  cat > MODULE.bazel <<'EOF'
repo = use_repo_rule("//:rules.bzl", "repo")
repo(name = "ext")
EOF
  setup_module_dot_bazel "MODULE.bazel"
  touch BUILD

  # a.txt is intentionally not created
  echo "bbbb" > b.txt
  echo "cccc" > c.txt

  # The missing file dependency is tolerated.
  bazel build @ext//:all &> "$TEST_log" || fail "Expected repository rule to build"
  expect_log "b.txt: bbbb"
  expect_log "c.txt: cccc"

  echo "not_cccc" > c.txt
  bazel build @ext//:all &> "$TEST_log" || fail "Expected repository rule to build"
  expect_log "b.txt: bbbb"
  expect_log "c.txt: not_cccc"
}

run_suite "Starlark repo prefetching tests"
