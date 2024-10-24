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
# Test legitimate path assumptions when working with external repositories.
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }
source "${CURRENT_DIR}/remote_helpers.sh" \
  || { echo "remote_helpers.sh not found!" >&2; exit 1; }


repo_with_local_include() {
  # Generate a repository, in the current working directory, with a target
  # //src:hello that includes a file via a local path.

  setup_module_dot_bazel
  mkdir src
  cat > src/main.c <<'EOF'
#include <stdio.h>
#include "src/consts/greeting.h"

int main(int argc, char **argv) {
  printf("%s\n", GREETING);
  return 0;
}
EOF
  mkdir src/consts
  cat > src/consts/greeting.h <<'EOF'
#define GREETING "Hello World"
EOF
  cat > src/BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c", "consts/greeting.h"],
)
EOF
}

library_with_local_include() {
  # Generates a repository, in the current directory, where a target //lib:hello
  # is a library with headers that include via paths relative to the root of
  # that repository

  setup_module_dot_bazel
  mkdir lib
  cat > lib/lib.h <<'EOF'
#include "lib/constants.h"

int greet(char *);

EOF
  cat > lib/constants.h <<'EOF'
#define TARGET "World"
EOF
  cat > lib/lib.c <<'EOF'
#include <stdio.h>

int greet(char *s) {
  printf("Hello %s\n", s);
  return 0;
}
EOF
  cat > lib/BUILD <<EOF
cc_library(
  name="lib",
  srcs=["lib.c"],
  hdrs=["lib.h", "constants.h"],
  visibility = ["//visibility:public"],
)
EOF
}


test_local_paths_main () {
  # Verify that a target in the main repository may refer to a truly source
  # file in its own repository by a path relative to the repository root.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  cd main
  repo_with_local_include

  bazel build //src:hello || fail "Expected build to succeed"
  bazel run //src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_local_paths_remote() {
  # Verify that a target in an external repository may refer to a truly source
  # file in its own repository by a path relative to the root of that repository
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && repo_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remote",
  strip_prefix="remote",
  urls=["file://${WRKDIR}/remote.tar"],
)
EOF

  bazel build @remote//src:hello || fail "Expected build to succeed"
  bazel run @remote//src:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_lib_paths_main() {
  # Verify that libraries from the main repository can be used via include
  # path relative to their repository root and that they may refer to other
  # truly source files from the same library via paths relative to their
  # repository root.

  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  cd main
  library_with_local_include

  cat > main.c <<'EOF'
#include "lib/lib.h"

int main(int argc, char **argv) {
  greet(TARGET);
  return 0;
}
EOF
  cat > BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c"],
  deps=["//lib:lib"],
)
EOF

  bazel build //:hello || fail "Expected build to succeed"
  bazel run //:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_lib_paths_remote() {
  # Verify that libraries from an external repository can be used via include
  # path relative to their repository root and that they may refer to other
  # truly source files from the same library via paths relative to their
  # repository root.

  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && library_with_local_include)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remote",
  strip_prefix="remote",
  urls=["file://${WRKDIR}/remote.tar"],
)
EOF
  cat > main.c <<'EOF'
#include "lib/lib.h"

int main(int argc, char **argv) {
  greet(TARGET);
  return 0;
}
EOF
  cat > BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c"],
  deps=["@remote//lib:lib"],
)
EOF

  bazel build //:hello || fail "Expected build to succeed"
  bazel run //:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

test_lib_paths_all_remote() {
  # Verify that libraries from an external repository can be used by another
  # external repository via include path relative to their repository root and
  # that they may refer to other truly source files from the same library via
  # paths relative to their repository root.

  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remotelib
  (cd remotelib && library_with_local_include)
  tar cvf remotelib.tar remotelib
  rm -rf remotelib

  mkdir remotemain
  (cd remotemain
  cat > main.c <<'EOF'
#include "lib/lib.h"

int main(int argc, char **argv) {
  greet(TARGET);
  return 0;
}
EOF
  cat > BUILD <<'EOF'
cc_binary(
  name="hello",
  srcs=["main.c"],
  deps=["@remotelib//lib:lib"],
)
EOF
)
  tar cvf remotemain.tar remotemain
  rm -rf remotemain

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remotelib",
  strip_prefix="remotelib",
  urls=["file://${WRKDIR}/remotelib.tar"],
)
http_archive(
  name="remotemain",
  strip_prefix="remotemain",
  urls=["file://${WRKDIR}/remotemain.tar"],
)
EOF
  bazel build @remotemain//:hello || fail "Expected build to succeed"
  bazel run @remotemain//:hello | grep 'Hello World' \
      || fail "Expected output 'Hello World'"
}

repo_with_local_path_reference() {
  # create, in the current working directory, a package called
  # withpath, that contains rule depending on hard-code path relative
  # to the repository root.
  setup_module_dot_bazel
  mkdir -p withpath
  cat > withpath/BUILD <<'EOF'
genrule(
  name = "it",
  srcs = ["double.sh", "data.txt"],
  outs = ["it.txt"],
  cmd = "sh $(location double.sh) > $@",
  visibility = ["//visibility:public"],
)
EOF
  cat > withpath/double.sh <<'EOF'
#!/bin/sh
cat withpath/data.txt withpath/data.txt
EOF
  cat > withpath/data.txt <<'EOF'
Hello world
EOF
}

test_fixed_path_local() {
  # Verify that hard-coded path relative to the repository root can
  # be used in internal targets.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  cd main
  repo_with_local_path_reference

  bazel build //withpath:it || fail "Expected success"
}

# TODO(aehlig): enable, once our execroot change is far enough
# to make this (desirbale) property true.
DISABLED_test_fixed_path_remote() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && repo_with_local_path_reference)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="remote",
  strip_prefix="remote",
  urls=["file://${WRKDIR}/remote.tar"],
)
EOF

  bazel build @remote//withpath:it || fail "Expected success"
}
repo_with_local_implicit_dependencies() {
  # create, in the current working directory, a package called rule
  # that has an implicit dependency on a target in the same repository;
  # the point here is that this dependency can be named without knowledge
  #  of the repository name.
  setup_module_dot_bazel
  mkdir -p rule
  cat > rule/BUILD <<'EOF'
exports_files(["to_upper.sh"])
EOF
  cat > rule/to_upper.sh <<'EOF'
cat $1 | tr 'a-z' 'A-Z' > $2
EOF
  cat > rule/to_upper.bzl <<'EOF'
def _to_upper_impl(ctx):
  output = ctx.actions.declare_file(ctx.label.name + ".txt")
  ctx.actions.run(
    inputs = ctx.files.src + ctx.files._toupper_sh,
    outputs = [output],
    executable = "/bin/sh",
    arguments = [f.path for f in ctx.files._toupper_sh]
              +  [f.path for f in ctx.files.src] + [output.path],
    use_default_shell_env = True,
    mnemonic = "ToUpper",
    progress_message = "Uppercasing %s" % ctx.label,
  )

to_upper = rule(
  implementation = _to_upper_impl,
  attrs = {
    "src" : attr.label(allow_files=True),
    "_toupper_sh" : attr.label(cfg="exec", allow_files=True,
                               default = Label("//rule:to_upper.sh")),
  },
  outputs = {"upper": "%{name}.txt"},
  )
EOF
}

test_local_rules() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  cd main
  repo_with_local_implicit_dependencies
  mkdir call
  echo hello world > call/hello.txt
  cat > call/BUILD <<'EOF'
load("//rule:to_upper.bzl", "to_upper")
to_upper(
  name = "upper_hello",
  src = "hello.txt"
)
EOF

  bazel build -s //call:upper_hello || fail "Expected success"
  cat `bazel info bazel-bin`/call/upper_hello.txt | grep 'HELLO WORLD' \
    || fail "not the expected output"

}

test_remote_rules() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && repo_with_local_implicit_dependencies)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="r",
  strip_prefix="remote",
  urls=["file://${WRKDIR}/remote.tar"],
)
EOF
  mkdir call
  echo hello world > call/hello.txt
  cat > call/BUILD <<'EOF'
load("@r//rule:to_upper.bzl", "to_upper")
to_upper(
  name = "upper_hello",
  src = "hello.txt"
)
EOF

  bazel build -s //call:upper_hello || fail "Expected success"
  cat `bazel info bazel-bin`/call/upper_hello.txt | grep 'HELLO WORLD' \
    || fail "not the expected output"
}

test_remote_remote_rules() {
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir a
  (cd a && repo_with_local_implicit_dependencies)
  tar cvf a.tar a
  rm -rf a

  mkdir b
  (cd b
  mkdir call
  echo hello world > call/hello.txt
  cat > call/BUILD <<'EOF'
load("@a//rule:to_upper.bzl", "to_upper")
to_upper(
  name = "upper_hello",
  src = "hello.txt"
)
EOF
  )
  tar cvf b.tar b
  rm -rf b

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="a",
  strip_prefix="a",
  urls=["file://${WRKDIR}/a.tar"],
)
http_archive(
  name="b",
  strip_prefix="b",
  urls=["file://${WRKDIR}/b.tar"],
)
EOF

  bazel build -s @b//call:upper_hello || fail "Expected success"
}

repo_with_embedded_paths() {
  # create, in the current working directory, a package called rule
  # that has an implicit dependency on a target in the same repository
  # that is referred-to by an embedded path.
  setup_module_dot_bazel
  mkdir -p rule
  cat > rule/preamb.html <<'EOF'
<html>
<body>
<pre>
EOF
  cat > rule/postamb.html <<'EOF'
</pre>
</body>
</html>
EOF
  cat > rule/BUILD <<'EOF'
exports_files(["preamb.html", "postamb.html"], visibility = ["//visibility:public"])

genrule(
  name = "to_html",
  outs = ["to_html.sh"],
  srcs = [":preamb.html", ":postamb.html"], # the output actually does not depend on those files
  cmd = "echo '#!/bin/sh' > $@; echo 'cat $(location :preamb.html) $$1 $(location :postamb.html) > $$2' >> $@",
  visibility = ["//visibility:public"],
)
EOF
  cat > rule/to_html.bzl <<'EOF'
def _to_html_impl(ctx):
  output = ctx.actions.declare_file(ctx.label.name + ".html")
  ctx.actions.run(
    inputs = ctx.files.src + ctx.files._to_html + ctx.files._preamb + ctx.files._postamb,
    outputs = [output],
    executable = "/bin/sh",
    arguments = [f.path for f in ctx.files._to_html]
              +  [f.path for f in ctx.files.src] + [output.path],
    use_default_shell_env = True,
    mnemonic = "ToHtml",
    progress_message = "htmlifying %s" % ctx.label,
  )

to_html = rule(
  implementation = _to_html_impl,
  attrs = {
    "src" : attr.label(allow_files=True),
    "_to_html" : attr.label(cfg="exec", allow_files=True,
                               default = Label("//rule:to_html")),
    # knowledge of which paths are embedded is duplicated here!
    "_preamb" : attr.label(cfg="exec", allow_files=True,
                               default = Label("//rule:preamb.html")),
    "_postamb" : attr.label(cfg="exec", allow_files=True,
                               default = Label("//rule:postamb.html")),
  },
  outputs = {"upper": "%{name}.html"},
  )
EOF
}


test_embedded_local() {
  # Verify that files with embedded paths can be used locally.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  repo_with_embedded_paths
  mkdir call
  cat > call/plain.txt <<'EOF'
Hello World!
EOF
  cat > call/BUILD <<'EOF'
load('//rule:to_html.bzl', 'to_html')

to_html(name="hello", src="plain.txt")
EOF

  bazel build -s //call:hello || fail 'Expected success'
  cat `bazel info bazel-bin`/call/hello.html | grep '<html>' \
    || fail "not the expected output"
  cat `bazel info bazel-bin`/call/hello.html | grep '</html>' \
    || fail "not the expected output"
}

test_embedded_remote() {
  # Verify that files with embedded paths can be used if coming
  # from an external repository.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir remote
  (cd remote && repo_with_embedded_paths)
  tar cvf remote.tar remote
  rm -rf remote

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="r",
  strip_prefix="remote",
  urls=["file://${WRKDIR}/remote.tar"],
)
EOF
  mkdir call
  cat > call/plain.txt <<'EOF'
Hello World!
EOF
  cat > call/BUILD <<'EOF'
load('@r//rule:to_html.bzl', 'to_html')

to_html(name="hello", src="plain.txt")
EOF

  bazel build -s //call:hello || fail 'Expected success'
  cat `bazel info bazel-bin`/call/hello.html | grep '<html>' \
    || fail "not the expected output"
  cat `bazel info bazel-bin`/call/hello.html | grep '</html>' \
    || fail "not the expected output"
}

test_embedded_remote_remote() {
  # Verify that files with embedded path can be used by a remote
  # repository if coming from an external repository.
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir r
  (cd r && repo_with_embedded_paths)
  tar cvf r.tar r
  rm -rf r

  mkdir b
  (cd b
  mkdir call
  cat > call/plain.txt <<'EOF'
Hello World!
EOF
  cat > call/BUILD <<'EOF'
load('@r//rule:to_html.bzl', 'to_html')

to_html(name="hello", src="plain.txt")
EOF
  )
  tar cvf b.tar b
  rm -rf b

  mkdir main
  cd main
  cat >> $(setup_module_dot_bazel) <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="r",
  strip_prefix="r",
  urls=["file://${WRKDIR}/r.tar"],
)
http_archive(
  name="b",
  strip_prefix="b",
  urls=["file://${WRKDIR}/b.tar"],
)
EOF

  bazel build -s @b//call:hello || fail "Expected success"
}

repo_with_embedded_foreign_path() {
  # create, in the current working directory, a package called
  # rule that has an implicit dependency on @data//:file.txt, a target
  # of a different, external, repository.
  mkdir -p rule
  cat > rule/BUILD <<'EOF'
genrule(
  name = "add_preamb",
  outs = ["add_preamb.sh"],
  srcs = ["@data//:file.txt"], # the output actually does not depend on the contents of those files
  cmd = " echo '#!/bin/sh' > $@; echo 'cat $(location @data//:file.txt) $$1 > $$2' >> $@",
  visibility = ["//visibility:public"],
)
EOF
  cat > rule/add_preamb.bzl <<'EOF'
def _add_preamb_impl(ctx):
  output = ctx.actions.declare_file(ctx.label.name + ".txt")
  ctx.actions.run(
    inputs = ctx.files.src + ctx.files._add_preamb + ctx.files._preamb,
    outputs = [output],
    executable = "/bin/sh",
    arguments = [f.path for f in ctx.files._add_preamb] \
              +  [f.path for f in ctx.files.src] + [output.path],
    use_default_shell_env = True,
    mnemonic = "AddPreamb",
    progress_message = "Add preamble to %s" % ctx.label,
  )

add_preamb = rule(
  implementation = _add_preamb_impl,
  attrs = {
    "src" : attr.label(allow_files=True),
    "_add_preamb" : attr.label(cfg="exec", allow_files=True,
                               default = Label("//rule:add_preamb")),
    # knowledge of which paths are embedded is duplicated here!
    "_preamb" : attr.label(cfg="exec", allow_files=True,
                               default = Label("@data//:file.txt")),
  },
  outputs = {"with_preamb": "%{name}.txt"},
)
EOF
}

repo_data_file() {
  # Create, in the current directory, an archive of a data repository containing
  # //:file.txt, and add a corresponding entry to ./main/MODULE.bazel.
  mkdir data
  cat > data/file.txt <<'EOF'
Copyright ...
EOF
  cat > data/BUILD <<'EOF'
exports_files(["file.txt"], visibility = ["//visibility:public"])
EOF
  tar cvf data.tar data
  rm -rf data
  cat >> $(setup_module_dot_bazel "main/MODULE.bazel") <<EOF
http_archive = use_repo_rule("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
  name="data",
  strip_prefix="data",
  urls=["file://$(pwd)/data.tar"],
)
EOF
}

test_embedded_foreign_paths_local() {
  # Verify that a rule in a local repository can embed a path to a foreigh
  # repository
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  repo_data_file

  cd main
  repo_with_embedded_foreign_path
  echo Hello world > foo.txt
  cat > BUILD <<'EOF'
load('//rule:add_preamb.bzl', 'add_preamb')

add_preamb(name='main', src='foo.txt')
EOF

  bazel build -s //:main || fail 'Expected success'
  cat `bazel info bazel-bin`/main.txt | grep 'world' \
    || fail "not the expected output"
  cat `bazel info bazel-bin`/main.txt | grep 'Copyright' \
    || fail "not the expected output"
}

test_embedded_foreign_paths_remote() {
  # Verify that a rule in a local repository can embed a path to a foreigh
  # repository
  WRKDIR=$(mktemp -d "${TEST_TMPDIR}/testXXXXXX")
  cd "${WRKDIR}"

  mkdir main
  repo_data_file

  mkdir rule
  (cd rule && repo_with_embedded_foreign_path)
  tar cvf rule.tar rule
  rm -rf rule
  cat >> main/MODULE.bazel <<EOF
http_archive(
  name="rule",
  strip_prefix="rule",
  urls=["file://$(pwd)/rule.tar"],
)
EOF

  cd main
  echo Hello world > foo.txt
  cat > BUILD <<'EOF'
load('@rule//rule:add_preamb.bzl', 'add_preamb')

add_preamb(name='main', src='foo.txt')
EOF

  bazel build -s //:main || fail 'Expected success'
  cat `bazel info bazel-bin`/main.txt | grep 'world' \
    || fail "not the expected output"
  cat `bazel info bazel-bin`/main.txt | grep 'Copyright' \
    || fail "not the expected output"
}

run_suite "path tests for multiple repositories"
