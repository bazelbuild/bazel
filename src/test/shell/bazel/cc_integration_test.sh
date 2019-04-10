#!/bin/bash -eu
#
# Copyright 2016 The Bazel Authors. All rights reserved.
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
# Tests the behavior of C++ rules.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

function test_extra_action_for_compile() {
  mkdir -p ea
  cat > ea/BUILD <<EOF
action_listener(
    name = "al",
    extra_actions = [":ea"],
    mnemonics = ["CppCompile"],
    visibility = ["//visibility:public"],
)

extra_action(
    name = "ea",
    cmd = "if ! [[ -r ea/cc.cc ]]; then echo 'source file not in inputs'; exit 1; fi",
)

cc_library(
    name = "cc",
    srcs = ["cc.cc"],
)
EOF

  echo 'void cc() {}' > ea/cc.cc

  bazel build --experimental_action_listener=//ea:al //ea:cc || fail "expected success"
}

function test_cc_library_include_prefix_external_repository() {
  r="$TEST_TMPDIR/r"
  mkdir -p "$TEST_TMPDIR/r/foo/v1"
  touch "$TEST_TMPDIR/r/WORKSPACE"
  echo "#define FOO 42" > "$TEST_TMPDIR/r/foo/v1/foo.h"
  cat > "$TEST_TMPDIR/r/foo/BUILD" <<EOF
cc_library(
  name = "foo",
  hdrs = ["v1/foo.h"],
  include_prefix = "foolib",
  strip_include_prefix = "v1",
  visibility = ["//visibility:public"],
)
EOF

  cat > WORKSPACE <<EOF
local_repository(
  name = "foo",
  path = "$TEST_TMPDIR/r",
)
EOF

  cat > BUILD <<EOF
cc_binary(
  name = "ok",
  srcs = ["ok.cc"],
  deps = ["@foo//foo"],
)

cc_binary(
  name = "bad",
  srcs = ["bad.cc"],
  deps = ["@foo//foo"],
)
EOF

  cat > ok.cc <<EOF
#include <stdio.h>
#include "foolib/foo.h"
int main() {
  printf("FOO is %d\n", FOO);
}
EOF

  cat > bad.cc <<EOF
#include <stdio.h>
#include "foo/v1/foo.h"
int main() {
  printf("FOO is %d\n", FOO);
}
EOF

  bazel build :bad && fail "Should not have found include at repository-relative path"
  bazel build :ok || fail "Should have found include at synthetic path"
}

function test_tree_artifact_headers_are_invalidated() {
  mkdir -p "ta_headers"
  cat > "ta_headers/BUILD" <<EOF
load(":mygen.bzl", "mygen")

sh_binary(
  name = "mygen_sh",
  srcs = ["mygen.sh"],
  visibility = ["//visibility:public"],
)

mygen(
    name="mylib_generated",
    srcs=[":mydef.txt"],
)

cc_library(
    name = "mylib",
    srcs = [":mylib_generated"],
    hdrs = [":mylib_generated"],
)

cc_binary(
    name = "myexec",
    srcs = [],
    deps = [":mylib"],
)
EOF
  cat > "ta_headers/mygen.sh" <<'EOF'
#!/bin/bash

set -euo pipefail

src_files=$1
hdr_files=$2

fc_name=$(cat ta_headers/mydef.txt)

mkdir -p ${src_files}
mkdir -p ${hdr_files}

cat > ${src_files}/main.c <<EOT
#include "ta_headers/files.h/another.h"
int main(void) {
    return MYFC();
}
EOT

cat > ${src_files}/another.c <<EOT
#include "ta_headers/files.h/another.h"
int ${fc_name}(void) {
    return 0;
}
EOT

cat > ${hdr_files}/another.h <<EOT
#define MYFC ${fc_name}
int ${fc_name}(void);
EOT
EOF
  chmod +x ta_headers/mygen.sh
  cat > "ta_headers/mygen.bzl" <<EOF
def _mygen_impl(ctx):
  args = ctx.actions.args()
  treeC = ctx.actions.declare_directory("files.c")
  treeH = ctx.actions.declare_directory("files.h")
  args.add(treeC.path)
  args.add(treeH.path)
  ctx.actions.run(
      inputs = ctx.files.srcs,
      outputs = [treeC, treeH],
      arguments = [args],
      executable = ctx.executable._mygen,
  )
  return [DefaultInfo(files=depset([treeC, treeH]))]

mygen = rule(
  implementation=_mygen_impl,
  attrs={
    "srcs": attr.label_list(allow_files=True),
    "_mygen": attr.label(
      cfg="host",
      executable=True,
      allow_files=True,
      default=":mygen_sh",
    ),
  },
)
EOF

  # So we have another.h defining a macro that is used by both main.c and
  # another.c. :main depends on :another, and gets the header through the
  # tree artifact. First build is fine.
  echo "fc1" > "ta_headers/mydef.txt"
  bazel build //ta_headers:myexec || fail \
    "First build failed, something is wrong with the test."

  # Now we change the content of another.h to define a different macro.
  # This test verifies that not only another.c is recompiled, but also
  # main.c. This is a regression test for
  # https://github.com/bazelbuild/bazel/issues/5785.
  echo "fc2" > "ta_headers/mydef.txt"
  bazel build //ta_headers:myexec || fail \
    "Second build failed, tree artifact was not invalidated."
}

# This test tests that Bazel can produce dynamic libraries that have undefined
# symbols on Mac and Linux. Not sure it is a sane default to allow undefined
# symbols, but it's the default we had historically. This test creates
# an executable (main) that defines bar(), and a shared library (plugin) that
# calls bar(). When linking the libplugin.so, symbol 'bar' is undefined.
#    +-----------------------------+     +----------------------------------+
#    |  main                       |     |  libplugin.so                    |
#    |                             |     |                                  |
#    |   main() { return foo(); } +---------> foo() { return bar() - 42; }  |
#    |                             |     |       +                          |
#    |                             |     |       |                          |
#    |   bar() { return 42; } <------------------+                          |
#    |                             |     |                                  |
#    +-----------------------------+     +----------------------------------+
function test_undefined_dynamic_lookup() {
  if is_windows; then
    # Windows doesn't allow undefined symbols in shared libraries.
    return 0
  fi
  mkdir -p "dynamic_lookup"
  cat > "dynamic_lookup/BUILD" <<EOF
cc_binary(
  name = "libplugin.so",
  srcs = ["plugin.cc"],
  linkshared = 1,
)

cc_binary(
    name = "main",
    srcs = ["main.cc", "libplugin.so"],
)
EOF

  cat > "dynamic_lookup/plugin.cc" <<EOF
int bar();
int foo() { return bar() - 42; }
EOF

  cat > "dynamic_lookup/main.cc" <<EOF
int foo();
int bar() { return 42; }
int main() { return foo(); }
EOF

  bazel build //dynamic_lookup:main || fail "Bazel couldn't build the binary."
  bazel run //dynamic_lookup:main || fail "Run of the binary failed."
}

function test_save_feature_state() {
  mkdir -p ea
  cat > ea/BUILD <<EOF
cc_library(
    name = "cc",
    srcs = ["cc.cc", "cc1.cc"],
    features = ["test_feature"],
)
EOF

  echo 'void cc() {}' > ea/cc.cc
  echo 'void cc1() {}' > ea/cc1.cc

  bazel build --experimental_save_feature_state //ea:cc || fail "expected success"
  ls bazel-bin/ea/feature_debug/cc/requested_features.txt || "requested_features.txt not created"
  ls bazel-bin/ea/feature_debug/cc/enabled_features.txt || "enabled_features.txt not created"
  # This assumes "grep" is supported in any environment bazel is used.
  grep "test_feature" bazel-bin/ea/feature_debug/cc/requested_features.txt || "test_feature should have been found in  requested_features."
}

# TODO: test include dirs and defines
function setup_cc_starlark_api_test() {
  local pkg="$1"

  touch "$pkg"/WORKSPACE

  mkdir "$pkg"/include_dir
  touch "$pkg"/include_dir/include.h
  mkdir "$pkg"/system_include_dir
  touch "$pkg"/system_include_dir/system_include.h
  mkdir "$pkg"/quote_include_dir
  touch "$pkg"/quote_include_dir/quote_include.h



  cat > "$pkg"/BUILD << EOF
load("//${pkg}:cc_api_rules.bzl", "cc_lib", "cc_bin")

cc_lib(
    name = "a",
    srcs = [
        "a1.cc",
        "a2.cc",
    ],
    private_hdrs = [
      "a2.h",
      "include_dir/include.h",
      "system_include_dir/system_include.h",
      "quote_include_dir/quote_include.h"
    ],
    user_compile_flags = ["-DA_DEFINITION_LOCAL"],
    public_hdrs = ["a.h"],
    includes = ["$pkg/include_dir"],
    system_includes = ["$pkg/system_include_dir"],
    quote_includes = ["$pkg/quote_include_dir"],
    defines = ["A_DEFINITION"],
    deps = [
        ":b",
        ":d",
    ],
)

cc_lib(
    name = "b",
    srcs = [
        "b.cc",
    ],
    public_hdrs = ["b.h"],
    deps = [":c"],
)

cc_lib(
    name = "c",
    srcs = [
        "c.cc",
    ],
    public_hdrs = ["c.h"],
)

cc_lib(
    name = "d",
    srcs = ["d.cc"],
    public_hdrs = ["d.h"],
)

cc_bin(
    name = "e",
    srcs = ["e.cc"],
    data = [":f"],
    linkstatic = 1,
    user_link_flags = [
        "-ldl",
        "-lm",
        "-Wl,-rpath,bazel-bin/${pkg}",
    ],
    deps = [
        ":a",
    ],
)

cc_bin(
    name = "f",
    srcs = ["f.cc"],
    linkshared = 1,
    deps = [
        ":a",
    ],
)
EOF

  cat > $pkg/a1.cc << EOF
#include <system_include.h>
#include "include.h"

#include "quote_include.h"
#include "a.h"
#include "a2.h"

#ifdef A_DEFINITION_LOCAL
#include "b.h"
#include "d.h"
#endif

using namespace std;

string alongernamethanusual() { return "a1" + a2() + b() + d(); }
EOF

  cat > $pkg/a2.cc << EOF
#include <string>
using namespace std;

string a2() { return "a2"; }
EOF

  cat > $pkg/a.h << EOF
#ifndef HEADER_A
#define HEADER_A
#include <string>
using namespace std;
string alongernamethanusual();
#endif
EOF

  cat > $pkg/a2.h << EOF
#ifndef HEADER_A2
#define HEADER_A2
#include <string>
using namespace std;
string a2();
#endif
EOF

  cat > $pkg/b.cc << EOF
#include "b.h"
#include <string>
#include "c.h"
using namespace std;

string b() { return "b" + c(); }
EOF

  cat > $pkg/b.h << EOF
#ifndef HEADER_B
#define HEADER_B
#include <string>
using namespace std;
string b();
#endif
EOF

  cat > $pkg/c.cc << EOF
#include "c.h"
#include <algorithm>
#include <string>

using namespace std;

string c() { return "c"; }
EOF

  cat > $pkg/c.h << EOF
#ifndef HEADER_C
#define HEADER_C
#include <string>
using namespace std;
string c();
#endif
EOF

  cat > $pkg/d.cc << EOF
#include "d.h"
#include <string>
using namespace std;

string d() { return "d"; }
EOF

  cat > $pkg/d.h << EOF
#ifndef HEADER_D
#define HEADER_D
#include <string>
using namespace std;
string d();
#endif
EOF

  cat > $pkg/e.cc << EOF
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifdef A_DEFINITION
#include "a.h"
#endif

#ifdef A_DEFINITION_LOCAL
#include "thisdoesntexist.h"
#endif

using namespace std;

int main() {
  void* handle = dlopen("libf.so", RTLD_LAZY);

  typedef string (*f_t)();

  f_t f = (f_t)dlsym(handle, "f");
  cout << alongernamethanusual() + f() << endl;
  return 0;
}

EOF

  cat > $pkg/f.cc << EOF
#include <algorithm>
#include <string>
#include "a.h"

using namespace std;

extern "C" string f() {
  string str = alongernamethanusual();
  reverse(str.begin(), str.end());
  return str;
}
EOF

  cat > $pkg/script.lds << EOF
VERS_42.0 {
  local:
    *;
};
EOF

  cp "$CURRENT_DIR"/cc_api_rules.bzl "$pkg"/cc_api_rules.bzl

}

function test_cc_starlark_api_default_values() {
  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"

  setup_cc_starlark_api_test "${FUNCNAME[0]}"

  bazel build --experimental_cc_skylark_api_enabled_packages=, --verbose_failures \
    //"$pkg":e  &>"$TEST_log" || fail "Build failed"

  nm -u bazel-bin/"$pkg"/e  | grep alongernamethanusual && \
    fail "alongernamethanusual is not defined"

  bazel-bin/"$pkg"/e | grep a1a2bcddcb2a1a || fail "output is incorrect"
}


function test_cc_starlark_api_link_static_false() {
  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"

  setup_cc_starlark_api_test "${FUNCNAME[0]}"

  cat >> "$pkg"/BUILD << EOF
cc_bin(
    name = "g",
    srcs = ["e.cc"],
    data = [":f"],
    linkstatic = 0,
    user_link_flags = [
        "-ldl",
        "-lm",
        "-Wl,-rpath,bazel-bin/${pkg}",
    ],
    deps = [
        ":a",
    ],
)
EOF

  bazel build --experimental_cc_skylark_api_enabled_packages=, --verbose_failures \
    //"$pkg":g  &>"$TEST_log" || fail "Build failed"

  nm -u bazel-bin/"$pkg"/g  | grep alongernamethanusual || fail "alongernamethanusual is defined"

  bazel-bin/"$pkg"/g | grep a1a2bcddcb2a1a || fail "output is incorrect"
}

function test_cc_starlark_api_additional_inputs() {
  # This uses --version-script which isn't available on Mac linker.
  [ "$PLATFORM" != "darwin" ] || return 0

  local pkg="${FUNCNAME[0]}"
  mkdir -p "$pkg"

  setup_cc_starlark_api_test "${FUNCNAME[0]}"

  cat >> "$pkg"/BUILD << EOF
cc_bin(
    name = "g",
    srcs = ["e.cc"],
    data = [":f"],
    linkstatic = 1,
    additional_linker_inputs = ["script.lds"],
    user_link_flags = [
        "-ldl",
        "-lm",
        "-Wl,-rpath,bazel-bin/${pkg}",
        "-Wl,--version-script=\$(location script.lds)",
    ],
    deps = [
        ":a",
    ],
)
EOF

  bazel build --experimental_cc_skylark_api_enabled_packages=, --verbose_failures \
    //"$pkg":g  &>"$TEST_log" || fail "Build failed"

  nm bazel-bin/"$pkg"/g  | grep VERS_42.0 || fail "VERS_42.0 not in binary"

  bazel-bin/"$pkg"/g | grep a1a2bcddcb2a1a || fail "output is incorrect"
}



run_suite "cc_integration_test"