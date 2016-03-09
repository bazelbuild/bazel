#!/bin/bash
#
# Copyright 2015 The Bazel Authors. All arights reserved.
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
# Tests the examples provided in Bazel
#

# Load test environment
source $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/test-setup.sh \
  || { echo "test-setup.sh not found!" >&2; exit 1; }

function setup_go() {
  copy_examples

  GO_REPO=$PWD/tools/build_rules/go/toolchain

  cat <<EOF > WORKSPACE
new_local_repository(
    name = "bazel_tools",
    path = __workspace_dir__,
    build_file = __workspace_dir__ + "/BUILD",
)
new_local_repository(
    name = "golang_linux_amd64",
    path = "$GO_REPO",
    build_file = "$GO_REPO/BUILD"
)
new_local_repository(
    name = "golang_darwin_amd64",
    path = "$GO_REPO",
    build_file = "$GO_REPO/BUILD"
)
EOF

  # avoid trying to download the SDK from within the test.
  rm -rf tools/build_rules/go/toolchain
  for p in golang_linux_amd64 golang_darwin_amd64 ; do
    d=$TEST_SRCDIR/external/${p}
    if [[ -d "${d}" ]]; then
      for f in $(cd ${d}; find -L . -type f -print ) ; do
        mkdir -p tools/build_rules/go/toolchain/$(dirname $f)
        ln -s ${d}/$f tools/build_rules/go/toolchain/$f
      done
    fi
  done
  cat <<'EOF' >tools/build_rules/go/toolchain/BUILD
package(
  default_visibility = [ "//visibility:public" ])

filegroup(
  name = "toolchain",
  srcs = glob(["go/bin/*", "go/pkg/**", ]),
)

filegroup(
  name = "go_tool",
  srcs = [ "go/bin/go" ],
)
EOF
  cat  <<EOF > BUILD
load("/tools/build_rules/go/def", "go_prefix")
go_prefix("prefix")
EOF

}

function test_basic() {
  setup_go
  mkdir -p ex/
  cat <<EOF > ex/m.go
package main
import (
  "fmt"

  "prefix/ex"
)
func main() {
  fmt.Println("F", ex.F())
}

EOF
  cat <<EOF > ex/l.go
package ex
func F() int { return 42 }
EOF

  cat <<EOF > ex/BUILD
load("/tools/build_rules/go/def", "go_library", "go_binary")
go_library(name = "go_default_library",
  srcs = [ "l.go"])
go_binary(name = "m",
  srcs = [ "m.go" ],
  deps = [ ":go_default_library" ])
EOF

  assert_build //ex:m
  test -x ./bazel-bin/ex/m || fail "binary not found"
  (./bazel-bin/ex/m > out) || fail "binary does not execute"
  grep "F 42" out || fail "binary output suspect"
}

function test_runfiles() {
  setup_go
  mkdir -p ex/

# Note this binary is also a test (for the correct handling of runfiles by
# Bazel's go_binary rule).
  cat <<EOF > ex/rf.go
package main
import (
  "fmt"
  "log"
  "io/ioutil"
)

func main() {
  rfcontent, err := ioutil.ReadFile("ex/runfile")
  if err != nil {
    log.Fatalf("Runfiles test binary: Error reading from runfile: %v", err)
  }

  fmt.Printf("Runfile: %s\n", rfcontent)
}

EOF

  cat <<EOF > ex/rf_test.go
package main
import (
  "fmt"
  "io/ioutil"
  "testing"
)

func TestRunfiles(t *testing.T) {
  rfcontent, err := ioutil.ReadFile("runfile")
  if err != nil {
    t.Errorf("TestRunfiles: Error reading from runfile: %v", err)
  }

  if string(rfcontent) != "12345\n" {
    t.Errorf("TestRunfiles: Read incorrect value from runfile: %s", rfcontent)
  }

  fmt.Printf("Runfile: %s\n", rfcontent)
}
EOF

  cat <<EOF > ex/runfile
12345
EOF

  cat <<EOF > ex/BUILD
load("/tools/build_rules/go/def", "go_binary", "go_test")
go_binary(name = "runfiles_bin",
  srcs = [ "rf.go" ],
  data = [ "runfile" ])
go_test(name = "runfiles_test",
  srcs = [ "rf_test.go" ],
  data = [ "runfile" ])
EOF

  assert_build //ex:runfiles_bin
  test -x ./bazel-bin/ex/runfiles_bin || fail "binary not found"
  (./bazel-bin/ex/runfiles_bin > out) || fail "binary does not execute"
  grep "Runfile: 12345" out || fail "binary output suspect"

  assert_build //ex:runfiles_test
  test -x ./bazel-bin/ex/runfiles_test || fail "binary not found"
  (./bazel-bin/ex/runfiles_test > out) || fail "binary does not execute"
  grep "Runfile: 12345" out || fail "binary output suspect"
}

function test_runfiles_lib() {
  setup_go
  mkdir -p ex/
  cat <<EOF > ex/m.go
package main
import (
  "fmt"
  "io/ioutil"
  "log"

  "prefix/ex"
)
func main() {
  rfcontent, err := ioutil.ReadFile(ex.RunfilePath())
  if err != nil {
    log.Fatalf("Runfiles test binary: Error reading from runfile: %v", err)
  }

  fmt.Printf("Runfile: %s\n", rfcontent)
}

EOF

  cat <<EOF > ex/l.go
package ex
func RunfilePath() string { return "ex/runfile" }
EOF

  cat <<EOF > ex/runfile
12345
EOF

    cat <<EOF > ex/BUILD
load("/tools/build_rules/go/def", "go_library", "go_binary")
go_library(name = "go_default_library",
  data = [ "runfile" ],
  srcs = [ "l.go"])
go_binary(name = "m",
  srcs = [ "m.go" ],
  deps = [ ":go_default_library" ])
EOF

  assert_build //ex:m
  test -x ./bazel-bin/ex/m || fail "binary not found"
  (./bazel-bin/ex/m > out) || fail "binary does not execute"
  grep "Runfile: 12345" out || fail "binary output suspect"
}


function test_empty_prefix() {
  setup_go

 cat  <<EOF > BUILD
load("/tools/build_rules/go/def", "go_prefix")
go_prefix("")
EOF

  rm -rf ex
  mkdir -p ex/
  cat <<EOF > ex/m.go
package main
import (
  "fmt"

  "library"
)
func main() {
  fmt.Println(library.F())
}
EOF
   cat <<EOF > ex/BUILD
load("/tools/build_rules/go/def", "go_library", "go_binary")
go_binary(name = "m",
  srcs = [ "m.go" ],
  deps = [ "//library:go_default_library" ])
EOF

  mkdir -p library

  cat <<EOF > library/BUILD
package(default_visibility=["//visibility:public"])
load("/tools/build_rules/go/def", "go_library", "go_binary")
go_library(name = "go_default_library",
  srcs = [ "l.go"])
EOF

   cat <<EOF > library/l.go
package library
func F() int { return 42 }
EOF

  assert_build //ex:m
  test -x ./bazel-bin/ex/m || fail "binary not found"
  (./bazel-bin/ex/m > out) || fail "binary does not execute"
  grep "42" out || fail "binary output suspect"
}

run_suite "go_examples"
