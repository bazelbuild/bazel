#!/bin/bash
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
# Tests the examples provided in Bazel
#

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

# Appends to "WORKSPACE" the declaration of 2 local repositories.
# Assumes the main content of WORKSPACE was created previously.
function add_local_repos_to_workspace() {
  cat >> WORKSPACE <<EOF
local_repository(
  name = "repo",
  path = "a/b"
)

local_repository(
  name = "main_repo",
  path = "c/d"
)
EOF
}

# Appends to the WORKSPACE file under a given path (the first argument) the dependencies needed
# for proto_library.
function write_workspace() {
  workspace=""
  if [ ! -z "$1" ];
  then
    workspace=$1
    mkdir -p "$workspace"
  fi

  cat >> "$workspace"WORKSPACE << EOF
# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "com_google_protobuf",
    sha256 = "cef7f1b5a7c5fba672bec2a319246e8feba471f04dcebfe362d55930ee7c1c30",
    strip_prefix = "protobuf-3.5.0",
    urls = ["https://github.com/google/protobuf/archive/v3.5.0.zip"],
)

# java_lite_proto_library rules implicitly depend on @com_google_protobuf_javalite//:javalite_toolchain,
# which is the JavaLite proto runtime (base classes and common utilities).
http_archive(
    name = "com_google_protobuf_javalite",
    sha256 = "d8a2fed3708781196f92e1e7e7e713cf66804bd2944894401057214aff4f468e",
    strip_prefix = "protobuf-5e8916e881c573c5d83980197a6f783c132d4276",
    urls = ["https://github.com/google/protobuf/archive/5e8916e881c573c5d83980197a6f783c132d4276.zip"],
)
EOF
}

# Creates directories and files with the structure:
#  x/
#   person/
#     BUILD
#     person.proto (imports "bar/bar.proto", has proto_source_root = "x/person")
#     phonenumber/
#       phonenumber.proto
#   phonebook/
#     BUILD
#     phonebook.proto (imports "person.proto" & "phonenumber/phonenumber.proto")
#
# The BUILD files could use directly "proto_library" rules or a macro called
# "proto_library_macro" that should be created beforehand using write_macro().
#
# Expected arguments:
# 1. The name of the proto library rule. Can be "proto_library" or
#    "proto_library_macro".
# 2. A row in the BUILD file that specifies the proto_source_root attribute on
#    proto_library. Should be left empty if a macro is used.
# 3. A load statement that includes a macro containing a wrapper around
#     proto_library.
function write_setup() {
  mkdir -p x/person/phonenumber
  proto_library_name=$1
  proto_source_root=$2
  include_macro=$3

  cat > x/person/BUILD << EOF
package(default_visibility = ["//visibility:public"])
$include_macro
$proto_library_name(
  name = "person_proto",
  srcs = ["person.proto"],
  deps = [":phonenumber_proto"],
  $proto_source_root
)

$proto_library_name(
  name = "phonenumber_proto",
  srcs = ["phonenumber/phonenumber.proto"],
)
EOF

  cat > x/person/person.proto << EOF
syntax = "proto2";

option java_package = "person";
option java_outer_classname = "Person";

import "phonenumber/phonenumber.proto";

message PersonProto {
  optional string name = 1;

  optional PhoneNumberProto phone = 2;
}
EOF

  cat > x/person/phonenumber/phonenumber.proto << EOF
syntax = "proto2";

option java_package = "person.phonenumber";
option java_outer_classname = "PhoneNumber";

message PhoneNumberProto {
  required string number = 1;
}
EOF

  mkdir -p x/phonebook

  cat > x/phonebook/BUILD << EOF
$proto_library_name(
  name = "phonebook",
  srcs = ["phonebook.proto"],
  deps = ["//x/person:person_proto"],
)
EOF

  cat > x/phonebook/phonebook.proto << EOF
import "person.proto";
import "phonenumber/phonenumber.proto";

message Agenda {
  required PersonProto person = 1;
  required PhoneNumberProto number = 2;
}
EOF
}

# Creates the files in the following directory structure:
#  proto_library/
#                BUILD <- empty
#                src/
#                    BUILD <- has target "all" for all .proto
#                    address.proto  <- imports "person.proto"
#                    person.proto   <- imports "address.proto"
#                    zip_code.proto
function write_regression_setup() {
  mkdir -p proto_library/src
  touch proto_library/BUILD

  cat > proto_library/src/BUILD << EOF
proto_library(
    name = "all",
    srcs = glob(["*.proto"]),
    proto_source_root = package_name(),
)
EOF

  cat > proto_library/src/address.proto <<EOF
syntax = "proto3";

package demo; // Required to generate valid code.

// Always import protos with a full path relative to the WORKSPACE file.
import "zip_code.proto";

message Address {
  string city = 1;
  ZipCode zip_code = 2;
}
EOF

  cat > proto_library/src/person.proto <<EOF
syntax = "proto3";

package demo; // Required to generate valid code.

// Always import protos with a full path relative to the WORKSPACE file.
import "address.proto";

message Person {
  string name = 1;
  int32 id = 2;
  string email = 3;
  Address address = 4;
}
EOF

  cat > proto_library/src/zip_code.proto <<EOF
syntax = "proto3";

package demo; // Requried to generate valid code.

message ZipCode {
  string code = 1;
}
EOF
}

# Creates the directories and files in following the structure:
#  a/b/
#      WORKSPACE <- workspace referenced as "repo"
#      BUILD <- empty
#      src/
#          BUILD <- "all_protos" with proto_source_root
#          address.proto <- imports "zip_code.proto"
#          zip_code.proto
#  c/d/
#      WORKSPACE <- workspace referenced as "main_repo"
#      BUILD <- empty
#      src/
#          BUILD <- "all_protos" with proto_source_root
#          person.proto <- imports "address.proto" and depends on @repo//src:all_protos
function write_workspaces_setup() {
  mkdir -p a/b/src

  touch a/b/BUILD
  cat > a/b/src/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
proto_library(
  name = "all_protos",
  srcs = glob(["*.proto"]),
  proto_source_root = package_name()
)
EOF


  cat > a/b/src/address.proto <<EOF
syntax = "proto3";

package demo; // Required to generate valid code.

// Always import protos with a full path relative to the WORKSPACE file.
import "zip_code.proto";

message Address {
  string city = 1;
  ZipCode zip_code = 2;
}
EOF

  cat > a/b/src/zip_code.proto <<EOF
syntax = "proto3";

package demo; // Requried to generate valid code.

message ZipCode {
  string code = 1;
}
EOF

  #### WORKSPACE(c/d) ####
  mkdir -p c/d/src

  touch c/d/BUILD

  cat > c/d/src/BUILD <<EOF
package(default_visibility = ["//visibility:public"])
proto_library(
  name = "all_protos",
  srcs = glob(["*.proto"]),
  proto_source_root = package_name(),
  deps = ["@repo//src:all_protos"]
)
EOF

  cat > c/d/src/person.proto <<EOF
syntax = "proto3";

package demo; // Required to generate valid code.

// Always import protos with a full path relative to the WORKSPACE file.
import "address.proto";

message Person {
  string name = 1;
  int32 id = 2;
  string email = 3;
  Address address = 4;
}
EOF

}


# Creates macros/BUILD and macros/proto_library_macro.bzl, which contains a
# macro that wraps the proto_library rule. The macro passes to proto_library the
# same "name", "srcs", "deps" and adds "proto_source_root = native.package_name()".
# This will be a common use case for the "proto_source_root" attribute.
function write_macro() {
  mkdir macros
  cat > macros/BUILD << EOF
export_files(["proto_library_macro.bzl])
EOF

  cat > macros/proto_library_macro.bzl << EOF
def proto_library_macro(name, srcs, deps = []):
  native.proto_library(
      name = name,
      srcs = srcs,
      deps = deps,
      proto_source_root = native.package_name()
  )
EOF
}

# Creates the directories and files from the following structure
#   java/com/google/src/
#                       BUILD
#                       A.java
function write_java_library() {
  # should depend on x/foo:foo
  mkdir -p java/com/google/src
  cat > java/com/google/src/BUILD << EOF
java_library(
  name = "top",
  srcs = ["A.java"],
  deps = [":jpl"]
)

java_proto_library(
  name = "jpl",
  deps = ["//x/person:person_proto"]
)
EOF

  cat > java/com/google/src/A.java << EOF
import person.Person.PersonProto;
import person.phonenumber.PhoneNumber.PhoneNumberProto;

public class A {
  private PersonProto person;

  public A(PersonProto person) {
    this.person = person;
    PhoneNumberProto number = person.getPhone();
  }
}

EOF

}

############# TESTS #############

function test_proto_source_root() {
  write_workspace ""
  write_setup "proto_library" "proto_source_root = 'x/person'" ""
  bazel build --noincompatible_package_name_is_a_function //x/person:person_proto > "$TEST_log" || fail "Expected success"
}

function test_proto_source_root_fails() {
  write_workspace ""
  # Don't specify the "proto_source_root" attribute and expect failure.
  write_setup "proto_library" "" ""
  bazel build --noincompatible_package_name_is_a_function //x/person:person_proto >& "$TEST_log"  && fail "Expected failure"
  expect_log "phonenumber/phonenumber.proto: File not found."
}

function test_proto_source_root_macro() {
  write_workspace ""
  write_macro
  write_setup "proto_library_macro" "" "load('//macros:proto_library_macro.bzl', 'proto_library_macro')"
  bazel build --noincompatible_package_name_is_a_function //x/person:person_proto > "$TEST_log" || fail "Expected success"
}

# Fails with "IllegalArgumentException: external/lcocal_jdk in
# DumpPlatformClassPath.dumpJDK9AndNewerBootClassPath.java:67
function DISABLED_test_proto_source_root_with_java_library() {
  write_workspace ""
  write_setup "proto_library" "proto_source_root = 'x/person'" ""
  write_java_library
  bazel build --noincompatible_package_name_is_a_function //java/com/google/src:top \
      --strict_java_deps=off > "$TEST_log"  || fail "Expected success"
}

function test_proto_source_root_glob() {
  write_workspace ""
  write_regression_setup
  bazel build --noincompatible_package_name_is_a_function //proto_library/src:all >& "$TEST_log" || fail "Expected success"
}

function test_proto_source_root_glob() {
  write_workspace ""
  write_regression_setup
  bazel build --noincompatible_package_name_is_a_function //proto_library/src:all --strict_proto_deps=off >& "$TEST_log" \
      || fail "Expected success"
}

# TODO(elenairina): Enable this after #4665 is fixed.
function DISABLED_test_proto_source_root_multiple_workspaces() {
  write_workspace "a/b/"
  write_workspace "c/d/"
  write_workspace ""
  add_local_repos_to_workspace
  write_workspaces_setup

  bazel build --noincompatible_package_name_is_a_function @main_repo//src:all_protos >& "$TEST_log" || fail "Expected success"
}

run_suite "Integration tests for proto_library"
