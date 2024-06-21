#!/bin/bash
#
# Copyright 2025 The Bazel Authors. All rights reserved.
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
# An end-to-end test of the behavior of tools/build_rules/prelude_bazel.

# Load the test setup defined in the parent directory
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CURRENT_DIR}/../integration_test_setup.sh" \
  || { echo "integration_test_setup.sh not found!" >&2; exit 1; }

#### SETUP #############################################################

set -e

#### TESTS #############################################################

# TODO: enable this once we have Android rules release
function disabled_test_removed_rule_loaded() {
  create_workspace_with_default_repos WORKSPACE

  cat > BUILD << EOF
aar_import(
    name = 'aar',
    aar = 'aar.file',
    deps = [],
)
EOF
  bazel build --incompatible_load_rules_externally=aar_import :aar >&$TEST_log 2>&1 || fail "build failed"
}

# TODO: enable this once we have Android rules release
function disabled_test_removed_rule_loaded_from_bzl() {
  create_workspace_with_default_repos WORKSPACE

  cat > macro.bzl << EOF
def macro():
    native.aar_import(
        name = 'aar',
        aar = 'aar.file',
        deps = [],
    )
EOF

  cat > BUILD << EOF
load(":macro.bzl", "macro")
macro()
EOF
  bazel build --incompatible_load_rules_externally=aar_import :aar >&$TEST_log 2>&1 || fail "build failed"
}


# TODO: enable this once we have a removed symbol
function disabled_test_removed_symbol_loaded() {
  create_workspace_with_default_repos WORKSPACE

  cat > symbol.bzl << EOF
def symbol():
  a = ProtoInfo
EOF

  cat > BUILD << EOF
load(":symbol.bzl", "symbol")
symbol()
EOF
  bazel build --incompatible_load_symbols_externally=ProtoInfo :all >&$TEST_log 2>&1 || fail "build failed"
}


function test_existing_rule_is_redirected() {
  create_workspace_with_default_repos WORKSPACE

  cat > BUILD << EOF
py_library(
    name = 'py_library',
)
EOF
  bazel query --incompatible_load_rules_externally=py_library ':py_library' --output=build >&$TEST_log 2>&1 || fail "build failed"
  expect_log "__PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__"  # This is set only in rules_python, calling native.py_library still works
}

function test_existing_rule_is_redirected_in_bzl() {
  create_workspace_with_default_repos WORKSPACE

  cat > macro.bzl << EOF
def macro():
    native.py_library(
        name = 'py_library',
    )
EOF

  cat > BUILD << EOF
load(":macro.bzl", "macro")
macro()
EOF
  bazel query --incompatible_load_rules_externally=py_library ':py_library' --output=build >&$TEST_log 2>&1 || fail "build failed"
  expect_log "__PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__"  # This is set only in rules_python, calling native.py_library still works
}


function test_removed_rule_not_loaded() {
  create_workspace_with_default_repos WORKSPACE

  cat > BUILD << EOF
aar_import(
    name = 'aar',
    aar = 'aar.file',
    deps = [],
    visibility = ['//visibility:public'],
)
EOF

  bazel build --incompatible_load_rules_externally= :aar >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "name 'aar_import' is not defined"
}

function test_removed_rule_not_loaded_in_bzl() {
  create_workspace_with_default_repos WORKSPACE

  cat > macro.bzl << EOF
def macro():
    native.aar_import(
      name = 'aar',
      aar = 'aar.file',
      deps = [],
      visibility = ['//visibility:public'],
    )
EOF

  cat > BUILD << EOF
load(":macro.bzl", "macro")
macro()
EOF

  bazel build --incompatible_load_rules_externally= :aar >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "no native function or rule 'aar_import'"
}

# TODO: enable once we have a removed symbol
function disabled_test_removed_symbol_not_loaded_in_bzl() {
  create_workspace_with_default_repos WORKSPACE

  cat > symbol.bzl << EOF
def symbol():
    a = ProtoInfo
EOF

  cat > BUILD << EOF
load(":symbol.bzl", "symbol")
symbol()
EOF

  bazel build --incompatible_load_symbols_externally= :all >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "name 'ProtoInfo' is not defined"
}


function test_removing_existing_rule() {
  create_workspace_with_default_repos WORKSPACE

  cat > BUILD << EOF
android_binary(
    name = "bin",
    srcs = [
        "MainActivity.java",
        "Jni.java",
    ],
    manifest = "AndroidManifest.xml",
    deps = [
        ":lib",
        ":jni"
    ],
)
EOF

  bazel build --incompatible_load_rules_externally=-android_binary :bin >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "name 'android_binary' is not defined"
}

function test_removing_existing_rule_in_bzl() {
  create_workspace_with_default_repos WORKSPACE

  cat > macro.bzl << EOF
def macro():
    native.android_binary(
        name = "bin",
        srcs = [
            "MainActivity.java",
            "Jni.java",
        ],
        manifest = "AndroidManifest.xml",
        deps = [
            ":lib",
            ":jni"
        ],
    )
EOF

  cat > BUILD << EOF
load(":macro.bzl", "macro")
macro()
EOF

  bazel build --incompatible_load_rules_externally=-android_binary :bin >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "no native function or rule 'android_binary'"
}

function test_removing_existing_symbol() {
  create_workspace_with_default_repos WORKSPACE

  cat > symbol.bzl << EOF
def symbol():
   a = ProtoInfo
EOF

  cat > BUILD << EOF
load(":symbol.bzl", "symbol")
symbol()
EOF

  bazel build --incompatible_load_symbols_externally=-ProtoInfo :all >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "name 'ProtoInfo' is not defined"
}

# A design decision not to report errors (not to break users of --incompatible_load_symbols_externally=-ProtoInfo after the
# symbol is really removed)
function test_removing_symbol_typo() {
  create_workspace_with_default_repos WORKSPACE

  cat > bzl_file.bzl << EOF
def bzl_file():
    pass
EOF

  cat > BUILD << EOF
load(":bzl_file.bzl", "bzl_file")
EOF

  bazel build --incompatible_load_symbols_externally=-ProtozzzInfo :all >&$TEST_log 2>&1 || fail "build failed"
}

# A design decision not to report errors (not to break users of --incompatible_load_rules_externally=-android_binary after the
# rule is really removed)
function test_removing_rule_typo() {
  create_workspace_with_default_repos WORKSPACE

  touch BUILD

  bazel build --incompatible_load_rules_externally=-androidzzz_binary :all >&$TEST_log 2>&1 || fail "build failed"
}

function test_redirecting_rule_with_bzl_typo() {
  create_workspace_with_default_repos WORKSPACE

  # Bzl file is evaluated first, so this should cover bzl file support
  cat > bzl_file.bzl << EOF
def bzl_file():
    pass
EOF

  cat > BUILD << EOF
load(":bzl_file.bzl", "bzl_file")
EOF

  bazel build --incompatible_load_rules_externally=pyzzz_library :all >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "There's no load specification for toplevel symbol or rule set by --incompatible_load_symbols_externally or --incompatible_load_rules_externally. Most likely a typo."
}

function test_redirecting_rule_typo() {
  create_workspace_with_default_repos WORKSPACE

  cat > BUILD << EOF
EOF


  bazel build --incompatible_load_rules_externally=pyzzz_library :all >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "There's no load specification for toplevel symbol or rule set by --incompatible_load_symbols_externally or --incompatible_load_rules_externally. Most likely a typo."
}

function test_redirecting_symbols_typo() {
  create_workspace_with_default_repos WORKSPACE

  # Bzl file is evaluated first, so this should cover bzl file support
  cat > bzl_file.bzl << EOF
def bzl_file():
    pass
EOF

  cat > BUILD << EOF
load(":bzl_file.bzl", "bzl_file")
EOF

  bazel build --incompatible_load_symbols_externally=ProotoInfo :all >&$TEST_log 2>&1 && fail "build unexpectedly succeeded"
  expect_log "There's no load specification for toplevel symbol or rule set by --incompatible_load_symbols_externally or --incompatible_load_rules_externally. Most likely a typo."
}

function test_bad_flag_value() {
  create_workspace_with_default_repos WORKSPACE

  cat > BUILD << EOF
py_library(
    name = 'py_library',
)
EOF
  bazel query --incompatible_load_rules_externally=py_library,-py_library ':py_library' --output=build >&$TEST_log 2>&1 || fail "build failed"
  expect_log "__PYTHON_RULES_MIGRATION_DO_NOT_USE_WILL_BREAK__"  # This is set only in rules_python, calling native.py_library still works
}

run_suite "load_removed_symbols"
