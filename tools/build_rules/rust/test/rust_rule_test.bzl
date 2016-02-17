# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""Tests for rust rules."""

load(
    "//tools/build_rules/rust:rust.bzl",
    "rust_library",
    "rust_binary",
    "rust_test",
)
load(
    "//tools/build_rules:test_rules.bzl",
    "rule_test",
)

def _rust_library_test(package):
  rule_test(
      name ="hello_lib_rule_test",
      generates = ["libhello_lib.rlib"],
      provides = {
          "rust_lib": "/libhello_lib.rlib$",
          "transitive_libs": "^\\[\\]$"
      },
      rule = package + "/hello_lib:hello_lib",
  )

def _rust_binary_test(package):
  rule_test(
      name = "hello_world_rule_test",
      generates = ["hello_world"],
      rule = package + "/hello_world:hello_world",
  )

def _rust_test_test(package):
  """Issue rule tests for rust_test."""
  rule_test(
      name = "greeting_rule_test",
      generates = ["greeting"],
      rule = package + "/hello_lib:greeting",
  )

def rust_rule_test(package):
  """Issue simple tests on rust rules."""
  _rust_library_test(package)
  _rust_binary_test(package)
  _rust_test_test(package)
