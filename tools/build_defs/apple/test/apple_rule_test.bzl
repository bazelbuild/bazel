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

"""Tests for apple rules."""

load(
    "//tools/build_defs/apple:apple.bzl",
    "swift_library",
)
load(
    "//tools/build_rules:test_rules.bzl",
    "rule_test",
)


def apple_rule_test():
  """Issue simple tests on apple rules."""
  swift_library(
      name = "test_lib",
      module_name = "test_lib",
      srcs = ["source.swift"]
  )

  rule_test(
      name = "simple_swift_library_test",
      generates = ["test_lib/_objs/test_lib.a",
                   "test_lib/_objs/test_lib.swiftmodule",
                   "test_lib-Swift.h"],
      provides = {
          "swift": "",
          "objc": ""
      },
      rule = ":test_lib",
  )
