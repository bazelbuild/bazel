# Copyright 2023 The Bazel Authors. All rights reserved.
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
"""Objective-C"""

load("@rules_cc//cc/private/rules_impl:objc_import.bzl", "objc_import")  # buildifier: disable=bzl-visibility
load("@rules_cc//cc/private/rules_impl:objc_library.bzl", "objc_library")  # buildifier: disable=bzl-visibility

# Build Encyclopedia entry point for Objc rules implemented in Starlark in Blaze's @_builtins

binary_rules = struct()

library_rules = struct(
    objc_library = objc_library,
    objc_import = objc_import,
)

test_rules = struct()

other_rules = struct()
