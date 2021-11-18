# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Exported builtins symbols that are not specific to OSS Bazel."""

load("@_builtins//:common/cc/cc_import.bzl", "cc_import")
load("@_builtins//:common/cc/cc_binary.bzl", "cc_binary")
load("@_builtins//:common/java/java_library_macro.bzl", "java_library")
load("@_builtins//:common/java/java_plugin.bzl", "java_plugin")
load("@_builtins//:common/cc/experimental_cc_shared_library.bzl", "cc_shared_library", "cc_shared_library_permissions")
load("@_builtins//:common/objc/objc_import.bzl", "objc_import")
load("@_builtins//:common/objc/objc_library.bzl", "objc_library")
load("@_builtins//:common/objc/apple_static_library.bzl", "apple_static_library")
load("@_builtins//:common/objc/compilation_support.bzl", "compilation_support")
load("@_builtins//:common/proto/proto_library.bzl", "proto_library")

exported_toplevels = {
    # This dummy symbol is not part of the public API; it is only used to test
    # that builtins injection is working properly. Its built-in value is
    # "original value".
    "_builtins_dummy": "overridden value",
}
exported_rules = {
    "+cc_import": cc_import,
    "+java_library": java_library,
    "+java_plugin": java_plugin,
    "objc_import": objc_import,
    "objc_library": objc_library,
    "-proto_library": proto_library,
    "+apple_static_library": apple_static_library,
    "+cc_shared_library": cc_shared_library,
    "+cc_shared_library_permissions": cc_shared_library_permissions,
    "-cc_binary": cc_binary,
}
exported_to_java = {
    "register_compile_and_archive_actions_for_j2objc": compilation_support.register_compile_and_archive_actions_for_j2objc,
}
