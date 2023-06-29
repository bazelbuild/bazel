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

"""Exported builtins symbols that are specific to OSS Bazel."""

load("@_builtins//:common/java/java_library.bzl", "JAVA_LIBRARY_ATTRS", "bazel_java_library_rule", "java_library")
load("@_builtins//:common/java/java_plugin.bzl", "java_plugin")
load("@_builtins//:common/java/java_import.bzl", "java_import")
load("@_builtins//:common/java/proto/java_proto_library.bzl", "java_proto_library")
load("@_builtins//:common/cc/cc_proto_library.bzl", "cc_proto_aspect", "cc_proto_library")
load(":bazel/java/bazel_java_binary_wrapper.bzl", "java_binary", "java_test")
load("@_builtins//:common/python/py_binary_macro.bzl", "py_binary")
load("@_builtins//:common/python/py_library_macro.bzl", "py_library")
load("@_builtins//:common/python/py_test_macro.bzl", "py_test")

exported_toplevels = {
    # This is an experimental export in Bazel. The interface will change in a way
    # that breaks users. In the future, Build API team will provide an interface
    # that is conceptually similar to this one and stable.
    "experimental_java_library_export_do_not_use": struct(
        bazel_java_library_rule = bazel_java_library_rule,
        JAVA_LIBRARY_ATTRS = JAVA_LIBRARY_ATTRS,
    ),
    "cc_proto_aspect": cc_proto_aspect,
}
exported_rules = {
    "java_library": java_library,
    "java_plugin": java_plugin,
    "+java_import": java_import,
    "java_proto_library": java_proto_library,
    "+cc_proto_library": cc_proto_library,
    "java_binary": java_binary,
    "java_test": java_test,
    "py_binary": py_binary,
    "py_test": py_test,
    "py_library": py_library,
}
exported_to_java = {}
