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

load("@_builtins//:common/python/py_internal.bzl", "py_internal")
load(":common/java/java_package_configuration.bzl", "java_package_configuration")
load(":common/java/java_runtime.bzl", "java_runtime")
load(":common/java/java_toolchain.bzl", "java_toolchain")

exported_toplevels = {
    "py_internal": py_internal,
}
exported_rules = {
    "java_package_configuration": java_package_configuration,
    "java_runtime": java_runtime,
    "java_toolchain": java_toolchain,
}
exported_to_java = {}
