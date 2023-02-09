# Copyright 2022 The Bazel Authors. All rights reserved.
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

"""Macro encapsulating the java_binary implementation

This is needed since the `executable` nature of the target must be computed from
the supplied value of the `create_executable` attribute.
"""

load(":bazel/java/bazel_java_binary.bzl", java_bin_exec = "java_binary")
load(":bazel/java/bazel_java_binary_nolauncher.bzl", java_bin_exec_no_launcher_flag = "java_binary")
load(":bazel/java/bazel_java_binary_nonexec.bzl", java_bin_nonexec = "java_binary")
load(":bazel/java/bazel_java_binary_deploy_jar.bzl", "deploy_jars")
load(":common/java/java_binary_wrapper.bzl", "register_java_binary_rules")

# TODO(hvd): add custom launcher rule if needed
def java_binary(**kwargs):
    register_java_binary_rules(
        java_bin_exec,
        java_bin_nonexec,
        java_bin_exec_no_launcher_flag,
        java_bin_exec,
        deploy_jars,
        **kwargs
    )
