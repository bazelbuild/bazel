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

load(":bazel/java/bazel_java_binary.bzl", _java_test = "java_test", java_bin_exec = "java_binary")
load(":bazel/java/bazel_java_binary_nonexec.bzl", java_bin_nonexec = "java_binary")
load(":bazel/java/bazel_java_binary_deploy_jar.bzl", "deploy_jars", "deploy_jars_nonexec")
load(":common/java/java_binary_wrapper.bzl", "register_java_binary_rules", "register_legacy_java_binary_rules")
load(":common/java/java_semantics.bzl", "semantics")

def java_binary(**kwargs):
    if _builtins.internal.java_common_internal_do_not_use.incompatible_disable_non_executable_java_binary():
        register_java_binary_rules(
            java_bin_exec,
            rule_deploy_jars = deploy_jars,
            **kwargs
        )
    else:
        register_legacy_java_binary_rules(
            java_bin_exec,
            java_bin_nonexec,
            rule_deploy_jars = deploy_jars,
            rule_deploy_jars_nonexec = deploy_jars_nonexec,
            **kwargs
        )

def java_test(**kwargs):
    if "stamp" in kwargs and type(kwargs["stamp"]) == type(True):
        kwargs["stamp"] = 1 if kwargs["stamp"] else 0
    if "use_launcher" in kwargs and not kwargs["use_launcher"]:
        kwargs["launcher"] = None
    else:
        # If launcher is not set or None, set it to config flag
        if "launcher" not in kwargs or not kwargs["launcher"]:
            kwargs["launcher"] = semantics.LAUNCHER_FLAG_LABEL
    _java_test(**kwargs)
