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

load(":common/java/java_semantics.bzl", "semantics")

def register_legacy_java_binary_rules(
        rule_exec,
        rule_nonexec,
        **kwargs):
    """Registers the correct java_binary rule and deploy jar rule

    Args:
        rule_exec: (Rule) The executable java_binary rule
        rule_nonexec: (Rule) The non-executable java_binary rule
        **kwargs: Actual args to instantiate the rule
    """

    create_executable = "create_executable" not in kwargs or kwargs["create_executable"]

    # TODO(hvd): migrate depot to integers / maybe use decompose_select_list()
    if "stamp" in kwargs and type(kwargs["stamp"]) == type(True):
        kwargs["stamp"] = 1 if kwargs["stamp"] else 0
    if not create_executable:
        rule_nonexec(**kwargs)
    else:
        if "use_launcher" in kwargs and not kwargs["use_launcher"]:
            kwargs["launcher"] = None
        else:
            # If launcher is not set or None, set it to config flag
            if "launcher" not in kwargs or not kwargs["launcher"]:
                kwargs["launcher"] = semantics.LAUNCHER_FLAG_LABEL
        rule_exec(**kwargs)

def register_java_binary_rules(
        java_binary,
        **kwargs):
    """Creates a java_binary rule and a deploy jar rule

    Args:
        java_binary: (Rule) The executable java_binary rule
        **kwargs: Actual args to instantiate the rule
    """

    # TODO(hvd): migrate depot to integers / maybe use decompose_select_list()
    if "stamp" in kwargs and type(kwargs["stamp"]) == type(True):
        kwargs["stamp"] = 1 if kwargs["stamp"] else 0

    if "use_launcher" in kwargs and not kwargs["use_launcher"]:
        kwargs["launcher"] = None
    else:
        # If launcher is not set or None, set it to config flag
        if "launcher" not in kwargs or not kwargs["launcher"]:
            kwargs["launcher"] = semantics.LAUNCHER_FLAG_LABEL
    java_binary(**kwargs)
