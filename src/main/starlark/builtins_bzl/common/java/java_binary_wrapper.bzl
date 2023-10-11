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

_DEPLOY_JAR_RULE_NAME_SUFFIX = "_deployjars_internal_rule"

def register_legacy_java_binary_rules(
        rule_exec,
        rule_nonexec,
        rule_deploy_jars = None,
        rule_deploy_jars_nonexec = None,
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

    if not create_executable:
        rule_deploy_jars = rule_deploy_jars_nonexec
    if rule_deploy_jars and (
        not kwargs.get("tags", []) or "nodeployjar" not in kwargs.get("tags", [])
    ):
        deploy_jar_args = _filtered_dict(kwargs, _DEPLOY_JAR_RULE_ATTRS)

        # Do not let the deploy jar be matched by wildcard target patterns.
        if "tags" not in deploy_jar_args or not deploy_jar_args["tags"]:
            deploy_jar_args["tags"] = []
        if "manual" not in deploy_jar_args["tags"]:
            tags = []
            tags.extend(deploy_jar_args["tags"])
            tags.append("manual")
            deploy_jar_args["tags"] = tags
        rule_deploy_jars(
            name = kwargs["name"] + _DEPLOY_JAR_RULE_NAME_SUFFIX,  # to avoid collision
            binary = kwargs["name"],
            **deploy_jar_args
        )

def register_java_binary_rules(
        java_binary,
        rule_deploy_jars = None,
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

    if rule_deploy_jars and (
        not kwargs.get("tags", []) or "nodeployjar" not in kwargs.get("tags", [])
    ):
        deploy_jar_args = _filtered_dict(kwargs, _DEPLOY_JAR_RULE_ATTRS)

        # Do not let the deploy jar be matched by wildcard target patterns.
        if "tags" not in deploy_jar_args or not deploy_jar_args["tags"]:
            deploy_jar_args["tags"] = []
        if "manual" not in deploy_jar_args["tags"]:
            tags = []
            tags.extend(deploy_jar_args["tags"])
            tags.append("manual")
            deploy_jar_args["tags"] = tags
        rule_deploy_jars(
            name = kwargs["name"] + _DEPLOY_JAR_RULE_NAME_SUFFIX,  # to avoid collision
            binary = kwargs["name"],
            **deploy_jar_args
        )

_DEPLOY_JAR_RULE_ATTRS = {key: None for key in [
    "visibility",
    "testonly",
    "tags",
    "compatible_with",
    "target_compatible_with",
]}

def _filtered_dict(input_dict, select_keys):
    res = {}
    if len(input_dict) > len(select_keys):
        for key in select_keys:
            if key in input_dict:
                res[key] = input_dict[key]
    else:
        for key in input_dict:
            if key in select_keys:
                res[key] = input_dict[key]
    return res
