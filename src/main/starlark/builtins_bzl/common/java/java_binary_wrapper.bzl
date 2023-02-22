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

_DEPLOY_JAR_RULE_NAME_SUFFIX = "_deployjars_internal_rule"

def register_java_binary_rules(rule_exec, rule_nonexec, rule_nolauncher, rule_customlauncher, rule_deploy_jars = None, is_test_rule_class = False, **kwargs):
    """Registers the correct java_binary rule and deploy jar rule

    Args:
        rule_exec: (Rule) The executable java_binary rule
        rule_nonexec: (Rule) The non-executable java_binary rule
        rule_nolauncher: (Rule) The executable java_binary rule without launcher flag resolution
        rule_customlauncher: (Rule) The executable java_binary rule with a custom launcher attr set
        rule_deploy_jars: (Rule) The auxiliary deploy jars rule
        is_test_rule_class: (bool) If this is a test rule
        **kwargs: Actual args to instantiate the rule
    """

    # TODO(hvd): migrate depot to integers / maybe use decompose_select_list()
    if "stamp" in kwargs and type(kwargs["stamp"]) == type(True):
        kwargs["stamp"] = 1 if kwargs["stamp"] else 0
    if "create_executable" in kwargs and not kwargs["create_executable"]:
        rule_nonexec(**kwargs)
    elif "use_launcher" in kwargs and not kwargs["use_launcher"]:
        rule_nolauncher(**kwargs)
    elif "launcher" in kwargs and type(kwargs["launcher"]) == type(""):
        rule_customlauncher(**kwargs)
    else:
        rule_exec(**kwargs)

    if rule_deploy_jars and (
        not kwargs.get("tags", []) or "nodeployjar" not in kwargs.get("tags", [])
    ):
        deploy_jar_args = _filtered_dict(kwargs, _DEPLOY_JAR_RULE_ATTRS)
        if is_test_rule_class:
            deploy_jar_args["testonly"] = True

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
