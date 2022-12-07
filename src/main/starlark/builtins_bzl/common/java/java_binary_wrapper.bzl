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

load(":common/java/java_binary_deploy_jar.bzl", "DEPLOY_JAR_RULE_NAME_SUFFIX")

def register_java_binary_rules(rule_exec, rule_nonexec, rule_nolauncher, rule_deploy_jars, **kwargs):
    """Registers the correct java_binary rule and deploy jar rule

    Args:
        rule_exec: (Rule) The executable java_binary rule
        rule_nonexec: (Rule) The non-executable java_binary rule
        rule_nolauncher: (Rule) The executable java_binary rule without launcher flag resolution
        rule_deploy_jars: (Rule) The auxiliary deploy jars rule
        **kwargs: Actual args to instantiate the rule
    """

    # TODO(hvd): migrate depot to integers / maybe use decompose_select_list()
    if "stamp" in kwargs and type(kwargs["stamp"]) == type(True):
        kwargs["stamp"] = 1 if kwargs["stamp"] else 0
    if "create_executable" in kwargs and not kwargs["create_executable"]:
        rule_nonexec(**kwargs)
    elif ("launcher" in kwargs and kwargs["launcher"]) or ("use_launcher" in kwargs and not kwargs["use_launcher"]):
        rule_nolauncher(**kwargs)
    else:
        rule_exec(**kwargs)

    if "nodeployjar" not in kwargs.get("tags", []):
        rule_deploy_jars(
            name = kwargs["name"] + DEPLOY_JAR_RULE_NAME_SUFFIX,  # to avoid collision
            binary = kwargs["name"],
            **_filtered_dict(kwargs, _DEPLOY_JAR_RULE_ATTRS)
        )

_DEPLOY_JAR_RULE_ATTRS = {key: None for key in [
    "stamp",
    "deploy_manifest_lines",
    "visibility",
    "testonly",
    "tags",
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
