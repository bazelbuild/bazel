# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""A collection of macros that retrieve targets from the remote java tools."""

JAVAC_VERSION_TO_TARGET_SUFFIX = {
    "javac9": "_javac9",
    "javac10": ""
}

def _get_args(target, attr, javac_version, **kwargs):
    if not javac_version:
        # The default javac version in Bazel is 10.
        javac_version = "javac10"
    workspace_target_dict = {
        "//src/conditions:linux_x86_64": ["@remote_java_tools_"+ javac_version + "_linux//" + target],
        "//src/conditions:darwin": ["@remote_java_tools_" + javac_version + "_darwin//" + target],
        "//src/conditions:darwin_x86_64": ["@remote_java_tools_" + javac_version + "_darwin//" + target],
        "//src/conditions:windows": ["@remote_java_tools_" + javac_version + "_windows//" + target],
        # On different platforms the linux repository can be used.
        # The deploy jars inside the linux repository are platform-agnostic.
        # The ijar target inside the repository identifies the different
        # platform and builds ijar from source instead of returning the
        # precompiled binary.
        "//conditions:default": ["@remote_java_tools_" + javac_version + "_linux//" + target],
    }
    workspace_target_select = select(workspace_target_dict)
    args = dict({attr: workspace_target_select})
    args.update(kwargs)
    return args

def remote_java_tools_filegroup(name, target, name_suffix = "", **kwargs):
    for javac_version, javac_in_name in JAVAC_VERSION_TO_TARGET_SUFFIX.items():
        args = _get_args(target, "srcs", javac_version, **kwargs)
        native.filegroup(name = name + javac_in_name + name_suffix, **args)

def remote_java_tools_java_import(name, target, name_suffix = "", **kwargs):
    for javac_version, javac_in_name in JAVAC_VERSION_TO_TARGET_SUFFIX.items():
        args = _get_args(target, "jars", javac_version, **kwargs)
        native.java_import(name = name + javac_in_name + name_suffix, **args)
