# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""Starlark tests for cc_static_library"""

load("@rules_testing//lib:analysis_test.bzl", "analysis_test", "test_suite")
load("@rules_testing//lib:util.bzl", "util")
load(":mock_toolchain.bzl", "mock_cc_toolchain")

def _set_up_subject(name):
    util.helper_target(
        native.cc_import,
        name = name + "_dynamic_import",
        shared_library = "mylib.dll",
    )
    util.helper_target(
        native.cc_import,
        name = name + "_interface_import",
        interface_library = "mylib.lib",
        shared_library = "mylib.dll",
    )
    util.helper_target(
        native.cc_import,
        name = name + "_static_import",
        static_library = "mylib.lib",
    )
    util.helper_target(
        native.cc_import,
        name = name + "_system_import",
        interface_library = "mylib.lib",
        system_provided = True,
    )
    util.helper_target(
        native.cc_library,
        name = name + "_imports",
        deps = [
            name + "_dynamic_import",
            name + "_interface_import",
            name + "_static_import",
            name + "_system_import",
        ],
    )
    util.helper_target(
        native.cc_library,
        name = name + "_dep_1",
        srcs = ["file.cc"],
        linkopts = [
            "dep_1_arg_1",
            "dep_1_arg_2",
            "dep_1_arg_1",
        ],
    )
    util.helper_target(
        native.cc_library,
        name = name + "_linkopts_only",
        linkopts = [
            "linkopts_only_arg_1",
            "linkopts_only_arg_2",
            "linkopts_only_arg_1",
        ],
        deps = [
            name + "_duplicate_linkopts",
        ],
    )
    util.helper_target(
        native.cc_library,
        name = name + "_duplicate_linkopts",
        linkopts = [
            "linkopts_only_arg_1",
            "linkopts_only_arg_2",
            "linkopts_only_arg_1",
        ],
    )
    util.helper_target(
        native.cc_static_library,
        name = name + "_subject",
        deps = [
            name + "_dep_1",
            name + "_linkopts_only",
            name + "_imports",
        ],
    )

def _test_default_outputs(name):
    _set_up_subject(name)
    mock_cc_toolchain(name + "_toolchain")
    analysis_test(
        name = name,
        impl = _test_default_outputs_impl,
        target = name + "_subject",
        config_settings = {
            "//command_line_option:extra_toolchains": [str(native.package_relative_label(name + "_toolchain"))],
            "//command_line_option:action_env": ["PATH=/usr/bin/special"],
        },
    )

def _test_default_outputs_impl(env, target):
    static_lib_path = target.label.package + "/" + "prefix" + target.label.name + ".lib"

    env.expect.that_target(target).default_outputs().contains_exactly([static_lib_path])

    action = env.expect.that_target(target).action_generating(static_lib_path)
    action.mnemonic().equals("CppTransitiveArchive")

    action_env = action.env()
    action_env.contains_at_least({
        "MY_KEY": "my_value",
        "PATH": "/usr/bin/special",
    })

    argv = action.argv()
    argv.contains_at_least([
        "/usr/bin/my-ar",
        "abc",
    ]).in_order()
    argv.contains_at_least([
        file.path
        for file in action.actual.inputs.to_list()
        if file.basename.endswith(".o")
    ]).in_order()
    argv.contains("--check-symbols")

def _test_output_groups(name):
    _set_up_subject(name)
    analysis_test(
        name = name,
        impl = _test_output_groups_impl,
        target = name + "_subject",
    )

def _test_output_groups_impl(env, target):
    path_prefix = target.label.package + "/" + target.label.name
    base_label = "//" + target.label.package + ":" + target.label.name.removesuffix("_subject")
    subject = env.expect.that_target(target)

    subject.output_group("linkdeps").contains_exactly([path_prefix + "_linkdeps.txt"])
    subject.action_generating(path_prefix + "_linkdeps.txt").content().split("\n").contains_exactly([
        base_label + "_system_import",
        base_label + "_static_import",
        base_label + "_interface_import",
        base_label + "_dynamic_import",
        "",
    ]).in_order()

    subject.output_group("linkopts").contains_exactly([path_prefix + "_linkopts.txt"])
    subject.action_generating(path_prefix + "_linkopts.txt").content().split("\n").contains_exactly([
        "dep_1_arg_1",
        "dep_1_arg_2",
        "dep_1_arg_1",
        "linkopts_only_arg_1",
        "linkopts_only_arg_2",
        "linkopts_only_arg_1",
        "linkopts_only_arg_1",
        "linkopts_only_arg_2",
        "linkopts_only_arg_1",
        "",
    ]).in_order()

def _test_validation_enabled(name):
    _set_up_subject(name)
    mock_cc_toolchain(name + "_toolchain", provide_validate_static_library = True)
    analysis_test(
        name = name,
        impl = _test_validation_enabled_impl,
        target = name + "_subject",
        config_settings = {
            "//command_line_option:extra_toolchains": [str(native.package_relative_label(name + "_toolchain"))],
            "//command_line_option:action_env": ["PATH=/usr/bin/special"],
        },
    )

def _test_validation_enabled_impl(env, target):
    action = env.expect.that_target(target).action_named("ValidateStaticLibrary")

    outputs = action.actual.outputs.to_list()
    env.expect.that_collection(outputs).has_size(1)
    validation_output = outputs[0]

    action.env().contains_at_least({"PATH": "/usr/bin/special"})

    static_lib = target[DefaultInfo].files.to_list()[0]
    action.argv().contains_exactly([
        target.label.package + "/validate_static_library.sh",
        static_lib.path,
        validation_output.path,
    ]).in_order()

    env.expect.that_target(target).output_group("_validation").contains_exactly(
        [validation_output.short_path],
    )

def _test_validation_disabled(name):
    _set_up_subject(name)
    mock_cc_toolchain(name + "_toolchain", provide_validate_static_library = False)
    analysis_test(
        name = name,
        impl = _test_validation_disabled_impl,
        target = name + "_subject",
        config_settings = {
            "//command_line_option:extra_toolchains": [str(native.package_relative_label(name + "_toolchain"))],
        },
    )

def _test_validation_disabled_impl(env, target):
    env.expect.that_collection(target.actions).transform(
        "mnemonics",
        map_each = lambda a: a.mnemonic,
    ).contains_none_of(["ValidateStaticLibrary"])

    env.expect.that_collection(dir(target[OutputGroupInfo])).contains_none_of(["_validation"])

def analysis_test_suite(name):
    test_suite(
        name = name,
        tests = [
            _test_default_outputs,
            _test_output_groups,
            _test_validation_enabled,
            _test_validation_disabled,
        ],
    )
