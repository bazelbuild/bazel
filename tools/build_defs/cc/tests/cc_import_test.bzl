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
#
"""Tests for Starlark implementation of cc_import"""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts")
load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")
load("//tools/build_defs/cc:cc_import.bzl", "cc_import")

TAGS = ["manual", "nobuilder"]
TESTS_PATH = "tools/build_defs/cc/tests/"

def _generic_cc_import_test_setup(
        name,
        test,
        tests_list,
        binary_kwargs = dict(),
        target_is_binary = True,
        **kwargs):
    test_target_name = "cc_import_" + name + "_test"
    cc_import_target_name = test_target_name + "_import"
    cc_binary_target_name = test_target_name + "_binary"

    cc_import(
        name = cc_import_target_name,
        hdrs = ["mylib.h"],
        tags = TAGS,
        **kwargs
    )
    cc_binary(
        name = cc_binary_target_name,
        deps = [":" + cc_import_target_name],
        srcs = ["source.cc"],
        tags = TAGS,
        **binary_kwargs
    )

    target_under_test = ":" + cc_import_target_name
    if target_is_binary:
        target_under_test = ":" + cc_binary_target_name

    test(
        name = test_target_name,
        target_under_test = target_under_test,
        tags = TAGS,
        size = "small",
    )

    tests_list.append(":" + test_target_name)

def _assert_linkopt_present(env, linkopt):
    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)

    found = False
    for action in actions:
        if action.mnemonic == "CppLink":
            found = linkopt in action.argv

    asserts.true(env, found, "'" + linkopt + "' should be included in arguments passed to the linker")

def _cc_import_linkopts_test_impl(ctx):
    env = analysistest.begin(ctx)
    _assert_linkopt_present(env, "-testlinkopt")
    return analysistest.end(env)

cc_import_linkopts_test = analysistest.make(_cc_import_linkopts_test_impl)

def _cc_import_includes_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    includes = target_under_test[CcInfo].compilation_context.includes.to_list()

    found = "testinclude" in includes

    asserts.true(env, found, "'testinclude' should be present in the includes of the compilation context")

    return analysistest.end(env)

cc_import_includes_test = analysistest.make(_cc_import_includes_test_impl)

def _cc_import_objects_archive_action_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)

    found = False
    for action in actions:
        if action.mnemonic == "CppArchive":
            found = True

    asserts.true(env, found, "Archive creation action should be present if object files were given")

    linker_inputs = target_under_test[CcInfo].linking_context.linker_inputs.to_list()
    asserts.equals(env, 1, len(linker_inputs))

    libraries = linker_inputs[0].libraries
    asserts.equals(env, 1, len(libraries))

    pic_static_library = libraries[0].pic_static_library
    asserts.true(env, pic_static_library, "Pic static library should be defined")

    expected_name = "libcc_import_objects_archive_action_test_import.pic.a"
    asserts.equals(env, expected_name, pic_static_library.basename)

    return analysistest.end(env)

cc_import_objects_archive_action_test = analysistest.make(_cc_import_objects_archive_action_test_impl)

def _cc_import_objects_two_archive_actions_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)

    found = 0
    for action in actions:
        if action.mnemonic == "CppArchive":
            found += 1

    asserts.equals(env, 2, found)

    return analysistest.end(env)

cc_import_objects_two_archive_actions_test = analysistest.make(_cc_import_objects_two_archive_actions_test_impl)

def _cc_import_no_objects_no_archive_action_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)

    not_found = True
    for action in actions:
        if action.mnemonic == "CppArchive":
            not_found = False

    asserts.true(env, not_found, "Archive creation action should not be present if no object files were given")

    return analysistest.end(env)

cc_import_no_objects_no_archive_action_test = analysistest.make(_cc_import_no_objects_no_archive_action_test_impl)

def _cc_import_objects_present_in_linking_context_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    linker_inputs = target_under_test[CcInfo].linking_context.linker_inputs.to_list()
    asserts.equals(env, 1, len(linker_inputs))

    libraries = linker_inputs[0].libraries
    asserts.equals(env, 1, len(libraries))

    objects = libraries[0].objects
    asserts.equals(env, 1, len(objects))
    asserts.equals(env, "object.o", objects[0].basename)

    pic_objects = libraries[0].pic_objects
    asserts.equals(env, 1, len(pic_objects))
    asserts.equals(env, "object.pic.o", pic_objects[0].basename)

    static_library = libraries[0].static_library
    asserts.true(env, static_library)
    asserts.equals(env, "lib.a", static_library.basename)

    pic_static_library = libraries[0].pic_static_library
    asserts.true(env, pic_static_library)
    asserts.equals(env, "lib.pic.a", pic_static_library.basename)

    return analysistest.end(env)

cc_import_objects_present_in_linking_context_test = analysistest.make(_cc_import_objects_present_in_linking_context_test_impl)

def _cc_import_deps_test_setup(
        name,
        test,
        tests_list):
    test_target_name = "cc_import_" + name + "_test"
    cc_import_target_name = test_target_name + "_import"
    cc_import_second_target_name = test_target_name + "_second_import"
    cc_binary_target_name = test_target_name + "_binary"

    cc_import(
        name = cc_import_target_name,
        hdrs = ["mylib.h"],
        static_library = "libmy.a",
        deps = [":" + cc_import_second_target_name],
        tags = TAGS,
    )

    cc_import(
        name = cc_import_second_target_name,
        static_library = "libmy2.a",
        hdrs = ["mylib2.h"],
        tags = TAGS,
    )
    cc_binary(
        name = cc_binary_target_name,
        deps = [":" + cc_import_target_name],
        srcs = ["source.cc"],
        tags = TAGS,
    )

    target_under_test = ":" + cc_import_target_name

    test(
        name = test_target_name,
        target_under_test = target_under_test,
        tags = TAGS,
        size = "small",
    )

    tests_list.append(":" + test_target_name)

def _cc_import_deps_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    headers = target_under_test[CcInfo].compilation_context.headers.to_list()
    header_names = []
    for header in headers:
        header_names.append(header.basename)
    asserts.true(env, "mylib2.h" in header_names, "'mylib2.h' should be present in cc_import's headers")

    return analysistest.end(env)

cc_import_deps_test = analysistest.make(_cc_import_deps_test_impl)

def _cc_import_data_test_setup(
        name,
        test,
        tests_list):
    test_target_name = "cc_import_" + name + "_test"
    cc_import_target_name = test_target_name + "_import"
    cc_import_second_target_name = test_target_name + "_second_import"
    cc_binary_target_name = test_target_name + "_binary"

    cc_import(
        name = cc_import_target_name,
        static_library = "libmy.a",
        deps = [":" + cc_import_second_target_name],
        data = ["data1.file"],
        tags = TAGS,
    )

    cc_import(
        name = cc_import_second_target_name,
        static_library = "libmy2.a",
        data = ["data2.file"],
        tags = TAGS,
    )
    cc_binary(
        name = cc_binary_target_name,
        deps = [":" + cc_import_target_name],
        srcs = ["source.cc"],
        tags = TAGS,
    )

    target_under_test = ":" + cc_binary_target_name

    test(
        name = test_target_name,
        target_under_test = target_under_test,
        tags = TAGS,
        size = "small",
    )

    tests_list.append(":" + test_target_name)

def _cc_import_data_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    runfiles = target_under_test[DefaultInfo].default_runfiles.files.to_list()
    runfile_names = []
    for runfile in runfiles:
        runfile_names.append(runfile.basename)

    # We are not currently propagating runfiles in cc_import.
    # Tests are disabled until we do.
    # asserts.true(env, "data1.file" in runfile_names, "'data1.file' should be present runfiles")
    # asserts.true(env, "data2.file" in runfile_names, "'data2.file' should be present runfiles")

    return analysistest.end(env)

cc_import_data_test = analysistest.make(_cc_import_data_test_impl)

def cc_import_test_suite(name):
    _tests = []

    _generic_cc_import_test_setup(
        name = "linkopts",
        test = cc_import_linkopts_test,
        tests_list = _tests,
        static_library = "libmylib.a",
        linkopts = ["-testlinkopt"],
    )
    _generic_cc_import_test_setup(
        name = "includes",
        test = cc_import_includes_test,
        tests_list = _tests,
        static_library = "libmylib.a",
        includes = ["testinclude"],
    )
    _generic_cc_import_test_setup(
        name = "objects_archive_action",
        test = cc_import_objects_archive_action_test,
        tests_list = _tests,
        pic_objects = ["mylib.pic.o"],
        target_is_binary = False,
    )
    _generic_cc_import_test_setup(
        name = "objects_two_archive_actions",
        test = cc_import_objects_two_archive_actions_test,
        tests_list = _tests,
        pic_objects = ["mylib.pic.o"],
        objects = ["mylib.o"],
        target_is_binary = False,
    )
    _generic_cc_import_test_setup(
        name = "no_objects_no_archive_action",
        test = cc_import_no_objects_no_archive_action_test,
        tests_list = _tests,
        static_library = "libmylib.a",
        objects = ["object.o"],
        target_is_binary = False,
    )
    _generic_cc_import_test_setup(
        name = "objects_present_in_linking_context",
        test = cc_import_objects_present_in_linking_context_test,
        tests_list = _tests,
        objects = ["object.o"],
        pic_objects = ["object.pic.o"],
        static_library = "lib.a",
        pic_static_library = "lib.pic.a",
        target_is_binary = False,
    )
    _cc_import_deps_test_setup(
        name = "deps",
        test = cc_import_deps_test,
        tests_list = _tests,
    )
    _cc_import_data_test_setup(
        name = "data",
        test = cc_import_data_test,
        tests_list = _tests,
    )

    native.test_suite(
        name = name,
        tests = _tests,
        tags = TAGS,
    )
