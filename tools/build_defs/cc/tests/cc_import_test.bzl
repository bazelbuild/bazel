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

    native.cc_binary(
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

    return analysistest.end(env)

cc_import_objects_archive_action_test = analysistest.make(_cc_import_objects_archive_action_test_impl)

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
    asserts.true(env, len(linker_inputs) == 1, "There should be 1 linker input")
    libraries = linker_inputs[0].libraries
    asserts.true(env, len(libraries) == 1, "There should be 1 library to link")
    objects = libraries[0].objects
    asserts.true(env, len(objects) == 1, "There should be 1 object file")
    asserts.true(env, objects[0].basename == "object.o", "Object's name should be 'object.o'")

    return analysistest.end(env)

cc_import_objects_present_in_linking_context_test = analysistest.make(_cc_import_objects_present_in_linking_context_test_impl)

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
        target_is_binary = False,
    )

    native.test_suite(
        name = name,
        tests = _tests,
        tags = TAGS,
    )
