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

def _generic_cc_import_test_setup(name, test, **kwargs):
    test_target_name = "cc_import_" + name + "_test"
    cc_import_target_name = test_target_name + "_import"
    cc_binary_target_name = test_target_name + "_binary"

    cc_import(
        name = cc_import_target_name,
        hdrs = ["mylib.h"],
        static_library = "libmylib.a",
        tags = TAGS,
        **kwargs
    )

    native.cc_binary(
        name = cc_binary_target_name,
        deps = [":" + cc_import_target_name],
        srcs = ["source.cc"],
        tags = TAGS,
    )

    test(
        name = test_target_name,
        target_under_test = ":" + cc_binary_target_name,
        tags = TAGS,
        size = "small",
    )

def _cc_import_linkopts_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)

    found = False
    for action in actions:
        if action.mnemonic == "CppLink":
            found = "-testlinkopt" in action.argv
    asserts.true(env, found, "'-testlinkopt' should be included in arguments passed to the linker")

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

def cc_import_test_suite(name):
    _generic_cc_import_test_setup(name = "linkopts", test = cc_import_linkopts_test, linkopts = ["-testlinkopt"])
    _generic_cc_import_test_setup(name = "includes", test = cc_import_includes_test, includes = ["testinclude"])

    native.test_suite(
        name = name,
        tests = [
            ":cc_import_linkopts_test",
            ":cc_import_includes_test",
        ],
        tags = TAGS,
    )
