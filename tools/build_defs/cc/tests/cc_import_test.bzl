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

def _cc_import_linkopts_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    actions = analysistest.target_actions(env)

    found = False
    for action in actions:
        if action.mnemonic == "CppLink":
            for arg in action.argv:
                if arg.find("-testlinkopt") != -1:
                    found = True
                    break
    asserts.true(env, found, "'-testlinkopt' should be included in arguments passed to linked")

    return analysistest.end(env)

cc_import_linkopts_test = analysistest.make(_cc_import_linkopts_test_impl)

def _test_cc_import_linkopts():
    cc_import(
        name = "cc_import_linkopts_test_import",
        linkopts = ["-testlinkopt"],
        hdrs = ["mylib.h"],
        static_library = "libmylib.a",
        tags = TAGS,
    )

    native.cc_binary(
        name = "cc_import_linkopts_test_binary",
        deps = [":cc_import_linkopts_test_import"],
        srcs = ["source.cc"],
        tags = TAGS,
    )

    cc_import_linkopts_test(
        name = "cc_import_linkopts_test",
        target_under_test = ":cc_import_linkopts_test_binary",
        tags = TAGS,
    )

def cc_import_test_suite(name):
    _test_cc_import_linkopts()

    native.test_suite(
        name = name,
        tests = [
            ":cc_import_linkopts_test",
        ],
        tags = TAGS,
    )
