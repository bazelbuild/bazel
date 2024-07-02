# Copyright 2018 The Bazel Authors. All rights reserved.
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(glob(["*.bzl"]))

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/cpp/modules_tools:srcs",
        "//tools/cpp/runfiles:srcs",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/cpp/modules_tools:embedded_tools",
        "//tools/cpp/runfiles:embedded_tools",
    ],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
)

cc_test(
    name = "bazel_build_info_translation_helper_test",
    srcs = ["bazel_build_info_translation_helper_test.cc"],
    data = [
        ":test_data/correct_file_format.txt",
        ":test_data/duplicated_key.txt",
        ":test_data/no_separator.txt",
    ],
    deps = [
        ":bazel_build_info_translation_helper",
        "@com_google_googletest//:gtest_main",
    ],
)
