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

load("//tools/python:private/py_test_alias.bzl", "py_test_alias")
load("//tools/python:private/defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

py_library(
    name = "aquery_differ_resolvers",
    srcs = glob(
        ["resolvers/*.py"],
    ),
)

py_binary(
    name = "aquery_differ",
    srcs = ["aquery_differ.py"],
    python_version = "PY3",
    deps = [
        ":aquery_differ_main_lib",
        "//third_party/py/six",
    ],
)

py_library(
    name = "aquery_differ_main_lib",
    srcs = ["aquery_differ.py"],
    deps = [
        ":aquery_differ_resolvers",
        "//src/main/protobuf:analysis_py_proto",
        "//third_party/py/abseil",
        "//third_party/py/six",
    ],
)

py_test_alias(
    name = "aquery_differ_test",
    srcs = ["aquery_differ_test.py"],
    deps = [
        ":aquery_differ_main_lib",
        "//src/main/protobuf:analysis_py_proto",
        "//third_party/py/mock",
        "//third_party/py/six",
    ],
)

py_binary(
    name = "aquery_differ_v2",
    srcs = ["aquery_differ_v2.py"],
    python_version = "PY3",
    deps = [
        ":aquery_differ_main_lib_v2",
        "//third_party/py/six",
    ],
)

py_library(
    name = "aquery_differ_main_lib_v2",
    srcs = ["aquery_differ_v2.py"],
    deps = [
        ":aquery_differ_resolvers",
        "//src/main/protobuf:analysis_v2_py_proto",
        "//third_party/py/abseil",
        "//third_party/py/six",
    ],
)

py_test_alias(
    name = "aquery_differ_v2_test",
    srcs = ["aquery_differ_v2_test.py"],
    deps = [
        ":aquery_differ_main_lib_v2",
        "//src/main/protobuf:analysis_v2_py_proto",
        "//third_party/py/mock",
        "//third_party/py/six",
    ],
)
