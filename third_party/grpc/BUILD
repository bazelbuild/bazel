# gRPC Bazel BUILD file.
#
# Copyright 2016 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@rules_java//java:defs.bzl", "java_import")

licenses(["notice"])  # Apache v2

exports_files(["grpc_1.26.0.patch"])

package(
    default_visibility = ["//visibility:public"],
    features = [
        "-layering_check",
        "-parse_headers",
    ],
)

load(
    ":build_defs.bzl",
    "grpc_cc_library",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//third_party/grpc/bazel:srcs",
    ],
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)

filegroup(
    name = "bootstrap-grpc-jars",
    srcs = [
        "grpc-api-1.26.0.jar",
        "grpc-auth-1.26.0.jar",
        "grpc-context-1.26.0.jar",
        "grpc-core-1.26.0.jar",
        "grpc-netty-1.26.0.jar",
        "grpc-protobuf-1.26.0.jar",
        "grpc-protobuf-lite-1.26.0.jar",
        "grpc-stub-1.26.0.jar",
    ],
)

java_import(
    name = "grpc-jar",
    jars = [":bootstrap-grpc-jars"],
    runtime_deps = [
        "//third_party:netty",
        "//third_party:opencensus-api",
        "//third_party:perfmark-api",
    ],
    deps = [
        "//third_party:guava",
    ],
)

cc_binary(
    name = "grpc-java-plugin",
    srcs = [
        "compiler/src/java_plugin/cpp/java_generator.cpp",
        "compiler/src/java_plugin/cpp/java_generator.h",
        "compiler/src/java_plugin/cpp/java_plugin.cpp",
    ],
    copts = ["-w"],
    deps = ["//third_party/protobuf:protoc_lib"],
)

alias(
    name = "cpp_plugin",
    actual = "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
)

alias(
    name = "grpc++_codegen_proto",
    actual = "@com_github_grpc_grpc//:grpc++_codegen_proto",
)

alias(
    name = "grpc++_unsecure",
    actual = "@com_github_grpc_grpc//:grpc++_unsecure",
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = [
        "BUILD.tools",
        "build_defs.bzl",
    ],
)
