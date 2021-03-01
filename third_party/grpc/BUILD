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

load("//tools/distributions:distribution_rules.bzl", "distrib_java_import", "distrib_jar_filegroup", "distrib_cc_binary")

licenses(["notice"])  # Apache v2

exports_files(["grpc_1.33.1.patch"])

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//third_party/grpc/bazel:srcs",
    ],
)

distrib_jar_filegroup(
    name = "bootstrap-grpc-jars",
    srcs = [
        "grpc-api-1.33.1.jar",
        "grpc-auth-1.33.1.jar",
        "grpc-context-1.33.1.jar",
        "grpc-core-1.33.1.jar",
        "grpc-netty-1.33.1.jar",
        "grpc-protobuf-1.33.1.jar",
        "grpc-protobuf-lite-1.33.1.jar",
        "grpc-stub-1.33.1.jar",
    ],
    enable_distributions = ["debian"],
)

distrib_java_import(
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
    enable_distributions = ["debian"],
)

distrib_cc_binary(
    name = "grpc-java-plugin",
    enable_distributions = ["debian"],
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
    actual = select({
        "//src/conditions:debian_build": "@debian_bin_deps//:grpc-cpp-plugin",
        "//conditions:default": "@com_github_grpc_grpc//src/compiler:grpc_cpp_plugin",
    }),
)

alias(
    name = "grpc++_codegen_proto",
    actual = select({
        "//src/conditions:debian_build": "@debian_cc_deps//:grpc++_unsecure",
        "//conditions:default": "@com_github_grpc_grpc//:grpc++_codegen_proto",
    }),
)

alias(
    name = "grpc++_unsecure",
    actual = select({
        "//src/conditions:debian_build": "@debian_cc_deps//:grpc++_unsecure",
        "//conditions:default": "@com_github_grpc_grpc//:grpc++_unsecure",
    }),
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = [
        "BUILD.tools",
        "build_defs.bzl",
    ],
)
