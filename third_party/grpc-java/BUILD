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

load("@rules_license//rules:license.bzl", "license")
load("//tools/distributions:distribution_rules.bzl", "distrib_cc_binary", "distrib_jar_filegroup", "distrib_java_import")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache v2

license(
    name = "license",
    package_name = "grpc-java",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
    package_version = "1.45",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

distrib_jar_filegroup(
    name = "bootstrap-grpc-jars",
    srcs = [
        "grpc-api-1.45.1.jar",
        "grpc-api-1.47.0.jar",
        "grpc-auth-1.45.1.jar",
        "grpc-auth-1.47.0.jar",
        "grpc-context-1.45.1.jar",
        "grpc-context-1.47.0.jar",
        "grpc-core-1.45.1.jar",
        "grpc-core-1.47.0.jar",
        "grpc-netty-1.45.1.jar",
        "grpc-netty-1.47.0.jar",
        "grpc-protobuf-1.45.1.jar",
        "grpc-protobuf-1.47.0.jar",
        "grpc-protobuf-lite-1.45.1.jar",
        "grpc-protobuf-lite-1.47.0.jar",
        "grpc-stub-1.45.1.jar",
        "grpc-stub-1.47.0.jar",
    ],
    enable_distributions = ["debian"],
)

distrib_java_import(
    name = "grpc-jar",
    enable_distributions = ["debian"],
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

distrib_cc_binary(
    name = "grpc-java-plugin",
    srcs = [
        "compiler/src/java_plugin/cpp/java_generator.cpp",
        "compiler/src/java_plugin/cpp/java_generator.h",
        "compiler/src/java_plugin/cpp/java_plugin.cpp",
    ],
    copts = ["-w"],
    enable_distributions = ["debian"],
    deps = ["//third_party/protobuf:protoc_lib"],
)
