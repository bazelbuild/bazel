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

load("//tools/distributions:distribution_rules.bzl", "distrib_cc_binary")

licenses(["notice"])  # Apache v2

package(default_visibility = ["//visibility:public"])

distrib_cc_binary(
    name = "grpc-java-plugin",
    enable_distributions = ["debian"],
    srcs = [
        "src/java_plugin/cpp/java_generator.cpp",
        "src/java_plugin/cpp/java_generator.h",
        "src/java_plugin/cpp/java_plugin.cpp",
    ],
    copts = ["-w"],
    deps = ["//third_party/protobuf:protoc_lib"],
)
