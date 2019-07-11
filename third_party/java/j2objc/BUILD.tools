# Copyright 2015 The Bazel Authors. All rights reserved.
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

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

package(
    default_visibility = ["//visibility:public"],
)

java_library(
    name = "annotations",
    exports = ["@bazel_j2objc//:annotations"],
)

genrule(
    name = "jre_emul_jar",
    srcs = ["@bazel_j2objc//:jre_emul.jar"],
    outs = ["jre_emul.jar"],
    cmd = "cp $< $@",
)

filegroup(
    name = "jre_emul_module",
    srcs = ["@bazel_j2objc//:jre_emul_module"],
)

objc_library(
    name = "jre_core_lib",
    deps = ["@bazel_j2objc//:jre_core_lib"],
)

filegroup(
    name = "proto_plugin",
    srcs = ["@bazel_j2objc//:proto_plugin"],
)

objc_library(
    name = "proto_runtime",
    deps = ["@bazel_j2objc//:proto_runtime"],
)
