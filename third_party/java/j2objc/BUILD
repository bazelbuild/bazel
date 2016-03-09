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

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

java_library(
    name = "annotations",
    srcs = ["@bazel_j2objc//:annotations"],
)

objc_library(
    name = "jre_emul_lib",
    sdk_dylibs = [
        "libicucore",
        "libz",
    ],
    sdk_frameworks = [
        "Security",
    ],
    deps = ["@bazel_j2objc//:jre_emul_archive"],
)

# Proto support is a work in progress.
# We put an empty filegroup here so Bazel analysis doesn't fail.
filegroup(
    name = "proto_runtime",
    srcs = [],
)
