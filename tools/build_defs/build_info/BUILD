# Copyright 2023 The Bazel Authors. All rights reserved.
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

load(":bazel_cc_build_info.bzl", "bazel_cc_build_info")
load(":bazel_java_build_info.bzl", "bazel_java_build_info")

package(default_visibility = ["//visibility:public"])

bazel_cc_build_info(
    name = "cc_build_info",
)

bazel_java_build_info(
    name = "java_build_info",
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*"]),
)
