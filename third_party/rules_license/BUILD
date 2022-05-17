# Copyright 2022 The Bazel Authors. All rights reserved.
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

load("@rules_license//rules:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "@rules_license//tools:srcs",
        "@rules_license//rules:srcs",
        "@rules_license//licenses/spdx:srcs",
    ],
    visibility = ["@io_bazel//third_party:__pkg__"],
)

license(
    name = "license",
    license_kinds = [
        "@rules_license//licenses/spdx:Apache-2.0",
    ],
    license_text = "LICENSE",
)

exports_files(
    ["WORKSPACE"],
    visibility = ["//visibility:public"],
)

exports_files(
    glob([
        "*.bzl",
    ]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "standard_package",
    srcs = glob([
        "*.bzl",
        "*.md",
    ]) + [
        "BUILD",
        "LICENSE",
    ],
    visibility = ["//distro:__pkg__"],
)
