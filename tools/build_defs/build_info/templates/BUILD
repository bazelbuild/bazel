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

exports_files(
    [
        "non_volatile_file.h.template",
        "volatile_file.h.template",
        "redacted_file.h.template",
        "non_volatile_file.properties.template",
        "volatile_file.properties.template",
        "redacted_file.properties.template",
    ],
    visibility = [
        "@bazel_tools//tools/build_defs/build_info:__pkg__",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*"]),
    visibility = ["//tools/build_defs:__pkg__"],
)
