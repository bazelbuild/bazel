# Copyright 2024 The Bazel Authors. All rights reserved.
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

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
    ),
    visibility = ["//tools/cpp:__pkg__"],
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
)

COPTS = select({
    ":windows": ["/std:c++17"],
    "//conditions:default": ["-std=c++17"],
})

cc_library(
    name = "common",
    srcs = [
        "common/common.cc",
    ],
    hdrs = [
        "common/common.h",
        "common/json.hpp",
    ],
    copts = COPTS,
    includes = ["."],
)

cc_library(
    name = "aggregate-ddi-lib",
    srcs = ["aggregate-ddi/aggregate-ddi.cc"],
    hdrs = ["aggregate-ddi/aggregate-ddi.h"],
    copts = COPTS,
    deps = [":common"],
)

cc_binary(
    name = "aggregate-ddi",
    srcs = ["aggregate-ddi/main.cc"],
    copts = COPTS,
    deps = [
        ":aggregate-ddi-lib",
    ],
)

cc_library(
    name = "generate-modmap-lib",
    srcs = ["generate-modmap/generate-modmap.cc"],
    hdrs = ["generate-modmap/generate-modmap.h"],
    copts = COPTS,
    deps = [":common"],
)

cc_binary(
    name = "generate-modmap",
    srcs = ["generate-modmap/main.cc"],
    copts = COPTS,
    deps = [":generate-modmap-lib"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":aggregate-ddi",
        ":generate-modmap",
    ],
    visibility = ["//tools/cpp:__pkg__"],
)

cc_test(
    name = "generate-modmap_test",
    srcs = ["generate-modmap/generate-modmap_test.cc"],
    copts = COPTS,
    deps = [
        ":generate-modmap-lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "aggregate-ddi_test",
    srcs = ["aggregate-ddi/aggregate-ddi_test.cc"],
    copts = COPTS,
    deps = [
        ":aggregate-ddi-lib",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "common_test",
    srcs = ["common/common_test.cc"],
    copts = COPTS,
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "json_test",
    srcs = ["common/json_test.cc"],
    copts = COPTS,
    deps = [
        ":common",
        "@com_google_googletest//:gtest_main",
    ],
)
