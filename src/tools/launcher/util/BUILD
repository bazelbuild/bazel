load("//src/tools/launcher:win_rules.bzl", "CC_17_COPT", "win_cc_library", "win_cc_test")

package(default_visibility = ["//src/tools/launcher:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

win_cc_library(
    name = "data_parser",
    srcs = ["data_parser.cc"],
    hdrs = ["data_parser.h"],
    copts = CC_17_COPT,
    deps = [":util"],
)

win_cc_library(
    name = "util",
    srcs = ["launcher_util.cc"],
    hdrs = ["launcher_util.h"],
    copts = CC_17_COPT,
    deps = ["//src/main/cpp/util:filesystem"],
)

win_cc_test(
    name = "util_test",
    srcs = ["launcher_util_test.cc"],
    copts = CC_17_COPT,
    deps = [
        ":util",
        "//src/main/native/windows:lib-file",
        "@bazel_tools//tools/cpp/runfiles",
        "@com_google_googletest//:gtest_main",
    ],
)

win_cc_test(
    name = "data_parser_test",
    srcs = ["data_parser_test.cc"],
    copts = CC_17_COPT,
    deps = [
        ":data_parser",
        "@com_google_googletest//:gtest_main",
    ],
)
