package(default_visibility = ["//src/tools/launcher:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "data_parser",
    srcs = ["data_parser.cc"],
    hdrs = ["data_parser.h"],
    deps = [":util"],
)

cc_library(
    name = "util",
    srcs = ["launcher_util.cc"],
    hdrs = ["launcher_util.h"],
)

cc_test(
    name = "util_test",
    srcs = ["launcher_util_test.cc"],
    deps = [
        ":util",
        "//third_party:gtest",
    ],
)

cc_test(
    name = "data_parser_test",
    srcs = ["data_parser_test.cc"],
    deps = [
        ":data_parser",
        "//third_party:gtest",
    ],
)
