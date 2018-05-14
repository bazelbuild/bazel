package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            ".*",
            "*~",
        ],
    ),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "runfiles.cc",
    ],
    visibility = ["//tools:__pkg__"],
)

cc_library(
    name = "runfiles",
    testonly = 1,
    srcs = ["runfiles.cc"],
    hdrs = ["runfiles.h"],
)

cc_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":runfiles",
        "//src/main/cpp/util:file",
        "@com_google_googletest//:gtest_main",
    ],
)
