filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/cpp:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "runfiles.cc",
        "runfiles.h",
    ],
    visibility = ["//tools/cpp:__pkg__"],
)

cc_library(
    name = "runfiles",
    srcs = ["runfiles.cc"],
    hdrs = ["runfiles.h"],
)

cc_test(
    name = "runfiles-test",
    srcs = ["runfiles_test.cc"],
    deps = [
        ":runfiles",
        "//src/main/cpp/util:file",
        "//third_party:gtest",
    ],
)
