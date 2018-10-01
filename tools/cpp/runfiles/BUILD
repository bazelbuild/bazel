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
        ":srcs_for_embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)

genrule(
    name = "srcs_for_embedded_tools",
    srcs = [
        "runfiles_src.cc",
        "runfiles_src.h",
    ],
    outs = [
        "generated_runfiles.cc",
        "generated_runfiles.h",
    ],
    cmd = ("sed " +
           "  's|^#include.*/runfiles_src.h.*|#include \"tools/cpp/runfiles/runfiles.h\"|' " +
           "  $(location runfiles_src.cc) > $(location generated_runfiles.cc) && " +
           "cp $(location runfiles_src.h) $(location generated_runfiles.h)"),
)

cc_library(
    name = "runfiles",
    testonly = 1,
    srcs = ["runfiles_src.cc"],
    hdrs = ["runfiles_src.h"],
    # TODO(laszlocsomor) Use @bazel_tools//tools/cpp/runfiles in singlejar
    visibility = ["//src/tools/singlejar:__pkg__"],
)

cc_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.cc"],
    visibility = ["//visibility:public"],
    deps = [
        ":runfiles",
        "//src/main/cpp/util:filesystem",
        "@com_google_googletest//:gtest_main",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)
