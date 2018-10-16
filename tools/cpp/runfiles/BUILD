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
    visibility = ["//tools/cpp:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":srcs_for_embedded_tools",
    ],
    visibility = ["//tools/cpp:__pkg__"],
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
)

# As of 2018-10-01, singlejar needs to temporarily use the C++ runfiles library
# from source, not from @bazel_tools.
# Nothing else should depend on this target.
#
# TODO(laszlocsomor): remove this library after Bazel is released with
# https://github.com/bazelbuild/bazel/commit/23bc3bee79d8a8b8dc15bbfb6072ec9f965dff96.
# Once Bazel is released with that patch, do the following:
# 1. remove this library
# 2. let Singlejar depend on @bazel_tools//tools/cpp/runfiles
# 3. change Singlejar's sources to include "tools/cpp/runfiles.h" instead of
#    "tools/cpp/runfiles_src.h"
cc_library(
    name = "runfiles_src_for_singlejar_only",
    testonly = 1,
    visibility = ["//src/tools/singlejar:__pkg__"],
    deps = [":runfiles"],
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
