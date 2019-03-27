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

# Rewrite the include path for runfiles.h in runfiles_src.cc, and create
# "generated_runfiles.{h,cc}". These files are renamed to "runfiles.{h,cc}" as
# part of creating the embedded tools of Bazel.
#
# We cannot just check in runfiles_src.{h,cc} as runfiles.{h,cc}, because it'd
# cause a header check failure on Windows when building targets in the Bazel
# source tree, if those targets depend on @bazel_tools//tools/cpp/runfiles,
# because due to lack of sandboxing they would accidentally pick up runfiles.h
# from //tools/cpp/runfiles.
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
    # Keep this transformation logic in sync with the
    # //scripts/bootstrap/compile.sh
    cmd = ("sed " +
           "  's|^#include.*/runfiles_src.h.*|#include \"tools/cpp/runfiles/runfiles.h\"|' " +
           "  $(location runfiles_src.cc) > $(location generated_runfiles.cc) && " +
           "cp $(location runfiles_src.h) $(location generated_runfiles.h)"),
)

cc_library(
    name = "for_src_tools_launcher",
    deps = [":runfiles"],
    visibility = ["//src/tools/launcher:__pkg__"],
)

cc_library(
    name = "runfiles",
    srcs = ["runfiles_src.cc"],
    hdrs = ["runfiles_src.h"],
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
