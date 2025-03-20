# Description:
#   C++ utility tests for Bazel
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + ["//src/test/cpp/util:srcs"],
    visibility = ["//src:__pkg__"],
)

cc_test(
    name = "blaze_util_test",
    srcs = select({
        "//src/conditions:windows": [
            "blaze_util_windows_test.cc",
        ],
        "//conditions:default": [
            "blaze_util_posix_test.cc",
            "blaze_util_test.cc",
        ],
    }),
    deps = [
        "//src/main/cpp:blaze_util",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "option_processor_test",
    size = "small",
    srcs = ["option_processor_test.cc"],
    tags = ["requires_internet"],
    deps = [
        "//src/main/cpp:bazel_startup_options",
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:option_processor",
        "//src/main/cpp:workspace_layout",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rc_file_test",
    size = "small",
    srcs = ["rc_file_test.cc"],
    tags = ["requires_internet"],
    deps = [
        "//src/main/cpp:bazel_startup_options",
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:option_processor",
        "//src/main/cpp:rc_file",
        "//src/main/cpp:workspace_layout",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rc_options_test",
    size = "small",
    srcs = ["rc_options_test.cc"],
    deps = [
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:option_processor",
        "//src/main/cpp:workspace_layout",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "test_util",
    testonly = 1,
    srcs = ["test_util.cc"],
    hdrs = ["test_util.h"],
    deps = [
        "//src/main/cpp:startup_options",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "startup_options_test",
    size = "small",
    srcs = ["startup_options_test.cc"],
    deps = [
        ":test_util",
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:startup_options",
        "//src/main/cpp:workspace_layout",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "//src/conditions:linux": ["//src/main/cpp/util"],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "bazel_startup_options_test",
    size = "small",
    srcs = ["bazel_startup_options_test.cc"],
    deps = [
        ":test_util",
        "//src/main/cpp:bazel_startup_options",
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:workspace_layout",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "workspace_layout_test",
    size = "small",
    srcs = ["workspace_layout_test.cc"],
    deps = [
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:workspace_layout",
        "//src/main/cpp/util",
        "@com_google_googletest//:gtest_main",
    ],
)

test_suite(name = "all_tests")
