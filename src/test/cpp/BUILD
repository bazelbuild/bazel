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
        "//src:windows": [
            "blaze_util_windows_test.cc",
        ],
        "//src:windows_msys": [
            "blaze_util_test.cc",
            "blaze_util_windows_test.cc",
        ],
        "//src:windows_msvc": [
            "blaze_util_windows_test.cc",
        ],
        "//conditions:default": [
            "blaze_util_test.cc",
            "blaze_util_posix_test.cc",
        ],
    }),
    deps = [
        "//src/main/cpp:blaze_util",
        "//src/main/cpp/util",
        "//third_party:gtest",
    ],
)

cc_test(
    name = "option_processor_test",
    size = "small",
    srcs = ["option_processor_test.cc"],
    tags = ["requires_internet"],
    deps = [
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:option_processor",
        "//src/main/cpp:workspace_layout",
        "//src/main/cpp/util",
        "//third_party:gtest",
        "@com_googlesource_code_re2//:re2",
    ],
)

cc_test(
    name = "startup_options_test",
    size = "small",
    srcs = ["startup_options_test.cc"],
    deps = [
        "//src/main/cpp:blaze_util",
        "//src/main/cpp:startup_options",
        "//src/main/cpp:workspace_layout",
        "//third_party:gtest",
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
        "//third_party:gtest",
    ],
)

test_suite(name = "all_tests")
