# Description:
#   C++ utility tests for Bazel
package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

cc_test(
    name = "windows_jni_test",
    size = "small",
    srcs = select({
        "//src/conditions:windows": [
            "windows/util_test.cc",
            "windows/file_test.cc",
        ],
        "//conditions:default": ["dummy_test.cc"],
    }),
    deps = select({
        "//src/conditions:windows": [
            "//src/main/native/windows:lib-file",
            "//src/test/cpp/util:windows_test_util",
            "@com_google_googletest//:gtest_main",
        ],
        "//conditions:default": [],
    }),
)

test_suite(name = "all_tests")

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
